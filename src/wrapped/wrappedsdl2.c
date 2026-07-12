#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#include <dlfcn.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <time.h>

#include "wrappedlibs.h"

#include "debug.h"
#include "wrapper.h"
#include "bridge.h"
#include "callback.h"
#include "librarian.h"
#include "elfloader.h"
#include "librarian/library_private.h"
#include "emu/x64emu_private.h"
#include "box64context.h"
#include "sdl2rwops.h"
#include "myalign.h"
#include "threads.h"
#include "gltools.h"

#include "generated/wrappedsdl2defs.h"

const char* sdl2Name = "libSDL2-2.0.so.0";
#define LIBNAME sdl2
static void* my_glhandle = NULL;
// Real static SDL_CreateWindow (captured in my2_SDL_DYNAPI_entry).  my2_SDL_CreateWindow
// strips SDL_WINDOW_OPENGL and calls this so the dummy video driver creates a valid
// window WITHOUT attempting (impossible) GL init; the GL context comes from ZFA/EGL.
static uintptr_t g_real_sdl_createwindow = 0;
static uintptr_t g_real_sdl_gl_loadlibrary = 0;  // real static SDL_GL_LoadLibrary (observe-only passthrough)
// Host dlopen handle to the GL provider (libgl4es.so), used to resolve GL
// function addresses when there is no real ARM64 libSDL2 to supply
// SDL_GL_GetProcAddress.  Without this, getGLProcAddress() would call a NULL
// procaddr (my->SDL_GL_GetProcAddress) and crash (SIGSEGV @0x0) the moment
// Unity starts loading OpenGL entry points after creating the GL context.
static void* g_gl4es_host_handle = NULL;
// For ZINK_ZFA: libzfa.so is loaded by rimdroid.c into the rimdroid namespace
// (so Zink can reach the Vulkan loader/driver).  A plain dlopen() here would
// fail to link those, so resolve GL entry points from that inherited handle.
extern __attribute__((weak)) void* g_zfa_handle;
// RD_SOFTPIPE: libOSMesa.so handle (set by rimdroid.c's rimdroid_init_osmesa).
// libOSMesa exports the full GL API as plain symbols, so a straight dlsym
// resolves every entry point softpipe provides (proven by the Milestone 1 smoke
// test). A miss means the symbol genuinely is absent → the GetProcAddress wrapper
// falls back to its zeroing/no-op stubs, same as for ZFA.
extern __attribute__((weak)) void* g_osmesa_handle;
static void* rimdroid_gl_proc_resolver(const char* name)
{
    if (&g_osmesa_handle && g_osmesa_handle) {
        return dlsym(g_osmesa_handle, name);
    }
    if (&g_zfa_handle && g_zfa_handle) {
        void* p = dlsym(g_zfa_handle, name);
        if (p) return p;
        // Mesa/Zink exposes CORE GL entry points (e.g. glGetInternalformativ
        // @4.2, glGetQueryObjectui64v @3.3) via eglGetProcAddress even when they
        // are NOT exported as plain dlsym symbols of libzfa.so.  Returning the
        // REAL function (instead of a zeroing stub) is critical: Unity calls
        // glGetInternalformativ to test render-target format support; a stub
        // returning 0 makes Unity believe formats are unsupported → it tears the
        // GfxDevice down and retries forever (the SDL_GL_DeleteContext(NULL) loop).
        static void* (*egl_gpa)(const char*) = NULL;
        static int egl_checked = 0;
        if (!egl_checked) {
            egl_checked = 1;
            egl_gpa = (void*(*)(const char*))dlsym(g_zfa_handle, "eglGetProcAddress");
        }
        if (egl_gpa) {
            void* q = egl_gpa(name);
            if (q) printf_log(LOG_NONE, "rimdroid_gl_proc_resolver: '%s' via eglGetProcAddress => %p\n", name, q);
            return q;
        }
        return NULL;
    }
    if (!g_gl4es_host_handle) {
        const char* libgl = BOX64ENV(libgl) ? BOX64ENV(libgl) : "libGL.so.1";
        g_gl4es_host_handle = dlopen(libgl, RTLD_LAZY | RTLD_GLOBAL);
        if (!g_gl4es_host_handle && strcmp(libgl, "libgl4es.so") != 0)
            g_gl4es_host_handle = dlopen("libgl4es.so", RTLD_LAZY | RTLD_GLOBAL);
        if (!g_gl4es_host_handle)
            printf_log(LOG_NONE, "rimdroid_gl_proc_resolver: cannot dlopen GL lib '%s' (%s)\n", libgl, dlerror());
    }
    if (!g_gl4es_host_handle)
        return NULL;
    void* p = dlsym(g_gl4es_host_handle, name);
    if (!p) {
        // Core GL names not exported directly — try GL4ES's own resolver.
        static void* (*gl4es_gpa)(const char*) = NULL;
        static int gpa_checked = 0;
        if (!gpa_checked) {
            gpa_checked = 1;
            gl4es_gpa = (void*(*)(const char*))dlsym(g_gl4es_host_handle, "gl4es_GetProcAddress");
        }
        if (gl4es_gpa)
            p = gl4es_gpa(name);
    }
    printf_log(LOG_DEBUG, "GL proc resolver('%s') handle=%p => %p\n", name, g_gl4es_host_handle, p);
    return p;
}

// No-op GL entry point returned for names libzfa.so does not export.  Mesa/Zink
// advertises some functions as core (by GL version) or via extensions, but the
// static libzfa lacks the actual symbol (e.g. glGetInternalformativ core@4.2,
// glGetQueryObjectui64v core@3.3).  Unity loads core entry points by version and
// calls a few unconditionally; returning NULL makes it jump to 0x0 → crash.
// Returning this stub (RAX=0) turns such calls into harmless no-ops.  Feature
// *selection* in Unity is gated on the GL version / extension string (which we
// control via MESA_*_OVERRIDE), NOT on a non-NULL pointer, so handing back a
// stub does not make Unity wrongly enable an unsupported path.
static int rimdroid_gl_noop(void) { return 0; }

// Getters libzfa.so does not export need MORE than a no-op: they must ZERO the
// caller's output buffer, otherwise Unity reads uninitialised garbage (MSAA
// sample counts, format support, texture params) and miscomputes a texture
// upload → glTexSubImage2D with a bad src pointer → crash.  Real signatures so
// box64's bridge marshals args correctly; the *params pointer is a guest address
// (valid in-process under box64).
static void rd_glGetInternalformativ(uint32_t target, uint32_t internalformat, uint32_t pname, int32_t count, int32_t* params) {
    (void)target;
    // PLAN B: libzfa doesn't export this (core GL4.2) and eglGetProcAddress
    // fallback didn't resolve it.  Returning 0 made Unity believe render-target
    // formats are UNSUPPORTED → it tore the GfxDevice down + retried forever
    // (the SDL_GL_DeleteContext(NULL) loop).  So LIE that formats are fully
    // supported, instead of zeroing.
    if (!params || count <= 0) return;
    int32_t v;
    switch (pname) {
        case 0x826F: v = 1;       break; // GL_INTERNALFORMAT_SUPPORTED → GL_TRUE
        case 0x8270: v = (int32_t)internalformat; break; // GL_INTERNALFORMAT_PREFERRED → echo
        case 0x9380: v = 1;       break; // GL_NUM_SAMPLE_COUNTS → 1
        case 0x80A9: v = 1;       break; // GL_SAMPLES → 1
        case 0x8286: case 0x8287: case 0x8288:          // COLOR/DEPTH/STENCIL_RENDERABLE
        case 0x8289: case 0x828A:                        // FRAMEBUFFER_RENDERABLE[_LAYERED]
        case 0x828B: case 0x828C: case 0x828D:          // FRAMEBUFFER_BLEND/READ_PIXELS[_FORMAT]
        case 0x827F: case 0x8280: case 0x8281: case 0x8282: case 0x8283: // GET/TEX image format/type
            v = 0x82B7; break;            // GL_FULL_SUPPORT
        case 0x826E: v = 1;       break; // GL_INTERNALFORMAT_PREFERRED-ish / supported
        default:     v = 1;       break; // default to "supported/yes" rather than 0
    }
    for (int32_t i = 0; i < count; ++i) params[i] = v;
    static int n=0; if(n<24){n++; printf_log(LOG_NONE, "RIMDROID glGetInternalformativ ifmt=0x%x pname=0x%x => %d\n", internalformat, pname, v);}
}
// Resolve a REAL Zink GL entry point (dlsym + eglGetProcAddress fallback).
static void* rd_zfa_gl(const char* n) { return rimdroid_gl_proc_resolver(n); }

// ---- RimDroid 1.6 sync-poll shim (AI-brief v15) -----------------------------
// Unity polls GLsync objects with zero-timeout waits during the texture-atlas
// bake. On Zink this can turn a poll loop into thousands of tiny flush/batch
// states per second. Keep the first real driver query, then coalesce repeated
// polls of the same still-unsignaled sync for a short window.
#define RD_GL_SYNC_FLUSH_COMMANDS_BIT 0x00000001u
#define RD_GL_TIMEOUT_EXPIRED        0x911Bu
#define RD_GL_WAIT_FAILED            0x911Du

static uint32_t (*p_rd_real_glClientWaitSync)(void*, uint32_t, uint64_t) = NULL;
static void (*p_rd_real_glDeleteSync)(void*) = NULL;

static __thread void* rd_sync_last = NULL;
static __thread uint64_t rd_sync_last_ns = 0;
static __thread uint32_t rd_sync_last_flags = 0;
static __thread uint32_t rd_sync_last_result = RD_GL_TIMEOUT_EXPIRED;

static uint64_t rd_now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t rd_sync_poll_window_ns(void)
{
    static int init = 0;
    static uint64_t window_ns = 1000000ull; // default: 1ms
    if (!init) {
        init = 1;
        const char* e = getenv("RIMDROID_GL_SYNC_POLL_US");
        if (e) {
            char* end = NULL;
            unsigned long long us = strtoull(e, &end, 0);
            if (end != e)
                window_ns = us * 1000ull;
        }
    }
    return window_ns;
}

static uint32_t rd_glClientWaitSync(void* sync, uint32_t flags, uint64_t timeout)
{
    if (!p_rd_real_glClientWaitSync)
        return RD_GL_WAIT_FAILED;

    const uint64_t window_ns = rd_sync_poll_window_ns();
    if (sync && timeout == 0 && window_ns) {
        uint64_t now = rd_now_ns();
        int needs_first_flush = (flags & RD_GL_SYNC_FLUSH_COMMANDS_BIT) &&
                                !(rd_sync_last_flags & RD_GL_SYNC_FLUSH_COMMANDS_BIT);
        if (now && sync == rd_sync_last && rd_sync_last_result == RD_GL_TIMEOUT_EXPIRED &&
            rd_sync_last_ns && now - rd_sync_last_ns < window_ns && !needs_first_flush) {
            static uint64_t skipped = 0;
            skipped++;
            if ((skipped & (skipped - 1)) == 0 || (skipped % 50000ull) == 0) {
                printf_log(LOG_NONE, "RIMDROID GLSYNC coalesced %llu zero-timeout glClientWaitSync polls (window=%lluus)\n",
                           (unsigned long long)skipped,
                           (unsigned long long)(window_ns / 1000ull));
                fflush(NULL);
            }
            return RD_GL_TIMEOUT_EXPIRED;
        }
        rd_sync_last_ns = now;
    }

    {
        static uint64_t rd_wait_total = 0;
        rd_wait_total++;
        if ((rd_wait_total % 10000) == 1)
            { printf_log(LOG_NONE, "RIMDROID SYNCSTAT wait_total=%llu timeout=%llu\n", (unsigned long long)rd_wait_total, (unsigned long long)timeout); fflush(NULL); }
    }
    uint32_t ret = p_rd_real_glClientWaitSync(sync, flags, timeout);
    if (sync && timeout == 0) {
        rd_sync_last = sync;
        rd_sync_last_flags = flags;
        rd_sync_last_result = ret;
    }
    return ret;
}

static void rd_glDeleteSync(void* sync)
{
    if (sync == rd_sync_last) {
        rd_sync_last = NULL;
        rd_sync_last_ns = 0;
        rd_sync_last_flags = 0;
        rd_sync_last_result = RD_GL_TIMEOUT_EXPIRED;
    }
    if (p_rd_real_glDeleteSync)
        p_rd_real_glDeleteSync(sync);
}

// ---- RimDroid 1.6 arg-sanity shims (AI-brief v13) ----------------------------
// At the splash-unload frame ONE GL call carries a garbage/huge size: with
// ARB_buffer_storage it GPU-faulted (DEVICE_LOST on both drivers); with the
// extension hidden the same frame dies in an "mmap failed: Out of memory" loop
// (CPU staging of the garbage size). These shims log anomalous sizes to NAME the
// call. Threshold 64MB — RimWorld's legit buffers are far smaller.
#define RD_GL_SANE_SIZE (64ull*1024*1024)
static void rd_upload_pace(uint64_t sz);   // defined below with the texture accounting
// ---- GL op-logger (device-lost culprit hunt, 2026-07-11) --------------------------
// Every death has IDENTICAL counters at the last 256MB-crossing print (sub=13623 etc.) →
// the guilty command sits at a fixed position in the deterministic upload stream. Gate on
// the subimage counter: past RIMDROID_GL_LOG_AFTER_SUB, EVERY shimmed GL call is printed
// with a sequence number, and pacing (RIMDROID_PACE_MB, e.g. 8) adds a glFinish per flush
// in the armed zone — the last "pace finish OK" exonerates everything before it, so the
// culprit is among the handful of ops logged after it.
static uint64_t rd_sub_calls;              // real definition below with the sub shims
static uint64_t rd_gl_op_seq = 0;
static int rd_oplog_armed_logged = 0;
static int rd_gl_oplog_on(void) {
    static int64_t after = -2;
    if (after == -2) {
        const char* e = getenv("RIMDROID_GL_LOG_AFTER_SUB");
        after = (e && e[0]) ? atoll(e) : -1;
    }
    rd_gl_op_seq++;
    if (after < 0 || (int64_t)rd_sub_calls < after) return 0;
    if (!rd_oplog_armed_logged) {
        rd_oplog_armed_logged = 1;
        printf_log(LOG_NONE, "RIMDROID OPLOG armed at sub=%llu seq=%llu\n",
                   (unsigned long long)rd_sub_calls, (unsigned long long)rd_gl_op_seq);
        fflush(NULL);
    }
    return 1;
}
#define RD_OPLOG(...) do { if (rd_gl_oplog_on()) { printf_log(LOG_NONE, __VA_ARGS__); fflush(NULL); } } while(0)
static void (*p_rd_real_glBufferStorage)(uint32_t,int64_t,const void*,uint32_t) = NULL;
static void (*p_rd_real_glBufferData)(uint32_t,int64_t,const void*,uint32_t) = NULL;
static void (*p_rd_real_glBufferSubData)(uint32_t,int64_t,int64_t,const void*) = NULL;
static void* (*p_rd_real_glMapBufferRange)(uint32_t,int64_t,int64_t,uint32_t) = NULL;
static uint64_t rd_buf_total = 0, rd_buf_calls = 0;
static void rd_glBufferStorage(uint32_t target, int64_t size, const void* data, uint32_t flags) {
    if (size > 0) { rd_buf_total += (uint64_t)size; rd_buf_calls++; }
    if ((uint64_t)size > RD_GL_SANE_SIZE)
        { printf_log(LOG_NONE, "RIMDROID GLSANITY glBufferStorage target=0x%x size=%lld flags=0x%x", target, (long long)size, flags); fflush(NULL); }
    if (p_rd_real_glBufferStorage) p_rd_real_glBufferStorage(target, size, data, flags);
    if (size > 0) rd_upload_pace((uint64_t)size);
}
static void rd_glBufferData(uint32_t target, int64_t size, const void* data, uint32_t usage) {
    if (size > 0) { rd_buf_total += (uint64_t)size; rd_buf_calls++; }
    if ((uint64_t)size > RD_GL_SANE_SIZE)
        { printf_log(LOG_NONE, "RIMDROID GLSANITY glBufferData target=0x%x size=%lld data=%p usage=0x%x\n", target, (long long)size, data, usage); fflush(NULL); }
    if (p_rd_real_glBufferData) p_rd_real_glBufferData(target, size, data, usage);
    if (size > 0) rd_upload_pace((uint64_t)size);
}
static void rd_glBufferSubData(uint32_t target, int64_t offset, int64_t size, const void* data) {
    if ((uint64_t)size > RD_GL_SANE_SIZE || offset < 0)
        { printf_log(LOG_NONE, "RIMDROID GLSANITY glBufferSubData target=0x%x offset=%lld size=%lld data=%p\n", target, (long long)offset, (long long)size, data); fflush(NULL); }
    if (p_rd_real_glBufferSubData) p_rd_real_glBufferSubData(target, offset, size, data);
    if (size > 0) rd_upload_pace((uint64_t)size);
}
static void* rd_glMapBufferRange(uint32_t target, int64_t offset, int64_t length, uint32_t access) {
    if ((uint64_t)length > RD_GL_SANE_SIZE || offset < 0)
        { printf_log(LOG_NONE, "RIMDROID GLSANITY glMapBufferRange target=0x%x offset=%lld length=%lld access=0x%x\n", target, (long long)offset, (long long)length, access); fflush(NULL); }
    return p_rd_real_glMapBufferRange ? p_rd_real_glMapBufferRange(target, offset, length, access) : NULL;
}
// Texture-family sanity: the death frame is RimWorld's atlas bake, and Turnip dies on a GPU-BO
// mmap ENOMEM ("mmap failed:" from freedreno_bo.c). Log EVERY allocation >= 2048px (capped) and
// flag insane dims, to see the atlas size stream that exhausts kgsl mappable memory.
static void (*p_rd_real_glTexStorage2D)(uint32_t,int32_t,uint32_t,int32_t,int32_t) = NULL;
static void (*p_rd_real_glTexImage2D)(uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*) = NULL;
static void (*p_rd_real_glRenderbufferStorageMultisample)(uint32_t,int32_t,uint32_t,int32_t,int32_t) = NULL;
static int rd_texlog_n = 0;
// Cumulative texture-memory estimate: individual sizes are sane, so test the AGGREGATE
// (kgsl BO mmap dies with ENOMEM). bpp: compressed (DXT/BPTC 0x83xx/0x8e8x) ~1B/px, else 4B/px;
// mip chains ~×4/3. Logs at every +256MB crossing.
static uint64_t rd_tex_total = 0;
static uint64_t rd_rb_total = 0;
// UPLOAD PACING (the fix for the 3GB kgsl cap): RimWorld 1.6's atlas bake uploads ~1.3GB of
// textures inside ONE frame — zink keeps every staging BO alive until a flush, so peak GPU-BO
// usage doubles+ and slams kgsl's ~3GB per-process limit ("kgsl-3d0" = 3.00GB in /proc/maps at
// death) → mmap ENOMEM → device lost. Force a glFlush every ~192MB of accounted allocations
// (staging gets submitted+recycled) and a full glFinish every ~768MB (hard reclaim).
static uint64_t rd_flush_acc = 0, rd_finish_acc = 0;
// Reset the pacing accumulator at every present: pacing exists to split a GIANT single-frame
// upload burst (the 1.6 atlas bake records >1GB before the first swap). Uploads spread across
// normal frames are already submitted by the per-frame flush — without this reset, steady
// gameplay traffic (Unity's dynamic font/UI atlas glTexSubImage2D) crossed the 192MB threshold
// every few seconds and the forced mid-frame glFlush showed up as a periodic hitch.
void rd_upload_pace_frame_reset(void) { rd_flush_acc = 0; }
static void rd_upload_pace(uint64_t sz) {
    static void (*p_flush)(void) = NULL; static void (*p_finish)(void) = NULL; static int init = 0;
    static uint64_t pace_bytes = 0;
    if (!pace_bytes) {
        const char* e = getenv("RIMDROID_PACE_MB");
        pace_bytes = (e && e[0] && atoi(e) > 0) ? ((uint64_t)atoi(e) << 20) : (192ull << 20);
    }
    rd_flush_acc += sz; rd_finish_acc += sz;
    if (rd_flush_acc < pace_bytes) return;
    if (!init) { init = 1; p_flush = rd_zfa_gl("glFlush"); p_finish = rd_zfa_gl("glFinish"); }
    rd_flush_acc = 0;
    // Culprit hunt: in the armed op-log zone, a finish per pacing flush pins the guilty
    // window — the last "pace finish OK" proves the GPU was alive and done at that point.
    if (rd_oplog_armed_logged && p_finish) {
        p_finish();
        printf_log(LOG_NONE, "RIMDROID pace finish OK @sub=%llu seq=%llu\n",
                   (unsigned long long)rd_sub_calls, (unsigned long long)rd_gl_op_seq);
        fflush(NULL);
        rd_finish_acc = 0;
        return;
    }
    if (0 && p_finish) {   /* v14 test verdict: finish+flushsync = 2MB-slab alloc storm on the flush thread, net 3.2GB in seconds, died at 30s. Reverted. */
        rd_finish_acc = 0;
        p_finish();
        printf_log(LOG_NONE, "RIMDROID GLSANITY pacing glFinish (total=%lluMB)\n", (unsigned long long)(rd_tex_total>>20)); fflush(NULL);
        return;
    }
    // glFlush ONLY: it alone keeps kgsl under the ~3GB cap (mmap fails went 82 -> 0). A mid-bake
    // glFinish crashed zink (libzfa+0xdc8e1c, NULL+0x30) exactly at the first 768MB threshold —
    // do not force full waits during the upload storm.
    if (p_flush) {
        p_flush();
        printf_log(LOG_NONE, "RIMDROID GLSANITY pacing glFlush (total=%lluMB)\n", (unsigned long long)(rd_tex_total>>20)); fflush(NULL);
    }
}
static uint64_t rd_tex_calls = 0;
static void rd_tex_account(uint32_t ifmt, int32_t levels, int32_t w, int32_t h) {
    if (w <= 0 || h <= 0) return;
    rd_tex_calls++;
    uint64_t bpp = ((ifmt & 0xff00) == 0x8300 || (ifmt & 0xfff0) == 0x8e80) ? 1 : 4;
    uint64_t sz = (uint64_t)w * h * bpp;
    if (levels > 1) sz = sz * 4 / 3;
    uint64_t before = rd_tex_total / (256ull*1024*1024);
    rd_tex_total += sz;
    if (rd_tex_total / (256ull*1024*1024) != before)
        { printf_log(LOG_NONE, "RIMDROID GLSANITY cumulative tex=%lluMB calls=%llu buf=%lluMB bufcalls=%llu rb=%lluMB\n", (unsigned long long)(rd_tex_total/1048576), (unsigned long long)rd_tex_calls, (unsigned long long)(rd_buf_total/1048576), (unsigned long long)rd_buf_calls, (unsigned long long)(rd_rb_total/1048576)); fflush(NULL); }
    rd_upload_pace(sz);
}
static void rd_glTexStorage2D(uint32_t target, int32_t levels, uint32_t ifmt, int32_t w, int32_t h) {
    RD_OPLOG("RIMDROID OP#%llu TexStorage2D lvls=%d ifmt=0x%x %dx%d\n", (unsigned long long)rd_gl_op_seq, levels, ifmt, w, h);
    rd_tex_account(ifmt, levels, w, h);
    if ((w >= 2048 || h >= 2048 || w < 0 || h < 0 || levels > 16) && rd_texlog_n < 48)
        { rd_texlog_n++; printf_log(LOG_NONE, "RIMDROID GLSANITY glTexStorage2D target=0x%x levels=%d ifmt=0x%x %dx%d\n", target, levels, ifmt, w, h); fflush(NULL); }
    if (p_rd_real_glTexStorage2D) p_rd_real_glTexStorage2D(target, levels, ifmt, w, h);
}
static void rd_glTexImage2D(uint32_t target, int32_t level, int32_t ifmt, int32_t w, int32_t h, int32_t border, uint32_t fmt, uint32_t type, const void* px) {
    if (level == 0) rd_tex_account((uint32_t)ifmt, 1, w, h);
    if ((w >= 2048 || h >= 2048 || w < 0 || h < 0) && rd_texlog_n < 48)
        { rd_texlog_n++; printf_log(LOG_NONE, "RIMDROID GLSANITY glTexImage2D target=0x%x level=%d ifmt=0x%x %dx%d\n", target, level, ifmt, w, h); fflush(NULL); }
    if (p_rd_real_glTexImage2D) p_rd_real_glTexImage2D(target, level, ifmt, w, h, border, fmt, type, px);
}
static void (*p_rd_real_glTexStorage3D)(uint32_t,int32_t,uint32_t,int32_t,int32_t,int32_t) = NULL;
static void rd_glTexStorage3D(uint32_t target, int32_t levels, uint32_t ifmt, int32_t w, int32_t h, int32_t d) {
    if (d > 0) { for (int i = 0; i < d; i++) rd_tex_account(ifmt, levels, w, h); }
    if (rd_texlog_n < 96 && (w >= 1024 || h >= 1024 || d >= 8))
        { rd_texlog_n++; printf_log(LOG_NONE, "RIMDROID GLSANITY glTexStorage3D target=0x%x levels=%d ifmt=0x%x %dx%dx%d\n", target, levels, ifmt, w, h, d); fflush(NULL); }
    if (p_rd_real_glTexStorage3D) p_rd_real_glTexStorage3D(target, levels, ifmt, w, h, d);
}
static void (*p_rd_real_glTexImage3D)(uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*) = NULL;
static void rd_glTexImage3D(uint32_t target, int32_t level, int32_t ifmt, int32_t w, int32_t h, int32_t d, int32_t border, uint32_t fmt, uint32_t type, const void* px) {
    if (level == 0 && d > 0) { for (int i = 0; i < d; i++) rd_tex_account((uint32_t)ifmt, 1, w, h); }
    if (p_rd_real_glTexImage3D) p_rd_real_glTexImage3D(target, level, ifmt, w, h, d, border, fmt, type, px);
}
static void (*p_rd_real_glCompressedTexImage2D)(uint32_t,int32_t,uint32_t,int32_t,int32_t,int32_t,int32_t,const void*) = NULL;
static void rd_glCompressedTexImage2D(uint32_t target, int32_t level, uint32_t ifmt, int32_t w, int32_t h, int32_t border, int32_t imageSize, const void* data) {
    if (level == 0 && imageSize > 0) { rd_tex_total += (uint64_t)imageSize; rd_upload_pace((uint64_t)imageSize); }
    if (p_rd_real_glCompressedTexImage2D) p_rd_real_glCompressedTexImage2D(target, level, ifmt, w, h, border, imageSize, data);
}
static void (*p_rd_real_glRenderbufferStorage)(uint32_t,uint32_t,int32_t,int32_t) = NULL;
static void rd_glRenderbufferStorage(uint32_t target, uint32_t ifmt, int32_t w, int32_t h) {
    if (w > 0 && h > 0) rd_rb_total += (uint64_t)w*h*4;
    if (p_rd_real_glRenderbufferStorage) p_rd_real_glRenderbufferStorage(target, ifmt, w, h);
}
static void rd_glRenderbufferStorageMultisample(uint32_t target, int32_t samples, uint32_t ifmt, int32_t w, int32_t h) {
    if (w > 0 && h > 0) rd_rb_total += (uint64_t)w*h*4*(samples>0?samples:1);
    if (rd_texlog_n < 48)
        { rd_texlog_n++; printf_log(LOG_NONE, "RIMDROID GLSANITY glRenderbufferStorageMultisample samples=%d ifmt=0x%x %dx%d\n", samples, ifmt, w, h); fflush(NULL); }
    if (p_rd_real_glRenderbufferStorageMultisample) p_rd_real_glRenderbufferStorageMultisample(target, samples, ifmt, w, h);
}
// COPY PACING (device-lost fix #2): RimWorld's atlas ASSEMBLY records thousands of
// Graphics.CopyTexture GPU->GPU copies into ONE batch (~19s with zero flushes — the upload
// pacing above only counts glTex*/glBuffer* and is blind to copies). The resulting monster
// IB trips kgsl's GPU-hang watchdog (gpufault_procs +1 with NO pagefault) → the context is
// killed → VK_ERROR_DEVICE_LOST at the next vkQueueSubmit. Feed copy sizes into the same
// pacing accumulator so glFlush keeps splitting the batch during the assembly too.
static void (*p_rd_real_glCopyImageSubData)(uint32_t,uint32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t) = NULL;
static void (*p_rd_real_glCopyTexSubImage2D)(uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t) = NULL;
static void (*p_rd_real_glBlitFramebuffer)(int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t) = NULL;
static uint64_t rd_copy_total = 0, rd_copy_calls = 0;
static void rd_copy_account(int64_t w, int64_t h, int64_t d) {
    if (w <= 0 || h <= 0) return;
    if (d <= 0) d = 1;
    // assume 4B/px: overestimating a compressed destination just flushes a little sooner
    uint64_t sz = (uint64_t)(w * h * d) * 4;
    rd_copy_calls++;
    uint64_t before = rd_copy_total >> 28;
    rd_copy_total += sz;
    if ((rd_copy_total >> 28) != before)
        { printf_log(LOG_NONE, "RIMDROID GLSANITY cumulative copy=%lluMB copycalls=%llu\n", (unsigned long long)(rd_copy_total>>20), (unsigned long long)rd_copy_calls); fflush(NULL); }
    rd_upload_pace(sz);
}
// The atlas assembly turned out to be CPU-side page UPLOADS, not GL copies: the 64MB
// host-visible maps during the pre-death burst are glTexSubImage2D/glCompressedTexSubImage2D
// writes into existing 4096 pages (Unity's CopyTexture CPU fallback for DXT). Those entry
// points were unaccounted → the pacing never fired during the burst → monster batch → hang.
static void (*p_rd_real_glTexSubImage2D)(uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*) = NULL;
static void (*p_rd_real_glCompressedTexSubImage2D)(uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,int32_t,const void*) = NULL;
static void (*p_rd_real_glTexSubImage3D)(uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*) = NULL;
static uint64_t rd_sub_total = 0, rd_sub_calls = 0;
static int rd_sublog_n = 0;
static void rd_sub_account(uint64_t sz) {
    rd_sub_calls++;
    uint64_t before = rd_sub_total >> 28;
    rd_sub_total += sz;
    if ((rd_sub_total >> 28) != before)
        { printf_log(LOG_NONE, "RIMDROID GLSANITY cumulative sub=%lluMB subcalls=%llu copy=%lluMB copycalls=%llu\n", (unsigned long long)(rd_sub_total>>20), (unsigned long long)rd_sub_calls, (unsigned long long)(rd_copy_total>>20), (unsigned long long)rd_copy_calls); fflush(NULL); }
    rd_upload_pace(sz);
}
static void rd_glTexSubImage2D(uint32_t target, int32_t level, int32_t xo, int32_t yo, int32_t w, int32_t h, uint32_t fmt, uint32_t type, const void* px) {
    if (w > 0 && h > 0) {
        if (rd_sublog_n < 4) { rd_sublog_n++; printf_log(LOG_NONE, "RIMDROID GLSANITY glTexSubImage2D level=%d %dx%d fmt=0x%x\n", level, w, h, fmt); fflush(NULL); }
        RD_OPLOG("RIMDROID OP#%llu TexSubImage2D lvl=%d %d,%d %dx%d fmt=0x%x\n", (unsigned long long)rd_gl_op_seq, level, xo, yo, w, h, fmt);
        rd_sub_account((uint64_t)w * h * 4);
    }
    // Self-resolve fallback: this shim is also installed via the SDL_GL_GetProcAddress
    // special-case path, which does not populate p_rd_real (see the AddBridge call below).
    if (!p_rd_real_glTexSubImage2D && &g_zfa_handle && g_zfa_handle)
        p_rd_real_glTexSubImage2D = (void(*)(uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*))dlsym(g_zfa_handle, "glTexSubImage2D");
    if (p_rd_real_glTexSubImage2D) p_rd_real_glTexSubImage2D(target, level, xo, yo, w, h, fmt, type, px);
}
static void rd_glCompressedTexSubImage2D(uint32_t target, int32_t level, int32_t xo, int32_t yo, int32_t w, int32_t h, uint32_t fmt, int32_t imageSize, const void* data) {
    if (imageSize > 0) {
        if (rd_sublog_n < 4) { rd_sublog_n++; printf_log(LOG_NONE, "RIMDROID GLSANITY glCompressedTexSubImage2D level=%d %dx%d fmt=0x%x size=%d\n", level, w, h, fmt, imageSize); fflush(NULL); }
        RD_OPLOG("RIMDROID OP#%llu CompressedTexSubImage2D lvl=%d %d,%d %dx%d fmt=0x%x sz=%d\n", (unsigned long long)rd_gl_op_seq, level, xo, yo, w, h, fmt, imageSize);
        rd_sub_account((uint64_t)imageSize);
    }
    if (p_rd_real_glCompressedTexSubImage2D) p_rd_real_glCompressedTexSubImage2D(target, level, xo, yo, w, h, fmt, imageSize, data);
}
static void rd_glTexSubImage3D(uint32_t target, int32_t level, int32_t xo, int32_t yo, int32_t zo, int32_t w, int32_t h, int32_t d, uint32_t fmt, uint32_t type, const void* px) {
    if (w > 0 && h > 0 && d > 0) rd_sub_account((uint64_t)w * h * d * 4);
    if (p_rd_real_glTexSubImage3D) p_rd_real_glTexSubImage3D(target, level, xo, yo, zo, w, h, d, fmt, type, px);
}
static void rd_glCopyImageSubData(uint32_t sn, uint32_t st, int32_t sl, int32_t sx, int32_t sy, int32_t sz_, uint32_t dn, uint32_t dt, int32_t dl, int32_t dx, int32_t dy, int32_t dz, int32_t w, int32_t h, int32_t d) {
    RD_OPLOG("RIMDROID OP#%llu CopyImageSubData src=%u lvl=%d %d,%d,%d dst=%u lvl=%d %d,%d,%d %dx%dx%d\n", (unsigned long long)rd_gl_op_seq, sn, sl, sx, sy, sz_, dn, dl, dx, dy, dz, w, h, d);
    rd_copy_account(w, h, d);
    if (p_rd_real_glCopyImageSubData) p_rd_real_glCopyImageSubData(sn, st, sl, sx, sy, sz_, dn, dt, dl, dx, dy, dz, w, h, d);
}
static void rd_glCopyTexSubImage2D(uint32_t target, int32_t level, int32_t xo, int32_t yo, int32_t x, int32_t y, int32_t w, int32_t h) {
    rd_copy_account(w, h, 1);
    if (p_rd_real_glCopyTexSubImage2D) p_rd_real_glCopyTexSubImage2D(target, level, xo, yo, x, y, w, h);
}
static void rd_glBlitFramebuffer(int32_t sx0, int32_t sy0, int32_t sx1, int32_t sy1, int32_t dx0, int32_t dy0, int32_t dx1, int32_t dy1, uint32_t mask, uint32_t filter) {
    RD_OPLOG("RIMDROID OP#%llu BlitFramebuffer %d,%d-%d,%d -> %d,%d-%d,%d mask=0x%x\n", (unsigned long long)rd_gl_op_seq, sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask);
    rd_copy_account((int64_t)(dx1 > dx0 ? dx1 - dx0 : dx0 - dx1), (int64_t)(dy1 > dy0 ? dy1 - dy0 : dy0 - dy1), 1);
    if (p_rd_real_glBlitFramebuffer) p_rd_real_glBlitFramebuffer(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter);
}
// Draw/dispatch/mipmap shims — pure op-log passthroughs for the culprit hunt: draws were
// invisible to every previous instrument (syncdraw never engaged, upload pacing skips them),
// yet the killer may be the first draw sampling the freshly-baked atlases.
static void (*p_rd_real_glDrawArrays)(uint32_t,int32_t,int32_t) = NULL;
static void (*p_rd_real_glDrawElements)(uint32_t,int32_t,uint32_t,const void*) = NULL;
static void (*p_rd_real_glDrawElementsBaseVertex)(uint32_t,int32_t,uint32_t,const void*,int32_t) = NULL;
static void (*p_rd_real_glDrawArraysInstanced)(uint32_t,int32_t,int32_t,int32_t) = NULL;
static void (*p_rd_real_glDrawElementsInstanced)(uint32_t,int32_t,uint32_t,const void*,int32_t) = NULL;
static void (*p_rd_real_glDrawElementsInstancedBaseVertex)(uint32_t,int32_t,uint32_t,const void*,int32_t,int32_t) = NULL;
static void (*p_rd_real_glDispatchCompute)(uint32_t,uint32_t,uint32_t) = NULL;
static void (*p_rd_real_glGenerateMipmap)(uint32_t) = NULL;
static void rd_glDrawArrays(uint32_t mode, int32_t first, int32_t count) {
    RD_OPLOG("RIMDROID OP#%llu DrawArrays mode=0x%x first=%d n=%d\n", (unsigned long long)rd_gl_op_seq, mode, first, count);
    if (p_rd_real_glDrawArrays) p_rd_real_glDrawArrays(mode, first, count);
}
static void rd_glDrawElements(uint32_t mode, int32_t count, uint32_t type, const void* idx) {
    RD_OPLOG("RIMDROID OP#%llu DrawElements mode=0x%x n=%d type=0x%x\n", (unsigned long long)rd_gl_op_seq, mode, count, type);
    if (p_rd_real_glDrawElements) p_rd_real_glDrawElements(mode, count, type, idx);
}
static void rd_glDrawElementsBaseVertex(uint32_t mode, int32_t count, uint32_t type, const void* idx, int32_t base) {
    RD_OPLOG("RIMDROID OP#%llu DrawElementsBaseVertex mode=0x%x n=%d base=%d\n", (unsigned long long)rd_gl_op_seq, mode, count, base);
    if (p_rd_real_glDrawElementsBaseVertex) p_rd_real_glDrawElementsBaseVertex(mode, count, type, idx, base);
}
static void rd_glDrawArraysInstanced(uint32_t mode, int32_t first, int32_t count, int32_t inst) {
    RD_OPLOG("RIMDROID OP#%llu DrawArraysInstanced mode=0x%x n=%d inst=%d\n", (unsigned long long)rd_gl_op_seq, mode, count, inst);
    if (p_rd_real_glDrawArraysInstanced) p_rd_real_glDrawArraysInstanced(mode, first, count, inst);
}
static void rd_glDrawElementsInstanced(uint32_t mode, int32_t count, uint32_t type, const void* idx, int32_t inst) {
    RD_OPLOG("RIMDROID OP#%llu DrawElementsInstanced mode=0x%x n=%d inst=%d\n", (unsigned long long)rd_gl_op_seq, mode, count, inst);
    if (p_rd_real_glDrawElementsInstanced) p_rd_real_glDrawElementsInstanced(mode, count, type, idx, inst);
}
static void rd_glDrawElementsInstancedBaseVertex(uint32_t mode, int32_t count, uint32_t type, const void* idx, int32_t inst, int32_t base) {
    RD_OPLOG("RIMDROID OP#%llu DrawElementsInstancedBaseVertex mode=0x%x n=%d inst=%d base=%d\n", (unsigned long long)rd_gl_op_seq, mode, count, inst, base);
    if (p_rd_real_glDrawElementsInstancedBaseVertex) p_rd_real_glDrawElementsInstancedBaseVertex(mode, count, type, idx, inst, base);
}
static void rd_glDispatchCompute(uint32_t x, uint32_t y, uint32_t z) {
    RD_OPLOG("RIMDROID OP#%llu DispatchCompute %ux%ux%u\n", (unsigned long long)rd_gl_op_seq, x, y, z);
    if (p_rd_real_glDispatchCompute) p_rd_real_glDispatchCompute(x, y, z);
}
static void rd_glGenerateMipmap(uint32_t target) {
    RD_OPLOG("RIMDROID OP#%llu GenerateMipmap target=0x%x\n", (unsigned long long)rd_gl_op_seq, target);
    if (p_rd_real_glGenerateMipmap) p_rd_real_glGenerateMipmap(target);
}
// Shader identification (culprit = a draw hanging the GPU): the last UseProgram before the
// loss names the guilty program; ShaderSource dumps every shader's text (keyed by shader id)
// to RIMDROID_CACHE_DIR/rd_shaders.txt and AttachShader logs the program<->shader mapping.
static void (*p_rd_real_glUseProgram)(uint32_t) = NULL;
static void (*p_rd_real_glShaderSource)(uint32_t,int32_t,const char* const*,const int32_t*) = NULL;
static void (*p_rd_real_glAttachShader)(uint32_t,uint32_t) = NULL;
static void rd_glUseProgram(uint32_t prog) {
    RD_OPLOG("RIMDROID OP#%llu UseProgram %u\n", (unsigned long long)rd_gl_op_seq, prog);
    if (p_rd_real_glUseProgram) p_rd_real_glUseProgram(prog);
}
// TEXTURE-COMPRESSION FIX (2026-07-12): Unity's runtime BC-compression shader (Hidden/CompressBC,
// the only shader writing to a `uimage2D`) HANGS the GPU on Turnip/Adreno 830 — DEVICE_LOST, kgsl
// hang-class watchdog. It is an iterative BC encoder full of `while(true){ if(cond) break; ... }`
// loops whose exit condition reinterprets a float loop-counter via floatBitsToInt(); the prime
// hypothesis is that Turnip miscompiles that float<->int bitcast in the loop condition on A830 so
// a loop never exits -> infinite loop on the GPU. We already intercept GLSL here, so bound every
// such loop with a hard iteration cap: if the real break works the cap never fires (identical
// result); if the exit is miscompiled, the cap breaks the infinite loop (tiny quality cost). Real
// BC loops are <=~64 iterations, so the default 256 cap is safe. Env RIMDROID_BC_CAP overrides it;
// RIMDROID_BC_NOCAP=1 disables the transform (A/B). Only the compressor shader is touched.
static char* rd_bc_bound_loops(const char* src, int* out_n) {
    const char* NEEDLE = "while(true){";
    const size_t NL = 12;
    int cap = 16;   // experiment 2: BC endpoint/palette loops need <=~16 iters; a tight cap tests
                    // whether a miscompiled exit is running them far past termination (256 still
                    // DEVICE_LOST). Env RIMDROID_BC_CAP overrides.
    { const char* e = getenv("RIMDROID_BC_CAP"); if (e && atoi(e) > 0) cap = atoi(e); }
    char repl[96];
    int rl = snprintf(repl, sizeof(repl), "for(int _rd_g=0;_rd_g<%d;++_rd_g){", cap);
    // count occurrences to size the output
    int n = 0;
    for (const char* p = src; (p = strstr(p, NEEDLE)); p += NL) n++;
    if (n == 0) { if (out_n) *out_n = 0; return NULL; }
    size_t inlen = strlen(src);
    char* dst = (char*)malloc(inlen + (size_t)n * (size_t)(rl - (int)NL) + 1);
    if (!dst) { if (out_n) *out_n = 0; return NULL; }
    char* w = dst; const char* r = src;
    const char* hit;
    while ((hit = strstr(r, NEEDLE))) {
        memcpy(w, r, (size_t)(hit - r)); w += (hit - r);
        memcpy(w, repl, (size_t)rl);     w += rl;
        r = hit + NL;
    }
    strcpy(w, r);
    if (out_n) *out_n = n;
    return dst;
}
static void rd_glShaderSource(uint32_t shader, int32_t count, const char* const* strings, const int32_t* lengths) {
    static FILE* f = NULL; static int tried = 0;
    if (!tried) {
        tried = 1;
        const char* dir = getenv("RIMDROID_CACHE_DIR");
        if (dir && dir[0]) {
            char path[512];
            snprintf(path, sizeof(path), "%s/rd_shaders.txt", dir);
            f = fopen(path, "w");
            printf_log(LOG_NONE, "RIMDROID shader dump -> %s (%s)\n", path, f ? "ok" : "FAILED");
        }
    }
    if (f && strings && count > 0) {
        fprintf(f, "=== shader %u (%d parts) ===\n", shader, count);
        for (int32_t i = 0; i < count; i++) {
            if (!strings[i]) continue;
            if (lengths && lengths[i] >= 0) fwrite(strings[i], 1, (size_t)lengths[i], f);
            else fputs(strings[i], f);
        }
        fputc('\n', f);
        fflush(f);
    }
    // CompressBC loop-bounding: concatenate the parts, and if this is the compressor (writes a
    // uimage2D), hand the real driver a version with every while(true) loop hard-capped.
    static int nocap = -1;
    if (nocap == -1) nocap = getenv("RIMDROID_BC_NOCAP") ? 1 : 0;
    if (!nocap && strings && count > 0) {
        size_t total = 0;
        for (int32_t i = 0; i < count; i++)
            total += strings[i] ? (lengths && lengths[i] >= 0 ? (size_t)lengths[i] : strlen(strings[i])) : 0;
        char* joined = (char*)malloc(total + 1);
        if (joined) {
            char* w = joined;
            for (int32_t i = 0; i < count; i++) {
                if (!strings[i]) continue;
                size_t l = (lengths && lengths[i] >= 0) ? (size_t)lengths[i] : strlen(strings[i]);
                memcpy(w, strings[i], l); w += l;
            }
            *w = 0;
            if (strstr(joined, "uimage2D")) {
                int nloops = 0;
                char* fixed = rd_bc_bound_loops(joined, &nloops);
                if (fixed && nloops > 0) {
                    printf_log(LOG_NONE, "RIMDROID CompressBC shim: bounded %d while-loops on shader %u\n", nloops, shader);
                    fflush(NULL);
                    const char* one[1] = { fixed };
                    if (p_rd_real_glShaderSource) p_rd_real_glShaderSource(shader, 1, one, NULL);
                    free(fixed); free(joined);
                    return;
                }
                free(fixed);
            }
            free(joined);
        }
    }
    if (p_rd_real_glShaderSource) p_rd_real_glShaderSource(shader, count, strings, lengths);
}
static void rd_glAttachShader(uint32_t prog, uint32_t shader) {
    printf_log(LOG_NONE, "RIMDROID GLSANITY AttachShader prog=%u shader=%u\n", prog, shader);
    if (p_rd_real_glAttachShader) p_rd_real_glAttachShader(prog, shader);
}

// libzfa lacks the DSA getters (glGetTextureParameteriv/LevelParameteriv, GL4.5)
// but DOES export the classic GL-1.0 glGetTexParameteriv/glGetTexLevelParameteriv.
// Returning 0 (the old stub) made Unity read TEXTURE_WIDTH=0 on a freshly-created
// texture → "0x0 texture" → GfxDevice "device lost" → the infinite
// SDL_GL_DeleteContext(NULL) teardown loop.  Proxy DSA→classic: bind the texture
// to GL_TEXTURE_2D, query via the classic getter, restore the previous binding.
static void rd_glGetTextureParameteriv(uint32_t texture, uint32_t pname, int32_t* params) {
    static void (*bind)(uint32_t,uint32_t) = 0;
    static void (*get)(uint32_t,uint32_t,int32_t*) = 0;
    static void (*getiv)(uint32_t,int32_t*) = 0;
    static int init = 0;
    if (!init) { init=1; bind=rd_zfa_gl("glBindTexture"); get=rd_zfa_gl("glGetTexParameteriv"); getiv=rd_zfa_gl("glGetIntegerv"); }
    if (!params) return;
    params[0] = 0;
    if (bind && get && getiv) {
        int32_t prev = 0; getiv(0x8069 /*GL_TEXTURE_BINDING_2D*/, &prev);
        bind(0x0DE1 /*GL_TEXTURE_2D*/, texture);
        get(0x0DE1, pname, params);
        bind(0x0DE1, (uint32_t)prev);
    }
    static int n=0; if(n<16){n++; printf_log(LOG_NONE, "RIMDROID glGetTextureParameteriv tex=%u pname=0x%x => %d%s\n", texture, pname, params[0], (bind&&get)?"":" [no-proxy]");}
}
static void rd_glGetTextureLevelParameteriv(uint32_t texture, int32_t level, uint32_t pname, int32_t* params) {
    static void (*bind)(uint32_t,uint32_t) = 0;
    static void (*get)(uint32_t,int32_t,uint32_t,int32_t*) = 0;
    static void (*getiv)(uint32_t,int32_t*) = 0;
    static int init = 0;
    if (!init) { init=1; bind=rd_zfa_gl("glBindTexture"); get=rd_zfa_gl("glGetTexLevelParameteriv"); getiv=rd_zfa_gl("glGetIntegerv"); }
    if (!params) return;
    params[0] = 0;
    if (bind && get && getiv) {
        int32_t prev = 0; getiv(0x8069 /*GL_TEXTURE_BINDING_2D*/, &prev);
        bind(0x0DE1 /*GL_TEXTURE_2D*/, texture);
        get(0x0DE1, level, pname, params);
        bind(0x0DE1, (uint32_t)prev);
    }
    static int n=0; if(n<16){n++; printf_log(LOG_NONE, "RIMDROID glGetTextureLevelParameteriv tex=%u lvl=%d pname=0x%x => %d%s\n", texture, level, pname, params[0], (bind&&get)?"":" [no-proxy]");}
}
static void rd_glGetQueryObjectui64v(uint32_t id, uint32_t pname, uint64_t* params) {
    if (!params) return;
    // GL_QUERY_RESULT_AVAILABLE(0x8867) → TRUE (else Unity may wait forever / treat
    // GPU timers as broken = device failure); GL_QUERY_RESULT(0x8866) → 0.
    params[0] = (pname == 0x8867) ? 1u : 0u;
    static int n=0; if(n<16){n++; printf_log(LOG_NONE, "RIMDROID glGetQueryObjectui64v id=%u pname=0x%x => %llu\n", id, pname, (unsigned long long)params[0]);}
}

// (the old diagnostic glTexSubImage2D shim merged into the pacing shim above — one definition,
//  both install paths: the resolver populates p_rd_real, the SDL special-case self-resolves.)

// DL functions from wrappedlibdl.c
void* my_dlopen(x64emu_t* emu, void *filename, int flag);
int my_dlclose(x64emu_t* emu, void *handle);
void* my_dlsym(x64emu_t* emu, void *handle, void *symbol);

static int sdl_Yes() { return 1;}
static int sdl_No() { return 0;}
int EXPORT my2_SDL_Has3DNow(void) __attribute__((alias("sdl_No")));
int EXPORT my2_SDL_Has3DNowExt(void) __attribute__((alias("sdl_No")));
int EXPORT my2_SDL_HasAltiVec(void) __attribute__((alias("sdl_No")));
int EXPORT my2_SDL_HasMMX(void) __attribute__((alias("sdl_Yes")));
int EXPORT my2_SDL_HasMMXExt(void) __attribute__((alias("sdl_Yes")));
int EXPORT my2_SDL_HasNEON(void) __attribute__((alias("sdl_No")));   // No neon in x86_64 ;)
int EXPORT my2_SDL_HasRDTSC(void) __attribute__((alias("sdl_Yes")));
int EXPORT my2_SDL_HasSSE(void) __attribute__((alias("sdl_Yes")));
int EXPORT my2_SDL_HasSSE2(void) __attribute__((alias("sdl_Yes")));
int EXPORT my2_SDL_HasSSE3(void) __attribute__((alias("sdl_Yes")));
int EXPORT my2_SDL_HasSSE41(void) __attribute__((alias("sdl_Yes")));
int EXPORT my2_SDL_HasSSE42(void) {
    return BOX64ENV(sse42)?1:0;
}
int EXPORT my2_SDL_HasAVX(void) {
    return BOX64ENV(avx)?1:0;
}
int EXPORT my2_SDL_HasAVX2(void) {
    return BOX64ENV(avx2)?1:0;
}
int EXPORT my2_SDL_HasAVX512F(void) __attribute__((alias("sdl_No")));

typedef struct {
  int32_t freq;
  uint16_t format;
  uint8_t channels;
  uint8_t silence;
  uint16_t samples;
  uint16_t padding;
  uint32_t size;
  void (*callback)(void *userdata, uint8_t *stream, int32_t len);
  void *userdata;
} SDL2_AudioSpec;

typedef struct {
    uint8_t data[16];
} SDL_JoystickGUID;

typedef union {
    SDL_JoystickGUID guid;
    uint32_t         u[4];
} SDL_JoystickGUID_Helper;

typedef struct
{
    int32_t bindType;   // enum
    union
    {
        int button;
        int axis;
        struct {
            int hat;
            int hat_mask;
        } hat;
    } value;
} SDL_GameControllerButtonBind;


typedef void  (*vFv_t)();
typedef void  (*vFiupV_t)(int64_t, uint64_t, void*, va_list);
#define ADDED_FUNCTIONS() \
    GO(SDL_Quit, vFv_t)           \
    GO(SDL_AllocRW, sdl2_allocrw) \
    GO(SDL_FreeRW, sdl2_freerw)   \
    GO(SDL_LogMessageV, vFiupV_t)
#include "generated/wrappedsdl2types.h"

#include "wrappercallback.h"

#define SUPER() \
GO(0)   \
GO(1)   \
GO(2)   \
GO(3)   \
GO(4)

// Timer
#define GO(A)   \
static uintptr_t my_Timer_fct_##A = 0;                                      \
static uint64_t my_Timer_##A(uint64_t a, void* b)                           \
{                                                                           \
    return (uint64_t)RunFunctionFmt(my_Timer_fct_##A, "Up", a, b);    \
}
SUPER()
#undef GO
static void* find_Timer_Fct(void* fct)
{
    if(!fct) return NULL;
    void* p;
    if((p = GetNativeFnc((uintptr_t)fct))) return p;
    #define GO(A) if(my_Timer_fct_##A == (uintptr_t)fct) return my_Timer_##A;
    SUPER()
    #undef GO
    #define GO(A) if(my_Timer_fct_##A == 0) {my_Timer_fct_##A = (uintptr_t)fct; return my_Timer_##A; }
    SUPER()
    #undef GO
    printf_log(LOG_NONE, "Warning, no more slot for SDL2 Timer callback\n");
    return NULL;

}
// AudioCallback
#define GO(A)   \
static uintptr_t my_AudioCallback_fct_##A = 0;                      \
static void my_AudioCallback_##A(void* a, void* b, int c)           \
{                                                                   \
    RunFunctionFmt(my_AudioCallback_fct_##A, "ppi", a, b, c);  \
}
SUPER()
#undef GO
static void* find_AudioCallback_Fct(void* fct)
{
    if(!fct) return NULL;
    void* p;
    if((p = GetNativeFnc((uintptr_t)fct))) return p;
    #define GO(A) if(my_AudioCallback_fct_##A == (uintptr_t)fct) return my_AudioCallback_##A;
    SUPER()
    #undef GO
    #define GO(A) if(my_AudioCallback_fct_##A == 0) {my_AudioCallback_fct_##A = (uintptr_t)fct; return my_AudioCallback_##A; }
    SUPER()
    #undef GO
    printf_log(LOG_NONE, "Warning, no more slot for SDL2 AudioCallback callback\n");
    return NULL;

}
// eventfilter
#define GO(A)   \
static uintptr_t my_eventfilter_fct_##A = 0;                                \
static int my_eventfilter_##A(void* userdata, void* event)                  \
{                                                                           \
    return (int)RunFunctionFmt(my_eventfilter_fct_##A, "pp", userdata, event);    \
}
SUPER()
#undef GO
static void* find_eventfilter_Fct(void* fct)
{
    if(!fct) return NULL;
    void* p;
    if((p = GetNativeFnc((uintptr_t)fct))) return p;
    #define GO(A) if(my_eventfilter_fct_##A == (uintptr_t)fct) return my_eventfilter_##A;
    SUPER()
    #undef GO
    #define GO(A) if(my_eventfilter_fct_##A == 0) {my_eventfilter_fct_##A = (uintptr_t)fct; return my_eventfilter_##A; }
    SUPER()
    #undef GO
    printf_log(LOG_NONE, "Warning, no more slot for SDL2 eventfilter callback\n");
    return NULL;

}
static void* reverse_eventfilter_Fct(void* fct)
{
    if(!fct) return fct;
    if(CheckBridged(my_lib->w.bridge, fct))
        return (void*)CheckBridged(my_lib->w.bridge, fct);
    #define GO(A) if(my_eventfilter_##A == fct) return (void*)my_eventfilter_fct_##A;
    SUPER()
    #undef GO
    return (void*)AddBridge(my_lib->w.bridge, iFpp, fct, 0, NULL);
}

// LogOutput
#define GO(A)   \
static uintptr_t my_LogOutput_fct_##A = 0;                                  \
static void my_LogOutput_##A(void* a, int b, int c, void* d)                \
{                                                                           \
    RunFunctionFmt(my_LogOutput_fct_##A, "piip", a, b, c, d);  \
}
SUPER()
#undef GO
static void* find_LogOutput_Fct(void* fct)
{
    if(!fct) return fct;
    if(GetNativeFnc((uintptr_t)fct))  return GetNativeFnc((uintptr_t)fct);
    #define GO(A) if(my_LogOutput_fct_##A == (uintptr_t)fct) return my_LogOutput_##A;
    SUPER()
    #undef GO
    #define GO(A) if(my_LogOutput_fct_##A == 0) {my_LogOutput_fct_##A = (uintptr_t)fct; return my_LogOutput_##A; }
    SUPER()
    #undef GO
    printf_log(LOG_NONE, "Warning, no more slot for SDL2 LogOutput callback\n");
    return NULL;
}
static void* reverse_LogOutput_Fct(void* fct)
{
    if(!fct) return fct;
    if(CheckBridged(my_lib->w.bridge, fct))
        return (void*)CheckBridged(my_lib->w.bridge, fct);
    #define GO(A) if(my_LogOutput_##A == fct) return (void*)my_LogOutput_fct_##A;
    SUPER()
    #undef GO
    return (void*)AddBridge(my_lib->w.bridge, vFpiip, fct, 0, NULL);
}

// Hint
#define GO(A) \
static uintptr_t my_Hint_fct_##A = 0; \
static void my_Hint_##A(void* userdata, const char* name, const char* oldValue, const char* newValue) \
{ \
    RunFunctionFmt(my_Hint_fct_##A, "pppp", userdata, name, oldValue, newValue); \
}
SUPER()
#undef GO
static void* find_Hint_Fct(void* fct)
{
    if(!fct) return fct;
    if(GetNativeFnc((uintptr_t)fct))  return GetNativeFnc((uintptr_t)fct);
    #define GO(A) if(my_Hint_fct_##A == (uintptr_t)fct) return my_Hint_##A;
    SUPER()
    #undef GO
    #define GO(A) if(my_Hint_fct_##A == 0) {my_Hint_fct_##A = (uintptr_t)fct; return my_Hint_##A; }
    SUPER()
    #undef GO
    printf_log(LOG_NONE, "Warning, no more slot for SDL2 Hint callback\n");
    return NULL;
}
static void* reverse_Hint_Fct(void* fct)
{
    if(!fct) return fct;
    if(CheckBridged(my_lib->w.bridge, fct))
        return (void*)CheckBridged(my_lib->w.bridge, fct);
    #define GO(A) if(my_Hint_##A == fct) return (void*)my_Hint_fct_##A;
    SUPER()
    #undef GO
    return (void*)AddBridge(my_lib->w.bridge, vFpppp, fct, 0, NULL);
}

#undef SUPER

// TODO: track the memory for those callback
EXPORT int64_t my2_SDL_OpenAudio(x64emu_t* emu, void* d, void* o)
{
    SDL2_AudioSpec *desired = (SDL2_AudioSpec*)d;

    // create a callback
    void *fnc = (void*)desired->callback;
    desired->callback = find_AudioCallback_Fct(fnc);
    int ret = my->SDL_OpenAudio(desired, (SDL2_AudioSpec*)o);
    if (ret!=0) {
        // error, clean the callback...
        desired->callback = fnc;
        return ret;
    }
    // put back stuff in place?
    desired->callback = fnc;

    return ret;
}

EXPORT uint32_t my2_SDL_OpenAudioDevice(x64emu_t* emu, void* device, int iscapture, void* d, void* o, int allowed)
{
    SDL2_AudioSpec* desired = (SDL2_AudioSpec*)d;

    // create a callback
    void* fnc = (void*)desired->callback;
    desired->callback = find_AudioCallback_Fct(fnc);
    uint32_t ret = my->SDL_OpenAudioDevice(device, iscapture, desired, (SDL2_AudioSpec*)o, allowed);

    // put back stuff in place?
    desired->callback = fnc;

    return ret;
}

EXPORT void *my2_SDL_LoadFile_RW(x64emu_t* emu, void* a, void* b, int c)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    void* r = my->SDL_LoadFile_RW(rw, b, c);
    if(c==0)
        RWNativeEnd2(rw);
    return r;
}
EXPORT void *my2_SDL_LoadBMP_RW(x64emu_t* emu, void* a, int b)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    void* r = my->SDL_LoadBMP_RW(rw, b);
    if(b==0)
        RWNativeEnd2(rw);
    return r;
}
EXPORT int64_t my2_SDL_SaveBMP_RW(x64emu_t* emu, void* a, void* b, int c)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    int64_t r = my->SDL_SaveBMP_RW(rw, b, c);
    if(c==0)
        RWNativeEnd2(rw);
    return r;
}
EXPORT void *my2_SDL_LoadWAV_RW(x64emu_t* emu, void* a, int b, void* c, void* d, void* e)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    void* r = my->SDL_LoadWAV_RW(rw, b, c, d, e);
    if(b==0)
        RWNativeEnd2(rw);
    return r;
}
EXPORT int my2_SDL_GameControllerAddMappingsFromRW(x64emu_t* emu, void* a, int b)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    int r = my->SDL_GameControllerAddMappingsFromRW(rw, b);
    if(b==0)
        RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_ReadU8(x64emu_t* emu, void* a)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_ReadU8(rw);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_ReadBE16(x64emu_t* emu, void* a)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_ReadBE16(rw);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_ReadBE32(x64emu_t* emu, void* a)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_ReadBE32(rw);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_ReadBE64(x64emu_t* emu, void* a)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_ReadBE64(rw);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_ReadLE16(x64emu_t* emu, void* a)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_ReadLE16(rw);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_ReadLE32(x64emu_t* emu, void* a)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_ReadLE32(rw);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_ReadLE64(x64emu_t* emu, void* a)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_ReadLE64(rw);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_WriteU8(x64emu_t* emu, void* a, uint8_t v)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_WriteU8(rw, v);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_WriteBE16(x64emu_t* emu, void* a, uint16_t v)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_WriteBE16(rw, v);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_WriteBE32(x64emu_t* emu, void* a, uint64_t v)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_WriteBE32(rw, v);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_WriteBE64(x64emu_t* emu, void* a, uint64_t v)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_WriteBE64(rw, v);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_WriteLE16(x64emu_t* emu, void* a, uint16_t v)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_WriteLE16(rw, v);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_WriteLE32(x64emu_t* emu, void* a, uint64_t v)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_WriteLE32(rw, v);
    RWNativeEnd2(rw);
    return r;
}
EXPORT uint64_t my2_SDL_WriteLE64(x64emu_t* emu, void* a, uint64_t v)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    uint64_t r = my->SDL_WriteLE64(rw, v);
    RWNativeEnd2(rw);
    return r;
}

EXPORT void *my2_SDL_RWFromConstMem(x64emu_t* emu, void* a, int b)
{
    void* r = my->SDL_RWFromConstMem(a, b);
    return AddNativeRW2(emu, (SDL2_RWops_t*)r);
}
EXPORT void *my2_SDL_RWFromFP(x64emu_t* emu, void* a, int b)
{
    void* r = my->SDL_RWFromFP(a, b);
    return AddNativeRW2(emu, (SDL2_RWops_t*)r);
}
EXPORT void *my2_SDL_RWFromFile(x64emu_t* emu, void* a, void* b)
{
    void* r = my->SDL_RWFromFile(a, b);
    return AddNativeRW2(emu, (SDL2_RWops_t*)r);
}
EXPORT void *my2_SDL_RWFromMem(x64emu_t* emu, void* a, int b)
{
    void* r = my->SDL_RWFromMem(a, b);
    return AddNativeRW2(emu, (SDL2_RWops_t*)r);
}

EXPORT int64_t my2_SDL_RWseek(x64emu_t* emu, void* a, int64_t offset, int whence)
{
    //sdl2_my_t *my = (sdl2_my_t *)emu->context->sdl2lib->priv.w.p2;
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    int64_t ret = RWNativeSeek2(rw, offset, whence);
    RWNativeEnd2(rw);
    return ret;
}
EXPORT int64_t my2_SDL_RWtell(x64emu_t* emu, void* a)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    int64_t ret = RWNativeSeek2(rw, 0, 1);  //1 == RW_SEEK_CUR
    RWNativeEnd2(rw);
    return ret;
}
EXPORT size_t my2_SDL_RWread(x64emu_t* emu, void* a, void* ptr, size_t size, size_t maxnum)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    size_t ret = RWNativeRead2(rw, ptr, size, maxnum);
    RWNativeEnd2(rw);
    return ret;
}
EXPORT size_t my2_SDL_RWwrite(x64emu_t* emu, void* a, const void* ptr, size_t size, size_t maxnum)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    size_t ret = RWNativeWrite2(rw, ptr, size, maxnum);
    RWNativeEnd2(rw);
    return ret;
}
EXPORT int my2_SDL_RWclose(x64emu_t* emu, void* a)
{
    //sdl2_my_t *my = (sdl2_my_t *)emu->context->sdl2lib->priv.w.p2;
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    return RWNativeClose2(rw);
}

EXPORT int my2_SDL_SaveAllDollarTemplates(x64emu_t* emu, void* a)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    int ret = my->SDL_SaveAllDollarTemplates(rw);
    RWNativeEnd2(rw);
    return ret;
}

EXPORT int my2_SDL_SaveDollarTemplate(x64emu_t* emu, int gesture, void* a)
{
    SDL2_RWops_t *rw = RWNativeStart2(emu, (SDL2_RWops_t*)a);
    int ret = my->SDL_SaveDollarTemplate(gesture, rw);
    RWNativeEnd2(rw);
    return ret;
}

EXPORT void *my2_SDL_AddTimer(x64emu_t* emu, uint64_t a, void* f, void* p)
{
    return my->SDL_AddTimer(a, find_Timer_Fct(f), p);
}

EXPORT int my2_SDL_RemoveTimer(x64emu_t* emu, void* t)
{
    return my->SDL_RemoveTimer(t);
}

EXPORT void my2_SDL_SetEventFilter(x64emu_t* emu, void* p, void* userdata)
{
    my->SDL_SetEventFilter(find_eventfilter_Fct(p), userdata);
}
EXPORT int my2_SDL_GetEventFilter(x64emu_t* emu, void** f, void* userdata)
{
    int ret = my->SDL_GetEventFilter(f, userdata);
    *f = reverse_eventfilter_Fct(*f);
    return ret;
}

EXPORT void my2_SDL_LogGetOutputFunction(x64emu_t* emu, void** f, void* arg)
{

    my->SDL_LogGetOutputFunction(f, arg);
    if(*f) *f = reverse_LogOutput_Fct(*f);
}
EXPORT void my2_SDL_LogSetOutputFunction(x64emu_t* emu, void* f, void* arg)
{

    my->SDL_LogSetOutputFunction(find_LogOutput_Fct(f), arg);
}

EXPORT void my2_SDL_AddHintCallback(x64emu_t* emu, char* name, void* callback, void* userdata)
{
    my->SDL_AddHintCallback(name, find_Hint_Fct(callback), userdata);
}
EXPORT void my2_SDL_DelHintCallback(x64emu_t* emu, char* name, void* callback, void* userdata)
{
    my->SDL_DelHintCallback(name, reverse_Hint_Fct(callback), userdata);
}

EXPORT int my2_SDL_vsnprintf(x64emu_t* emu, void* buff, size_t s, void * fmt, x64_va_list_t b)
{
    (void)emu;
    #ifdef CONVERT_VALIST
    CONVERT_VALIST(b);
    #else
    myStackAlignValist(emu, (const char*)fmt, emu->scratch, b);
    PREPARE_VALIST;
    #endif
    int r = vsnprintf(buff, s, fmt, VARARGS);
    return r;
}

EXPORT void* my2_SDL_CreateThread(x64emu_t* emu, void* f, void* n, void* p)
{
    void* et = NULL;
    void* fnc = my_prepare_thread(emu, f, p, 0, &et);
    return my->SDL_CreateThread(fnc, n, et);
}

EXPORT int my2_SDL_snprintf(x64emu_t* emu, void* buff, size_t s, void * fmt, uint64_t * b) {
    myStackAlign(emu, (const char*)fmt, b, emu->scratch, R_EAX, 3);
    PREPARE_VALIST;
    return vsnprintf(buff, s, fmt, VARARGS);
}

static int get_sdl_priv(x64emu_t* emu, const char *sym_str, void **w, void **f)
{
    #define GO(sym, _w) \
        else if (strcmp(#sym, sym_str) == 0) \
        { \
            *w = _w; \
            *f = dlsym(emu->context->sdl2lib->w.lib, #sym); \
            return *f != NULL; \
        }
    #define GO2(sym, _w, sym2) \
        else if (strcmp(#sym, sym_str) == 0) \
        { \
            *w = _w; \
            *f = dlsym(emu->context->sdl2lib->w.lib, #sym2); \
            return *f != NULL; \
        }
    #define GOM(sym, _w) \
        else if (strcmp(#sym, sym_str) == 0) \
        { \
            *w = _w; \
            *f = dlsym(emu->context->box64lib, "my2_"#sym); \
            return *f != NULL; \
        }
    #define DATA

    if(0);
    #include "wrappedsdl2_private.h"

    #undef GO
    #undef GOM
    #undef GO2
    #undef DATA
    return 0;
}

int EXPORT my2_SDL_DYNAPI_entry(x64emu_t* emu, uint32_t version, uintptr_t *table, uint32_t tablesize)
{
    // tablesize is sizeof(SDL_DYNAPI_jump_table) in BYTES.
    // Convert to element count (each entry is one uintptr_t = function pointer).
    uint32_t n = tablesize / (uint32_t)sizeof(uintptr_t);
    uint32_t i = 0;
    uintptr_t tab[n];

    if (my->SDL_DYNAPI_entry) {
        // Real ARM64 libSDL2 available — populate tab with its jump table.
        // Pass tablesize in bytes, as SDL expects.
        int r = my->SDL_DYNAPI_entry(version, tab, tablesize);
        (void)r;
        printf_log(LOG_INFO, "SDL_DYNAPI_entry: real ARM64 SDL2 table loaded (%u bytes / %u entries)\n", tablesize, n);
    } else {
        // No real ARM64 libSDL2 (statically linked game).  The `table` we got is
        // SDL's jump_table in its INITIAL state: every entry points at a *_DEFAULT
        // stub that calls SDL_InitDynamicAPI() then tail-jumps to jump_table.fn.
        // Because SDL_DYNAMIC_API is set, SDL calls ONLY this external entry and
        // never its own built-in one — so the real static function pointers are
        // never installed.  If we seeded `tab` from these default stubs, every
        // non-wrapped SDL_* call would re-enter SDL_InitDynamicAPI and tail-jump
        // back to the default stub forever (single-threaded 100% CPU hang at
        // startup, observed as a spin inside SDL_AtomicLock).
        //
        // Fix: call the GAME's OWN built-in SDL_DYNAPI_entry to populate `table`
        // with the real static SDL2 implementations, then read them into `tab`.
        // We run it directly by address so box64 does not re-intercept it back
        // into this function.
        //
        // IMPORTANT (RimWorld 1.5 / Unity 2022): the built-in is NOT in elfs[0].
        // 1.5 ships a tiny 6 KB launcher ELF (elfs[0]) + the engine as a separate
        // UnityPlayer.so DYN library, and SDL2 (hence SDL_DYNAPI_entry) is linked
        // into UnityPlayer.so.  Searching only elfs[0] (as before, fine for the
        // monolithic 1.2 EXE) finds nothing → fallback default-stub table → every
        // non-wrapped SDL_* call hangs forever in SDL_InitDynamicAPI/SDL_AtomicLock.
        // So scan ALL loaded ELFs for the built-in entry.
        uintptr_t gstart = 0, gend = 0;
        int gver = 0, gveropt = 0;
        const char* gvername = NULL;
        int found_elf = -1;
        for (int ei = 0; ei < emu->context->elfsize; ++ei) {
            elfheader_t* e = emu->context->elfs[ei];
            if (!e) continue;
            gstart = gend = 0; gver = 0; gveropt = 0; gvername = NULL;
            if (ElfGetGlobalSymbolStartEnd(e, &gstart, &gend, "SDL_DYNAPI_entry",
                                           &gver, &gvername, 1, &gveropt) && gstart) {
                found_elf = ei;
                break;
            }
        }
        if (found_elf >= 0 && gstart) {
            RunFunctionWithEmu(emu, 0, gstart, 3,
                               (uint64_t)version,
                               (uint64_t)(uintptr_t)table,
                               (uint64_t)tablesize);
            memcpy(tab, table, tablesize);
            printf_log(LOG_NONE, "RIMDROID SDL_DYNAPI_entry: seeded from built-in @%p (elf #%d, %u entries)\n", (void*)gstart, found_elf, n);
        } else {
            // Built-in not found in ANY elf — fall back to the (broken) default-stub table.
            memcpy(tab, table, tablesize);
            printf_log(LOG_NONE, "RIMDROID SDL_DYNAPI_entry: built-in NOT found in any of %d elfs — non-wrapped SDL calls WILL hang (%u entries)\n", emu->context->elfsize, n);
        }
    }

    printf_log(LOG_NONE, "RIMDROID jump_table base=%p tablesize=%u n=%u\n", (void*)table, tablesize, n);

    // --- RimDroid corrective GL dynapi remap --------------------------------
    // RimWorldLinux ships Unity's own static SDL2, whose SDL_dynapi jump-table
    // index order differs from box64's canonical SDL_dynapi_procs.h.  The loop
    // below installs each wrapped bridge at box64's index; for the GL cluster
    // those indices are wrong for RimWorld (its GL cluster is uniformly shifted
    // -5 vs box64's SDL_dynapi order), and Unity's renderer detection lands our
    // bridges on the wrong slots (e.g. SDL_GL_GetAttribute hits our LoadLibrary
    // bridge).  RimWorld's real indices were obtained by disassembling its
    // built-in SDL_DYNAPI_entry and matching each static wrapper to the
    // SDL_VideoDevice GL vtable offset it calls.  We capture each wrapped GL
    // bridge + the box64 index it was placed at during the loop, then move it.
    static const struct { const char* name; uint32_t rw_idx; } rd_remap[] = {
        {"SDL_GL_LoadLibrary",    509},
        {"SDL_GL_GetProcAddress", 510},
        {"SDL_GL_CreateContext",  515},
        {"SDL_GL_MakeCurrent",    516},
        // FIX (2026-05-31): the game DOES have SDL_GL_GetDrawableSize (game idx 519,
        // verified by disassembling its SDL_GL_* stubs reading jump_table[base+idx*8]).
        // So the GL cluster shift is a UNIFORM -5, NOT -5-then-6. The old 521/522 put
        // the SwapWindow bridge where the game calls GetSwapInterval, and routed the
        // game's SwapWindow (idx 522) into our DeleteContext no-op → render thread span
        // SwapWindow forever, swap=0, black screen ("the loop"). Correct: 522/523.
        {"SDL_GL_SwapWindow",     522},
        {"SDL_GL_DeleteContext",  523},
        // 1.5 game indices (box64 has GetDrawableSize the game lacks → shift):
        {"SDL_GL_GetCurrentWindow",  517},
        {"SDL_GL_GetCurrentContext", 518},
        // SDL_CreateWindow: game idx 470 (box64 475, shift -5).  Route to
        // my2_SDL_CreateWindow so it STRIPS SDL_WINDOW_OPENGL — the dummy video
        // driver has no GL and fails SDL_CreateWindow when that flag is set,
        // leaving Unity with a NULL window (win=0x0) and a degenerate GfxDevice
        // that loops forever destroying contexts.  Stripping it lets the window
        // be created (we supply GL via ZFA regardless).
        {"SDL_CreateWindow",         470},
        // Window invariants (game idx; box64 479/488, shift -5):
        {"SDL_GetWindowFlags",       474},
        {"SDL_GetWindowSize",        483},
        // TEST: force display mode 1024x768 @ 60 Hz (game idx; box64 468/469, shift -5)
        {"SDL_GetDesktopDisplayMode", 463},
        {"SDL_GetCurrentDisplayMode", 464},
        // Phase A input: route the game's SDL_PollEvent (game idx 81, from
        // disassembling its stub) to our my2_SDL_PollEvent injector.
        {"SDL_PollEvent",            81},
        // SDL_GetMouseState (game idx 205) → return our injected cursor/buttons so
        // selection-drag and right-click targeting use the right position.
        {"SDL_GetMouseState",       205},
    };
    const int rd_nremap = (int)(sizeof(rd_remap)/sizeof(rd_remap[0]));
    uint32_t  rd_box64_idx[20];
    uintptr_t rd_bridge[20];
    for (int rj = 0; rj < rd_nremap; ++rj) { rd_box64_idx[rj] = (uint32_t)-1; rd_bridge[rj] = 0; }

    #define SDL_DYNAPI_PROC(ret, sym, args, parms, ...) \
        if (i < n) { \
            void *w = NULL; \
            void *f = NULL; \
            /* Capture the real static SDL_CreateWindow so my2_SDL_CreateWindow can \
               call it (with the SDL_WINDOW_OPENGL flag stripped). */ \
            if (!strcmp(#sym, "SDL_CreateWindow")) g_real_sdl_createwindow = tab[i]; \
            if (!strcmp(#sym, "SDL_GL_LoadLibrary")) g_real_sdl_gl_loadlibrary = tab[i]; \
            if (get_sdl_priv(emu, #sym, &w, &f)) { \
                table[i] = AddCheckBridge(my_lib->w.bridge, w, f, 0, #sym); \
                printf_log(LOG_DEBUG, "SDL_DYNAPI_entry: wrapped  %s => %p\n", #sym, (void*)table[i]); \
            } \
            else { \
                table[i] = tab[i]; \
            } \
            /* Record wrapped GL bridges so the remap below can relocate them. */ \
            for (int rj = 0; rj < rd_nremap; ++rj) \
                if (!strcmp(#sym, rd_remap[rj].name)) { rd_box64_idx[rj] = i; rd_bridge[rj] = table[i]; } \
            /* GROUND TRUTH: log box64 index + absolute slot addr for GL/window funcs */ \
            if (!strncmp(#sym, "SDL_GL_", 7) || !strcmp(#sym, "SDL_CreateWindow")) \
                printf_log(LOG_NONE, "RIMDROID MAP %s box64_idx=%u slot=%p\n", #sym, i, (void*)&table[i]); \
            i++; \
        }

    #include "SDL_dynapi_procs.h"

    // Re-capture real static SDL fns at the GAME indices.  The in-loop captures
    // above used `tab[i]` at box64's running index, but `tab` is seeded in the
    // GAME's DYNAPI order, which is shifted vs box64 (game lacks GL_GetDrawableSize
    // etc.).  So tab[box64_idx] is the WRONG function.  my2_SDL_CreateWindow then
    // called the wrong fn → returned garbage → Unity deref'd a bogus window
    // (SIGSEGV addr=0xffffffff).  Fix: use the game indices (same as the remap).
    if (470 < n) g_real_sdl_createwindow   = tab[470]; // game SDL_CreateWindow
    if (509 < n) g_real_sdl_gl_loadlibrary = tab[509]; // game SDL_GL_LoadLibrary

    // Remap is ONLY needed for old-SDL games (RimWorld 1.2 / Unity 2019) where
    // the bundled SDL had a different DYNAPI_PROC order than box64's SDL.
    // For new-SDL games (RimWorld 1.5+ / Unity 2022) the order is the same →
    // box64_idx == rw_idx → the bridges are already in the right slots →
    // remap would move them to the wrong (old) positions.
    // Controlled by RIMDROID_SDL_REMAP env: "0" = skip; default = apply.
    const char* remap_env = getenv("RIMDROID_SDL_REMAP");
    int do_remap = (!remap_env || strcmp(remap_env, "0") != 0);
    if (!do_remap) {
        printf_log(LOG_NONE, "RIMDROID REMAP: skipped (RIMDROID_SDL_REMAP=0, new-SDL game)\n");
        return 0;
    }
    // pass 1: undo the box64-indexed placement (restore the seeded RimWorld
    // static that genuinely belongs at that slot).  Done for ALL entries first
    // so overlaps between a box64 index and another entry's RimWorld index
    // (e.g. GetProcAddress@box64 515 vs CreateContext@rw 515) resolve correctly.
    for (int rj = 0; rj < rd_nremap; ++rj)
        if (rd_box64_idx[rj] != (uint32_t)-1)
            table[rd_box64_idx[rj]] = tab[rd_box64_idx[rj]];
    // pass 2: install our bridges at RimWorld's actual indices.
    for (int rj = 0; rj < rd_nremap; ++rj)
        if (rd_bridge[rj] && rd_remap[rj].rw_idx < n) {
            table[rd_remap[rj].rw_idx] = rd_bridge[rj];
            printf_log(LOG_NONE, "RIMDROID REMAP %s box64_idx=%u -> rw_idx=%u bridge=%p\n",
                       rd_remap[rj].name, rd_box64_idx[rj], rd_remap[rj].rw_idx, (void*)rd_bridge[rj]);
        }
    return 0;
}

char EXPORT *my2_SDL_GetBasePath(x64emu_t* emu) {
    char* p = strdup(emu->context->fullpath);
    char* b = strrchr(p, '/');
    if(b)
        *(b+1) = '\0';
    return p;
}

EXPORT void my2_SDL_LogCritical(x64emu_t* emu, int64_t cat, void* fmt, void *b) {
    // SDL_LOG_PRIORITY_CRITICAL == 6
    myStackAlign(emu, (const char*)fmt, b, emu->scratch, R_EAX, 2);
    PREPARE_VALIST;
    my->SDL_LogMessageV(cat, 6, fmt, VARARGS);
}

EXPORT void my2_SDL_LogError(x64emu_t* emu, int64_t cat, void* fmt, void *b) {
    // SDL_LOG_PRIORITY_ERROR == 5
    myStackAlign(emu, (const char*)fmt, b, emu->scratch, R_EAX, 2);
    PREPARE_VALIST;
    my->SDL_LogMessageV(cat, 5, fmt, VARARGS);
}

EXPORT void my2_SDL_LogWarn(x64emu_t* emu, int64_t cat, void* fmt, void *b) {
    // SDL_LOG_PRIORITY_WARN == 4
    myStackAlign(emu, (const char*)fmt, b, emu->scratch, R_EAX, 2);
    PREPARE_VALIST;
    my->SDL_LogMessageV(cat, 4, fmt, VARARGS);
}

EXPORT void my2_SDL_LogInfo(x64emu_t* emu, int64_t cat, void* fmt, void *b) {
    // SDL_LOG_PRIORITY_INFO == 3
    myStackAlign(emu, (const char*)fmt, b, emu->scratch, R_EAX, 2);
    PREPARE_VALIST;
    my->SDL_LogMessageV(cat, 3, fmt, VARARGS);
}

EXPORT void my2_SDL_LogDebug(x64emu_t* emu, int64_t cat, void* fmt, void *b) {
    // SDL_LOG_PRIORITY_DEBUG == 2
    myStackAlign(emu, (const char*)fmt, b, emu->scratch, R_EAX, 2);
    PREPARE_VALIST;
    my->SDL_LogMessageV(cat, 2, fmt, VARARGS);
}

EXPORT void my2_SDL_LogVerbose(x64emu_t* emu, int64_t cat, void* fmt, void *b) {
    // SDL_LOG_PRIORITY_VERBOSE == 1
    myStackAlign(emu, (const char*)fmt, b, emu->scratch, R_EAX, 2);
    PREPARE_VALIST;
    my->SDL_LogMessageV(cat, 1, fmt, VARARGS);
}

EXPORT void my2_SDL_Log(x64emu_t* emu, void* fmt, void *b) {
    // SDL_LOG_PRIORITY_INFO == 3
    // SDL_LOG_CATEGORY_APPLICATION == 0
    myStackAlign(emu, (const char*)fmt, b, emu->scratch, R_EAX, 1);
    PREPARE_VALIST;
    my->SDL_LogMessageV(0, 3, fmt, VARARGS);
}

// Shared GL proc-address resolver used by BOTH the SDL2 GL path (RimWorld 1.5) and the
// GLX path (RimWorld 1.6, wrappedlibgl.c my_glXGetProcAddress). Resolves a GL entry point
// from the active renderer (ZFA/OSMesa/GL4ES) via rimdroid_gl_proc_resolver and installs
// the critical zeroing-getter / no-op stubs for names libzfa.so does not export (a bare
// NULL pointer would make Unity jump to 0x0 or read garbage → GfxDevice teardown loop).
// NOT static: called from wrappedlibgl.c (same box64 .so). pa=NULL → use the resolver.
void* rimdroid_gl_getprocaddr(x64emu_t* emu, bridge_t* bridge, glprocaddress_t pa, const char* rname)
{
    if (!rname) return NULL;
    if (!pa) pa = rimdroid_gl_proc_resolver;
    // Arg-sanity shims (v13): hand back instrumented wrappers for the buffer-upload family.
    {
        wrapper_t w = NULL; void* fn = NULL;
        if      (!strcmp(rname, "glBufferStorage"))  { p_rd_real_glBufferStorage = rimdroid_gl_proc_resolver(rname); w = vFulpu; fn = (void*)rd_glBufferStorage; }
        else if (!strcmp(rname, "glBufferData"))     { p_rd_real_glBufferData    = rimdroid_gl_proc_resolver(rname); w = vFulpu; fn = (void*)rd_glBufferData; }
        else if (!strcmp(rname, "glBufferSubData"))  { p_rd_real_glBufferSubData = rimdroid_gl_proc_resolver(rname); w = vFullp; fn = (void*)rd_glBufferSubData; }
        else if (!strcmp(rname, "glMapBufferRange")) { p_rd_real_glMapBufferRange= rimdroid_gl_proc_resolver(rname); w = pFullu; fn = (void*)rd_glMapBufferRange; }
        else if (!strcmp(rname, "glTexStorage2D"))   { p_rd_real_glTexStorage2D  = rimdroid_gl_proc_resolver(rname); w = vFuiuii; fn = (void*)rd_glTexStorage2D; }
        else if (!strcmp(rname, "glTexImage2D"))     { p_rd_real_glTexImage2D    = rimdroid_gl_proc_resolver(rname); w = vFuiiiiiuup; fn = (void*)rd_glTexImage2D; }
        else if (!strcmp(rname, "glRenderbufferStorage")) { p_rd_real_glRenderbufferStorage = rimdroid_gl_proc_resolver(rname); w = vFuuii; fn = (void*)rd_glRenderbufferStorage; }
        else if (!strcmp(rname, "glRenderbufferStorageMultisample")) { p_rd_real_glRenderbufferStorageMultisample = rimdroid_gl_proc_resolver(rname); w = vFuiuii; fn = (void*)rd_glRenderbufferStorageMultisample; }
        else if (!strcmp(rname, "glTexStorage3D"))   { p_rd_real_glTexStorage3D  = rimdroid_gl_proc_resolver(rname); w = vFuiuiii; fn = (void*)rd_glTexStorage3D; }
        else if (!strcmp(rname, "glTexImage3D"))     { p_rd_real_glTexImage3D    = rimdroid_gl_proc_resolver(rname); w = vFuiiiiiiuup; fn = (void*)rd_glTexImage3D; }
        else if (!strcmp(rname, "glCompressedTexImage2D")) { p_rd_real_glCompressedTexImage2D = rimdroid_gl_proc_resolver(rname); w = vFuiuiiiip; fn = (void*)rd_glCompressedTexImage2D; }
        else if (!strcmp(rname, "glClientWaitSync")) { p_rd_real_glClientWaitSync = rimdroid_gl_proc_resolver(rname); w = uFpuU; fn = (void*)rd_glClientWaitSync; }
        else if (!strcmp(rname, "glDeleteSync"))     { p_rd_real_glDeleteSync     = rimdroid_gl_proc_resolver(rname); w = vFp;   fn = (void*)rd_glDeleteSync; }
        else if (!strcmp(rname, "glCopyImageSubData"))  { p_rd_real_glCopyImageSubData  = rimdroid_gl_proc_resolver(rname); w = vFuuiiiiuuiiiiiii; fn = (void*)rd_glCopyImageSubData; }
        else if (!strcmp(rname, "glCopyTexSubImage2D")) { p_rd_real_glCopyTexSubImage2D = rimdroid_gl_proc_resolver(rname); w = vFuiiiiiii; fn = (void*)rd_glCopyTexSubImage2D; }
        else if (!strcmp(rname, "glBlitFramebuffer"))   { p_rd_real_glBlitFramebuffer   = rimdroid_gl_proc_resolver(rname); w = vFiiiiiiiiuu; fn = (void*)rd_glBlitFramebuffer; }
        else if (!strcmp(rname, "glTexSubImage2D"))     { p_rd_real_glTexSubImage2D     = rimdroid_gl_proc_resolver(rname); w = vFuiiiiiuup; fn = (void*)rd_glTexSubImage2D; }
        else if (!strcmp(rname, "glCompressedTexSubImage2D")) { p_rd_real_glCompressedTexSubImage2D = rimdroid_gl_proc_resolver(rname); w = vFuiiiiiuip; fn = (void*)rd_glCompressedTexSubImage2D; }
        else if (!strcmp(rname, "glTexSubImage3D"))     { p_rd_real_glTexSubImage3D     = rimdroid_gl_proc_resolver(rname); w = vFuiiiiiiiuup; fn = (void*)rd_glTexSubImage3D; }
        else if (!strcmp(rname, "glDrawArrays"))        { p_rd_real_glDrawArrays        = rimdroid_gl_proc_resolver(rname); w = vFuii;  fn = (void*)rd_glDrawArrays; }
        else if (!strcmp(rname, "glDrawElements"))      { p_rd_real_glDrawElements      = rimdroid_gl_proc_resolver(rname); w = vFuiup; fn = (void*)rd_glDrawElements; }
        else if (!strcmp(rname, "glDrawElementsBaseVertex")) { p_rd_real_glDrawElementsBaseVertex = rimdroid_gl_proc_resolver(rname); w = vFuiupi; fn = (void*)rd_glDrawElementsBaseVertex; }
        else if (!strcmp(rname, "glDrawArraysInstanced")) { p_rd_real_glDrawArraysInstanced = rimdroid_gl_proc_resolver(rname); w = vFuiii; fn = (void*)rd_glDrawArraysInstanced; }
        else if (!strcmp(rname, "glDrawElementsInstanced")) { p_rd_real_glDrawElementsInstanced = rimdroid_gl_proc_resolver(rname); w = vFuiupi; fn = (void*)rd_glDrawElementsInstanced; }
        else if (!strcmp(rname, "glDrawElementsInstancedBaseVertex")) { p_rd_real_glDrawElementsInstancedBaseVertex = rimdroid_gl_proc_resolver(rname); w = vFuiupii; fn = (void*)rd_glDrawElementsInstancedBaseVertex; }
        else if (!strcmp(rname, "glDispatchCompute"))   { p_rd_real_glDispatchCompute   = rimdroid_gl_proc_resolver(rname); w = vFuuu; fn = (void*)rd_glDispatchCompute; }
        else if (!strcmp(rname, "glGenerateMipmap"))    { p_rd_real_glGenerateMipmap    = rimdroid_gl_proc_resolver(rname); w = vFu;   fn = (void*)rd_glGenerateMipmap; }
        else if (!strcmp(rname, "glUseProgram"))        { p_rd_real_glUseProgram        = rimdroid_gl_proc_resolver(rname); w = vFu;    fn = (void*)rd_glUseProgram; }
        else if (!strcmp(rname, "glShaderSource"))      { p_rd_real_glShaderSource      = rimdroid_gl_proc_resolver(rname); w = vFuipp; fn = (void*)rd_glShaderSource; }
        else if (!strcmp(rname, "glAttachShader"))      { p_rd_real_glAttachShader      = rimdroid_gl_proc_resolver(rname); w = vFuu;   fn = (void*)rd_glAttachShader; }

        if (fn && bridge) {
            void* b = (void*)AddBridge(bridge, w, fn, 0, rname);
            printf_log(LOG_NONE, "RIMDROID GLSANITY shim installed for %s\n", rname);
            return b;
        }
    }
    void* res = getGLProcAddress(emu, NULL, pa, rname);
    if (!res) {
        // libzfa lacks the symbol (Mesa still advertises it as core/extension).
        // GETTERS must zero/fill their output buffer (a bare no-op leaves it
        // uninitialised → Unity reads garbage → bad texture upload → crash).
        wrapper_t w = NULL; void* fn = NULL;
        if      (!strcmp(rname, "glGetInternalformativ"))        { w = vFuuuip; fn = (void*)rd_glGetInternalformativ; }
        else if (!strcmp(rname, "glGetTextureParameteriv"))      { w = vFuup;   fn = (void*)rd_glGetTextureParameteriv; }
        else if (!strcmp(rname, "glGetTextureLevelParameteriv")) { w = vFuiup;  fn = (void*)rd_glGetTextureLevelParameteriv; }
        else if (!strcmp(rname, "glGetQueryObjectui64v"))        { w = vFuup;   fn = (void*)rd_glGetQueryObjectui64v; }
        if (fn) {
            void* b = (void*)AddBridge(bridge, w, fn, 0, rname);
            printf_log(LOG_NONE, "RIMDROID GetProcAddress zeroing-stub for '%s' => %p\n", rname, b);
            return b;
        }
        // Everything else: a no-op (RAX=0, writes nothing) so a stray call does
        // not jump to 0x0.  Bridge created once and reused for every such name.
        static uintptr_t noop_bridge = 0;
        if (!noop_bridge)
            noop_bridge = AddBridge(bridge, iFv, (void*)rimdroid_gl_noop, 0, "rimdroid_gl_noop");
        printf_log(LOG_NONE, "RIMDROID GetProcAddress NULL for '%s' → no-op stub %p\n", rname, (void*)noop_bridge);
        res = (void*)noop_bridge;
    }
    return res;
}

EXPORT void* my2_SDL_GL_GetProcAddress(x64emu_t* emu, void* name)
{
    khint_t k;
    const char* rname = (const char*)name;
    printf_log(LOG_NONE, "RIMDROID SDL_GL_GetProcAddress('%s')\n", rname?rname:"(null)");
    // SDL_GL_GetProcAddress(NULL) must return NULL, not crash: getGLProcAddress()
    // hashes/strcmps the name and would dereference NULL.
    if (!rname)
        return NULL;
    static int lib_checked = 0;
    if(!lib_checked) {
        lib_checked = 1;
            // check if libGL is loaded, load it if not (helps some Haxe games, like DeadCells or Nuclear Blaze)
        if(!my_glhandle && !GetLibInternal(BOX64ENV(libgl)?BOX64ENV(libgl):"libGL.so.1"))
            // use a my_dlopen to actually open that lib, like SDL2 is doing...
            my_glhandle = my_dlopen(emu, BOX64ENV(libgl)?BOX64ENV(libgl):"libGL.so.1", RTLD_LAZY|RTLD_GLOBAL);
    }
    // With no real ARM64 libSDL2, my->SDL_GL_GetProcAddress is NULL.  Passing it
    // to getGLProcAddress() would call NULL → SIGSEGV.  Fall back to resolving
    // GL entry points straight from libgl4es.so.
    // DIAGNOSTIC: hand Unity our instrumented glTexSubImage2D (the crash site)
    // instead of the real one, to log args + PBO binding before the upload.
    if (!strcmp(rname, "glTexSubImage2D")) {
        static uintptr_t b = 0;
        if (!b) b = AddBridge(my_lib->w.bridge, vFuiiiiiuup, (void*)rd_glTexSubImage2D, 0, "rd_glTexSubImage2D");
        printf_log(LOG_NONE, "RIMDROID GetProcAddress('glTexSubImage2D') → instrumented %p\n", (void*)b);
        return (void*)b;
    }
    glprocaddress_t pa = (glprocaddress_t)my->SDL_GL_GetProcAddress;
    return rimdroid_gl_getprocaddr(emu, my_lib->w.bridge, pa, rname);
}

#define nb_once 16
typedef void(*sdl2_tls_dtor)(void*);
static uintptr_t dtor_emu[nb_once] = {0};
static void tls_dtor_callback(int n, void* a)
{
    if(dtor_emu[n]) {
        RunFunctionFmt(dtor_emu[n], "p", a);
    }
}
#define GO(N) \
void tls_dtor_callback_##N(void* a) \
{ \
    tls_dtor_callback(N, a); \
}

GO(0)
GO(1)
GO(2)
GO(3)
GO(4)
GO(5)
GO(6)
GO(7)
GO(8)
GO(9)
GO(10)
GO(11)
GO(12)
GO(13)
GO(14)
GO(15)
#undef GO
static const sdl2_tls_dtor dtor_cb[nb_once] = {
     tls_dtor_callback_0, tls_dtor_callback_1, tls_dtor_callback_2, tls_dtor_callback_3
    ,tls_dtor_callback_4, tls_dtor_callback_5, tls_dtor_callback_6, tls_dtor_callback_7
    ,tls_dtor_callback_8, tls_dtor_callback_9, tls_dtor_callback_10,tls_dtor_callback_11
    ,tls_dtor_callback_12,tls_dtor_callback_13,tls_dtor_callback_14,tls_dtor_callback_15
};
EXPORT int64_t my2_SDL_TLSSet(x64emu_t* emu, uint64_t id, void* value, void* dtor)
{
    if(!dtor)
        return my->SDL_TLSSet(id, value, NULL);
    int n = 0;
    while (n<nb_once) {
        if(!dtor_emu[n] || (dtor_emu[n])==((uintptr_t)dtor)) {
            dtor_emu[n] = (uintptr_t)dtor;
            return my->SDL_TLSSet(id, value, dtor_cb[n]);
        }
        ++n;
    }
    printf_log(LOG_NONE, "Error: SDL2 SDL_TLSSet with destructor: no more slot!\n");
    //emu->quit = 1;
    return -1;
}

EXPORT void my2_SDL_AddEventWatch(x64emu_t* emu, void* p, void* userdata)
{
    my->SDL_AddEventWatch(find_eventfilter_Fct(p), userdata);
}
EXPORT void my2_SDL_DelEventWatch(x64emu_t* emu, void* p, void* userdata)
{
    my->SDL_DelEventWatch(find_eventfilter_Fct(p), userdata);
}

EXPORT void* my2_SDL_LoadObject(x64emu_t* emu, void* sofile)
{
    return my_dlopen(emu, sofile, 0);   // TODO: check correct flag value...
}
EXPORT void my2_SDL_UnloadObject(x64emu_t* emu, void* handle)
{
    my_dlclose(emu, handle);
}
EXPORT void* my2_SDL_LoadFunction(x64emu_t* emu, void* handle, void* name)
{
    return my_dlsym(emu, handle, name);
}

EXPORT int64_t my2_SDL_IsJoystickPS4(x64emu_t* emu, uint16_t vendor, uint16_t product_id)
{
    if(my->SDL_IsJoystickPS4)
        return my->SDL_IsJoystickPS4(vendor, product_id);
    // fallback
    return 0;
}
EXPORT int64_t my2_SDL_IsJoystickNintendoSwitchPro(x64emu_t* emu, uint16_t vendor, uint16_t product_id)
{
    if(my->SDL_IsJoystickNintendoSwitchPro)
        return my->SDL_IsJoystickNintendoSwitchPro(vendor, product_id);
    // fallback
    return 0;
}
EXPORT int64_t my2_SDL_IsJoystickSteamController(x64emu_t* emu, uint16_t vendor, uint16_t product_id)
{
    if(my->SDL_IsJoystickSteamController)
        return my->SDL_IsJoystickSteamController(vendor, product_id);
    // fallback
    return 0;
}
EXPORT int64_t my2_SDL_IsJoystickXbox360(x64emu_t* emu, uint16_t vendor, uint16_t product_id)
{
    if(my->SDL_IsJoystickXbox360)
        return my->SDL_IsJoystickXbox360(vendor, product_id);
    // fallback
    return 0;
}
EXPORT int64_t my2_SDL_IsJoystickXboxOne(x64emu_t* emu, uint16_t vendor, uint16_t product_id)
{
    if(my->SDL_IsJoystickXboxOne)
        return my->SDL_IsJoystickXboxOne(vendor, product_id);
    // fallback
    return 0;
}
EXPORT int64_t my2_SDL_IsJoystickXInput(x64emu_t* emu, uint64_t a, uint64_t b)
{
    if(my->SDL_IsJoystickXInput)
        return my->SDL_IsJoystickXInput(a, b);
    // fallback
    return 0;
}
EXPORT int64_t my2_SDL_IsJoystickHIDAPI(x64emu_t* emu, uint64_t a, uint64_t b)
{
    if(my->SDL_IsJoystickHIDAPI)
        return my->SDL_IsJoystickHIDAPI(a, b);
    // fallback
    return 0;
}

void* my_vkGetInstanceProcAddr(x64emu_t* emu, void* device, void* name);
EXPORT void* my2_SDL_Vulkan_GetVkGetInstanceProcAddr(x64emu_t* emu)
{
    void* procaddr = my->SDL_Vulkan_GetVkGetInstanceProcAddr();
    if(!emu->context->vkprocaddress)
        emu->context->vkprocaddress = (vkprocaddess_t)procaddr;

    if(procaddr)
        return (void*)AddCheckBridge2(my_lib->w.bridge, pFEpp, my_vkGetInstanceProcAddr, procaddr, 0, "vkGetInstanceProcAddr");
    return NULL;
}

EXPORT void my2_SDL_GetJoystickGUIDInfo(SDL_JoystickGUID guid, uint16_t *vend, uint16_t *prod, uint16_t *ver, uint16_t* crc16)
{
    uint16_t dummy = 0;
    if(my->SDL_GetJoystickGUIDInfo)
        my->SDL_GetJoystickGUIDInfo(guid, vend, prod, ver, BOX64ENV(sdl2_jguid)?(&dummy):crc16);
    // fallback
    else {
        uint16_t *guid16 = (uint16_t *)guid.data;
        if (guid16[1]==0x0000 && guid16[3]==0x0000 && guid16[5]==0x0000)
            {
            if(vend) *vend = guid16[2];
            if(prod) *prod = guid16[4];
            if(ver)  *ver  = guid16[6];
        } else {
            if(vend) *vend = 0;
            if(prod) *prod = 0;
            if(ver)  *ver  = 0;
        }
    }
}

EXPORT unsigned long my2_SDL_GetThreadID(x64emu_t* emu, void* thread)
{
    unsigned long ret = my->SDL_GetThreadID(thread);
    int max = 10;
    while (!ret && max--) {
        sched_yield();
        ret = my->SDL_GetThreadID(thread);
    }
    return ret;
}

EXPORT int my2_SDL_GetCPUCount(x64emu_t* emu)
{
    int ret = my->SDL_GetCPUCount();
    if(BOX64ENV(maxcpu) && ret>BOX64ENV(maxcpu))
        ret = BOX64ENV(maxcpu);
    return ret;
}

struct my_FilterEvents_data {
    uintptr_t callback;
    void *userdata;
};
static int my_FilterEvents_callback(struct my_FilterEvents_data* data, void* event) {
    return (int) RunFunctionFmt(data->callback, "pp", data->userdata, event);
}
EXPORT void my2_SDL_FilterEvents(x64emu_t* emu, void* filter, void* userdata) {
    struct my_FilterEvents_data data = {
        .callback = (uintptr_t) filter,
        .userdata = userdata,
    };
    my->SDL_FilterEvents(my_FilterEvents_callback, &data);
}

// ---- GL4ES / EGL intercepts -------------------------------------------------
// These my2_ overrides replace the SDL2 GL context functions so that, when
// Unity calls SDL_GL_CreateContext() (via box64's SDL2 wrapper), we return the
// EGL context that rimdroid.c already created with GL4ES's ANativeWindow.
// Without this, SDL_VIDEODRIVER=dummy returns NULL from CreateContext and Unity
// prints "Unable to find a supported OpenGL core profile".

// EGL handles set up by rimdroid_init_gl4es_egl() before launch_rimworld_elf().
// Declared __weak so the standalone box64 executable links without error
// (symbols stay NULL there).  In librimdroidlinker.so the strong definitions
// from librimdroid.so override these at runtime.
extern __attribute__((weak)) void* g_egl_display;
extern __attribute__((weak)) void* g_egl_surface;
extern __attribute__((weak)) void* g_egl_context;

// ZINK_ZFA renderer: ZFA context created by rimdroid.c in the parent (Zink over
// Vulkan/Turnip).  When g_zfa_context is set we route the GL context calls to
// ZFA instead of EGL.  GL proc resolution already works via BOX64_LIBGL
// (=libzfa.so) in rimdroid_gl_proc_resolver, so SDL_GL_GetProcAddress is
// unchanged.  rimdroid_zfa_make_current/swap live in rimdroid.c (strong defs).
extern __attribute__((weak)) void* g_zfa_context;
extern __attribute__((weak)) int  rimdroid_zfa_make_current(void);
extern __attribute__((weak)) void rimdroid_zfa_swap(void);
extern __attribute__((weak)) int  rimdroid_zfa_release_current(void);

// RD_SOFTPIPE renderer: OSMesa (CPU softpipe) OFFSCREEN context created by
// rimdroid.c (rimdroid_init_osmesa). g_osmesa_context!=NULL selects the software
// path in the my2_SDL_GL_* handlers below. Unlike ZFA (which auto-presents via
// its kopper/Vulkan swapchain), OSMesa has no winsys, so my2_SDL_GL_SwapWindow
// MUST call rimdroid_osmesa_swap() to blit the CPU buffer to the ANativeWindow —
// it is the one and only present path for softpipe. make_current binds the
// context + buffer to the calling thread. Both live in rimdroid.c (strong defs).
extern __attribute__((weak)) void* g_osmesa_context;
extern __attribute__((weak)) int  rimdroid_osmesa_make_current(void);
extern __attribute__((weak)) int  rimdroid_osmesa_make_current_ctx(void* ctx);
extern __attribute__((weak)) void* rimdroid_osmesa_create_shared(void);
extern __attribute__((weak)) void rimdroid_osmesa_swap(void);

// Lazy-open libEGL.so and cache the handle.
static void* rimdroid_libegl(void) {
    static void* h = NULL;
    if (!h) {
        h = dlopen("libEGL.so", RTLD_LAZY | RTLD_GLOBAL);
        if (!h) h = dlopen("libEGL.so.1", RTLD_LAZY | RTLD_GLOBAL);
    }
    return h;
}

// eglMakeCurrent(display, draw, read, ctx) — make our context current on this thread.
static int rimdroid_egl_make_current(void) {
    void* h = rimdroid_libegl();
    if (!h || !g_egl_display || !g_egl_surface || !g_egl_context) return 0;
    typedef unsigned int (*fn_t)(void*, void*, void*, void*);
    fn_t fn = (fn_t)(uintptr_t)dlsym(h, "eglMakeCurrent");
    return fn ? (int)fn(g_egl_display, g_egl_surface, g_egl_surface, g_egl_context) : 0;
}

// SDL_CreateWindow(title,x,y,w,h,flags) → strip SDL_WINDOW_OPENGL and call the
// real (dummy-driver) SDL_CreateWindow.  The dummy driver cannot do GL, so a window
// requested WITH SDL_WINDOW_OPENGL makes SDL internally call SDL_GL_LoadLibrary →
// fails → returns NULL → Unity's renderer detection bails before SDL_GL_CreateContext.
// We hand Unity a valid (non-GL) window and provide the GL context separately via ZFA.
#define RD_SDL_WINDOW_OPENGL 0x00000002u
#define RD_SDL_WINDOW_SHOWN  0x00000004u
// Window invariants Unity queries (we strip OPENGL at creation so the dummy
// driver succeeds, but Unity must still SEE a GL window of the right size or it
// builds a degenerate GfxDevice render-target graph whose cleanup recurses
// forever — the SDL_GL_DeleteContext(NULL) loop).  Capture the ORIGINAL flags
// (incl. OPENGL) + requested size at creation and report THOSE back.
static uint32_t g_window_flags = RD_SDL_WINDOW_OPENGL | RD_SDL_WINDOW_SHOWN;
static int      g_window_w = 2340, g_window_h = 1080;  // native default (overwritten by CreateWindow)

// SDL_GetWindowFlags(window) → report the GL window flags Unity asked for
// (the real dummy window lacks SDL_WINDOW_OPENGL because we stripped it).
EXPORT uint32_t my2_SDL_GetWindowFlags(void* window) {
    uint32_t f = g_window_flags | RD_SDL_WINDOW_OPENGL | RD_SDL_WINDOW_SHOWN;
    static int n = 0;
    if (n < 4) { n++; printf_log(LOG_NONE, "RIMDROID SDL_GetWindowFlags(win=%p) => 0x%x\n", window, f); }
    return f;
}
// SDL_GetWindowSize(window, w, h) → report the requested size (dummy window may
// report 0x0 → Unity makes a 0x0 framebuffer).
EXPORT void my2_SDL_GetWindowSize(void* window, int* w, int* h) {
    if (w) *w = g_window_w;
    if (h) *h = g_window_h;
    static int n = 0;
    if (n < 4) { n++; printf_log(LOG_NONE, "RIMDROID SDL_GetWindowSize(win=%p) => %dx%d\n", window, g_window_w, g_window_h); }
}

// TEST (2026-05-30): consistent 1024x768 + 60 Hz.  Keep size at 1024x768 (matches
// window/FBO/dummy) but force refresh_rate=60 instead of dummy's 0 — testing
// whether "@ 0 Hz" (not the size) is the fullscreen-teardown trigger.
typedef struct { uint32_t format; int32_t w; int32_t h; int32_t refresh_rate; void* driverdata; } rd_SDL_DisplayMode;
static void rd_fill_display_mode(void* mode) {
    if (!mode) return;
    rd_SDL_DisplayMode* m = (rd_SDL_DisplayMode*)mode;
    m->format = 0x16161804;   // SDL_PIXELFORMAT_RGB888
    m->w = g_window_w; m->h = g_window_h;  // NATIVE (default 2340x1080). Reporting a
                                           // fake 1024x768 desktop made Unity render
                                           // 1024x768 into the native buffer → tiny
                                           // image in a corner. Match the real size.
    m->refresh_rate = 60;     // 60 Hz (dummy reports 0)
    m->driverdata = NULL;
}
EXPORT int my2_SDL_GetDesktopDisplayMode(int displayIndex, void* mode) {
    rd_fill_display_mode(mode);
    static int n = 0;
    if (n < 4) { n++; printf_log(LOG_NONE, "RIMDROID GetDesktopDisplayMode(%d) => 1024x768@60\n", displayIndex); }
    return 0;
}
EXPORT int my2_SDL_GetCurrentDisplayMode(int displayIndex, void* mode) {
    rd_fill_display_mode(mode);
    static int n = 0;
    if (n < 4) { n++; printf_log(LOG_NONE, "RIMDROID GetCurrentDisplayMode(%d) => 1024x768@60\n", displayIndex); }
    return 0;
}

EXPORT void* my2_SDL_CreateWindow(x64emu_t* emu, void* title, int x, int y, int w, int h, uint32_t flags) {
    uint32_t stripped = flags & ~RD_SDL_WINDOW_OPENGL;
    g_window_flags = flags; g_window_w = w; g_window_h = h;   // report these back to Unity
    if (!g_real_sdl_createwindow) {
        printf_log(LOG_NONE, "RIMDROID SDL_CreateWindow: no real fn captured!\n");
        return NULL;
    }
    printf_log(LOG_NONE, "RIMDROID SDL_CreateWindow(flags=0x%x→0x%x) %dx%d\n", flags, stripped, w, h);
    return (void*)(uintptr_t)RunFunctionWithEmu(emu, 0, g_real_sdl_createwindow, 6,
        (uint64_t)(uintptr_t)title, (uint64_t)(int64_t)x, (uint64_t)(int64_t)y,
        (uint64_t)(int64_t)w, (uint64_t)(int64_t)h, (uint64_t)stripped);
}

// SDL_GL_CreateContext(window) → return our pre-created context, bind to thread.
// The SDL_Window* Unity created its GL context for (captured in CreateContext);
// returned by my2_SDL_GL_GetCurrentWindow so Unity's "is a window/context
// current?" checks see a valid current window even though we bypass SDL's GL.
static void* g_gl_window = NULL;
// The context handle Unity currently considers "current" (a distinct opaque alias
// per CreateContext — see my2_SDL_GL_CreateContext).  Returned by GetCurrentContext.
static void* g_rd_ctx_current = NULL;
// SOFTPIPE: the first SDL_GL_CreateContext reuses the primary OSMesa context (made at init);
// later ones get their own SHARED context, so Unity's 2nd (shared) GL context has separate state.
static int g_sp_primary_used = 0;
// DEBUG counters to characterise the DeleteContext loop's GL call mix.
static unsigned long g_cnt_getctx=0, g_cnt_getwin=0, g_cnt_makecur=0,
                     g_cnt_swap=0, g_cnt_create=0, g_cnt_setattr=0, g_cnt_getattr=0;

// SDL_GL_GetCurrentContext / GetCurrentWindow read SDL's per-thread "current GL"
// TLS, which is set by SDL_GL_MakeCurrent.  We OWN the context via ZFA and never
// call the real SDL_GL_MakeCurrent, so that TLS stays NULL → Unity sees "no
// current context", believes the device was lost, and loops destroying/recreating
// it forever (observed: endless SDL_GL_DeleteContext(0x0), no frame).  Override
// both to report our context/window as current.
EXPORT void* my2_SDL_GL_GetCurrentContext(void) {
    g_cnt_getctx++;
    if (&g_osmesa_context && g_osmesa_context) return g_rd_ctx_current ? g_rd_ctx_current : g_osmesa_context;
    if (&g_zfa_context && g_zfa_context) return g_rd_ctx_current ? g_rd_ctx_current : g_zfa_context;
    if (g_egl_context) return g_egl_context;
    if (my->SDL_GL_GetCurrentContext) return my->SDL_GL_GetCurrentContext();
    return NULL;
}
EXPORT void* my2_SDL_GL_GetCurrentWindow(void) {
    g_cnt_getwin++;
    if (g_gl_window) return g_gl_window;
    if (my->SDL_GL_GetCurrentWindow) return my->SDL_GL_GetCurrentWindow();
    return NULL;
}

EXPORT void* my2_SDL_GL_CreateContext(void* win) {
    g_cnt_create++;
    g_gl_window = win;
    if (&g_osmesa_context && g_osmesa_context) {
        // Give each Unity GL context its OWN OSMesa context (separate GL state) that SHARES
        // resources with the primary — Unity's GLCore path makes a 2nd "shared" context, and
        // collapsing both onto one OSMesa context crossed their states → the menu rendered into
        // nothing (black). First call reuses the primary (made at init); later calls create a
        // shared one. The real, distinct OSMesa context IS the handle Unity holds.
        void* ctx;
        if (!g_sp_primary_used) {
            ctx = g_osmesa_context;
            g_sp_primary_used = 1;
        } else if (&rimdroid_osmesa_create_shared && rimdroid_osmesa_create_shared) {
            ctx = rimdroid_osmesa_create_shared();
        } else {
            ctx = g_osmesa_context;
        }
        if (!ctx) ctx = g_osmesa_context;
        if (&rimdroid_osmesa_make_current_ctx && rimdroid_osmesa_make_current_ctx)
            rimdroid_osmesa_make_current_ctx(ctx);
        g_rd_ctx_current = ctx;
        printf_log(LOG_NONE, "RIMDROID SDL_GL_CreateContext #%lu → OSMesa ctx %p win=%p tid=%ld — softpipe\n", g_cnt_create, ctx, win, (long)syscall(SYS_gettid));
        return ctx;
    }
    if (&g_zfa_context && g_zfa_context) {
        if (rimdroid_zfa_make_current) rimdroid_zfa_make_current();
        // Hand back a DISTINCT opaque handle per call (all alias the one ZFA ctx).
        // Unity 2019 GLCore creates a 2nd (shared) context; returning the SAME
        // pointer twice (create=2) confused its context bookkeeping → endless
        // DeleteContext teardown of NULL-ctx nodes. Distinct handles keep its
        // bookkeeping consistent. They are opaque to Unity (compared/passed only).
        void* fake = (void*)((uintptr_t)g_zfa_context + (uintptr_t)(g_cnt_create * 0x10000UL));
        g_rd_ctx_current = fake;
        printf_log(LOG_NONE, "RIMDROID SDL_GL_CreateContext #%lu → handle %p (ZFA %p) win=%p tid=%ld — MILESTONE reached\n", g_cnt_create, fake, g_zfa_context, win, (long)syscall(SYS_gettid));
        return fake;
    }
    if (g_egl_context) {
        rimdroid_egl_make_current();
        printf_log(LOG_NONE, "RIMDROID SDL_GL_CreateContext → EGL ctx %p (GL4ES) — MILESTONE reached\n", g_egl_context);
        return g_egl_context;
    }
    // No renderer context set up; fall back to SDL.
    return my->SDL_GL_CreateContext(win);
}

// SDL_GL_MakeCurrent(window, ctx) → re-bind our context on the calling thread.
// In ZFA/EGL mode WE own the GL context: the SDL dummy driver has no GL at all,
// so we must NEVER delegate to my->SDL_GL_MakeCurrent (with SDL_DYNAMIC_API that
// would route back through the jump_table into us, or dive into the dummy
// driver's NULL GL hooks → crash).  ctx==NULL is an unbind → succeed as a no-op
// (our single context just stays alive; the next bind re-routes here).
EXPORT int my2_SDL_GL_MakeCurrent(void* win, void* ctx) {
    g_cnt_makecur++;
    (void)win;
    printf_log(LOG_NONE, "RIMDROID SDL_GL_MakeCurrent(win=%p ctx=%p) tid=%ld zfa=%p egl=%p\n",
               win, ctx, (long)syscall(SYS_gettid), (&g_zfa_context)?g_zfa_context:NULL, g_egl_context);
    if (&g_osmesa_context && g_osmesa_context) {
        // OSMesaMakeCurrent rebinds the context (+ CPU buffer) to whatever thread
        // calls it, so Unity's main↔render-worker handoff just works: ctx==NULL is
        // an unbind (no-op — the context migrates on the next bind), ctx!=NULL
        // rebinds here. No cross-thread st_context hazard like Zink (each
        // OSMesaMakeCurrent fully (re)binds), so no release dance is needed.
        if (!ctx) { g_rd_ctx_current = NULL; return 0; }
        g_rd_ctx_current = ctx;
        // Bind the SPECIFIC context Unity asked for (ctx == a real OSMesa context handle we returned
        // from CreateContext) so its separate GL state is the one active. Fall back to the primary
        // binder if the per-ctx entry point isn't linked.
        int ok = (&rimdroid_osmesa_make_current_ctx && rimdroid_osmesa_make_current_ctx)
                 ? rimdroid_osmesa_make_current_ctx(ctx)
                 : (rimdroid_osmesa_make_current && rimdroid_osmesa_make_current());
        printf_log(LOG_NONE, "RIMDROID SDL_GL_MakeCurrent → OSMesa rebind ctx=%p %s\n", ctx, ok ? "OK" : "FAIL");
        return ok ? 0 : -1;
    }
    if (&g_zfa_context && g_zfa_context) {
        if (!ctx) {
            // CONFIRMED ROOT CAUSE (tid log): Unity's threaded renderer hands the
            // GL context between the main thread and a render worker via
            // MakeCurrent(NULL)[release-here] then MakeCurrent(ctx)[acquire-there].
            // A no-op here left the ZFA/Zink st_context "current" on the donor
            // thread too → one context current on TWO threads → concurrent pipe_context
            // use → device-lost → infinite teardown.  Honour the unbind: release the
            // st_context from THIS thread via libzfa's zfaReleaseCurrent
            // (= st_api_make_current(NULL,NULL,NULL)).  The handoff is serialized, so
            // a single context legally migrates between threads.
            // NOTE: no-op (rel=-1) until a rebuilt libzfa exporting zfaReleaseCurrent
            // is installed — then this serializes ownership and should kill the loop.
            int rel = (rimdroid_zfa_release_current) ? rimdroid_zfa_release_current() : -1;
            g_rd_ctx_current = NULL;
            printf_log(LOG_NONE, "RIMDROID SDL_GL_MakeCurrent → ZFA unbind (release=%d)\n", rel);
            return 0;
        }
        g_rd_ctx_current = ctx;   // track what Unity now considers the current context
        int ok = (rimdroid_zfa_make_current && rimdroid_zfa_make_current());
        printf_log(LOG_NONE, "RIMDROID SDL_GL_MakeCurrent → ZFA rebind %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : -1;
    }
    if (&g_egl_context && g_egl_context) {
        if (!ctx) return 0;  // unbind no-op
        return rimdroid_egl_make_current() ? 0 : -1;
    }
    return my->SDL_GL_MakeCurrent(win, ctx);
}

// SDL_GL_SwapWindow(window) → present the frame (ZFA flush or eglSwapBuffers).
// FPS overlay: librimdroid counts presented frames; the Java overlay polls it.
// Weak so a box64 build without librimdroid still links (call is then skipped).
extern __attribute__((weak)) void rimdroid_frame_tick(void);
EXPORT void my2_SDL_GL_SwapWindow(void* win) {
    g_cnt_swap++;
    static int rd_fpstick_off = -1;   // TEST toggle (env RIMDROID_NO_FPSTICK=1): skip the FPS frame counter
    if (rd_fpstick_off < 0) rd_fpstick_off = getenv("RIMDROID_NO_FPSTICK") ? 1 : 0;
    if (rimdroid_frame_tick && !rd_fpstick_off) rimdroid_frame_tick();   // count this present (any renderer)
    rd_upload_pace_frame_reset();   // pacing only guards SINGLE-frame upload bursts (see def)
    (void)win;
    printf_log(LOG_NONE, "RIMDROID SDL_GL_SwapWindow(win=%p)\n", win);
    if (&g_osmesa_context && g_osmesa_context) {
        // The ONLY present path for softpipe: glFinish + blit CPU buffer → surface.
        if (rimdroid_osmesa_swap) rimdroid_osmesa_swap();
        return;
    }
    if (&g_zfa_context && g_zfa_context) {
        if (rimdroid_zfa_swap) rimdroid_zfa_swap();
        return;
    }
    if (g_egl_display && g_egl_surface) {
        void* h = rimdroid_libegl();
        if (h) {
            typedef unsigned int (*fn_t)(void*, void*);
            fn_t fn = (fn_t)(uintptr_t)dlsym(h, "eglSwapBuffers");
            if (fn) { fn(g_egl_display, g_egl_surface); return; }
        }
    }
    my->SDL_GL_SwapWindow(win);
}

// SDL_GL_DeleteContext(ctx) → no-op for our context (it outlives the game loop).
// ===================== RimDroid injected input (Phase A) =====================
// The injected-event ring lives in librimdroid.so (rimdroid.c) because box64 is a
// SEPARATE .so and only the reverse (box64 weakly referencing rimdroid symbols)
// links. rd_input_poll() fills the caller's x86_64 SDL_Event (56 bytes) from the
// ring and returns 1, else 0. We drain it before the real (dummy, empty) PollEvent.
extern __attribute__((weak)) int rd_input_poll(unsigned char* out56);

EXPORT int my2_SDL_PollEvent(void* event) {
    if (event && rd_input_poll && rd_input_poll((unsigned char*)event))
        return 1;
    if (my->SDL_PollEvent) return my->SDL_PollEvent(event);
    return 0;
}

// RimWorld polls the mouse position/buttons via SDL_GetMouseState for things like
// selection-drag and right-click targeting. The real (dummy) SDL doesn't know our
// injected cursor, so return it from rimdroid.c (weak rd_input_get_mouse).
extern __attribute__((weak)) unsigned int rd_input_get_mouse(int* x, int* y);
EXPORT uint32_t my2_SDL_GetMouseState(void* x, void* y) {
    if (rd_input_get_mouse) return rd_input_get_mouse((int*)x, (int*)y);
    if (my->SDL_GetMouseState) return my->SDL_GetMouseState(x, y);
    return 0;
}
// ============================================================================

EXPORT void my2_SDL_GL_DeleteContext(x64emu_t* emu, void* ctx) {
    (void)emu;
    // Rate-limited counter only (per-call logging flooded the log + slowed the
    // run). Lets us see whether DeleteContext calls keep growing (loop) or stop.
    static unsigned long rd_dc_count = 0;
    if ((++rd_dc_count % 20000UL) == 1UL)
        printf_log(LOG_NONE, "RIMDROID DeleteCtx#%lu | create=%lu makecur=%lu swap=%lu getctx=%lu getwin=%lu setattr=%lu getattr=%lu\n",
                   rd_dc_count, g_cnt_create, g_cnt_makecur, g_cnt_swap, g_cnt_getctx, g_cnt_getwin, g_cnt_setattr, g_cnt_getattr);
    // On the VERY FIRST DeleteContext (= start of the teardown loop), dump the
    // guest stack so we can see which UnityPlayer.so function initiated it. We
    // scan upward from the guest RSP (committed stack, safe to read) and print
    // any slot that resolves to a code address inside a loaded ELF (filtered via
    // FindElfAddress), naming it with getAddrFunctionName(). Runs exactly once.
    if (rd_dc_count == 1UL) {
        long rd_tid = (long)syscall(SYS_gettid);
        printf_log(LOG_NONE, "RIMDROID ===== DeleteCtx#1 GUEST STACK SCAN tid=%ld RIP=%p RSP=%p RBP=%p ctx=%p =====\n",
                   rd_tid, (void*)R_RIP, (void*)R_RSP, (void*)R_RBP, ctx);
        uintptr_t rd_sp = R_RSP;
        uintptr_t rd_lastval = 0;
        int rd_printed = 0;
        for (int i = 0; i < 1024 && rd_printed < 48; i++) {
            uintptr_t val = *(uintptr_t*)(rd_sp + (uintptr_t)i * 8u);
            if (!val || val == rd_lastval) continue;
            if (!FindElfAddress(my_context, val)) continue;
            rd_lastval = val;
            printf_log(LOG_NONE, "  [sp+0x%04x] %p  %s\n",
                       (unsigned)(i * 8), (void*)val, getAddrFunctionName(val));
            rd_printed++;
        }
        printf_log(LOG_NONE, "RIMDROID ===== DeleteCtx#1 STACK SCAN END (%d entries) =====\n", rd_printed);
    }
    // In ZFA/EGL mode WE own the GL context (created via zfaCreateContext /
    // eglCreateContext, NOT via SDL).  There is no real SDL GL context to
    // delete, and my->SDL_GL_DeleteContext is NULL for the statically-linked
    // game — so falling through to it (which the old `ctx && ctx==g_zfa_context`
    // guards did for ctx==NULL, as Unity passes on scene transition / teardown)
    // jumps to a NULL function pointer and SIGSEGVs at addr=0x0.  Whenever we
    // have our own context, this is ALWAYS a no-op; the context outlives the
    // game loop and is torn down by us at process exit.
    if (g_zfa_context || g_egl_context || g_osmesa_context) {
        printf_log(LOG_INFO, "SDL_GL_DeleteContext: ZFA/EGL/OSMesa no-op (ctx=%p, keeping our context)\n", ctx);
        return;
    }
    if (my->SDL_GL_DeleteContext)
        my->SDL_GL_DeleteContext(ctx);
}

// SDL_VIDEODRIVER=dummy reports no GL capability, so Unity's renderer-detection
// (LinuxStandalone main.cpp:623) bails BEFORE ever calling SDL_GL_CreateContext —
// our context (ZFA/EGL) is never reached and Unity prints "No supported renderers".
// These overrides make the detection believe a real OpenGL CORE 3.2+ context is
// available, so Unity proceeds to SDL_GL_CreateContext (→ ZFA/EGL).
// SDL_GLattr enum values:
#define RD_GL_RED_SIZE              0
#define RD_GL_GREEN_SIZE            1
#define RD_GL_BLUE_SIZE             2
#define RD_GL_ALPHA_SIZE            3
#define RD_GL_DOUBLEBUFFER          5
#define RD_GL_DEPTH_SIZE            6
#define RD_GL_STENCIL_SIZE          7
#define RD_GL_CONTEXT_MAJOR_VERSION 17
#define RD_GL_CONTEXT_MINOR_VERSION 18
#define RD_GL_CONTEXT_FLAGS         20
#define RD_GL_CONTEXT_PROFILE_MASK  21
#define RD_GL_CONTEXT_PROFILE_CORE  0x0001

// SDL_GL_LoadLibrary(path) → OBSERVE-ONLY: log + passthrough to the real static
// function (so behaviour is unchanged — no infinite loop), to map the detection.
EXPORT int my2_SDL_GL_LoadLibrary(x64emu_t* emu, void* path) {
    int r = -999;
    if (g_real_sdl_gl_loadlibrary)
        r = (int)RunFunctionWithEmu(emu, 0, g_real_sdl_gl_loadlibrary, 1, (uint64_t)(uintptr_t)path);
    printf_log(LOG_NONE, "RIMDROID SDL_GL_LoadLibrary(path=%p) => %d (real passthrough)\n", path, r);
    return r;
}

// SDL_GL_SetAttribute(attr, value) → accept any requested attribute.
EXPORT int my2_SDL_GL_SetAttribute(uint32_t attr, int value) {
    (void)attr; (void)value;
    return 0;
}

// SDL_GL_GetAttribute(attr, int* value) → report a real GL 4.3 CORE profile.
EXPORT int my2_SDL_GL_GetAttribute(uint32_t attr, void* value) {
    if (!value) return -1;
    int* v = (int*)value;
    switch (attr) {
        case RD_GL_CONTEXT_MAJOR_VERSION: *v = 4; break;
        case RD_GL_CONTEXT_MINOR_VERSION: *v = 3; break;
        case RD_GL_CONTEXT_PROFILE_MASK:  *v = RD_GL_CONTEXT_PROFILE_CORE; break;
        case RD_GL_CONTEXT_FLAGS:         *v = 0; break;
        case RD_GL_DEPTH_SIZE:            *v = 24; break;
        case RD_GL_STENCIL_SIZE:          *v = 8;  break;
        case RD_GL_RED_SIZE: case RD_GL_GREEN_SIZE:
        case RD_GL_BLUE_SIZE: case RD_GL_ALPHA_SIZE: *v = 8; break;
        case RD_GL_DOUBLEBUFFER:          *v = 1; break;
        default:                          *v = 0; break;
    }
    printf_log(LOG_NONE, "RIMDROID SDL_GL_GetAttribute(%u) => %d\n", attr, *v);
    return 0;
}

#undef HAS_MY

#define ALTMY my2_

// No real ARM64 libSDL2 needed — our my2_SDL_GL_* and my2_SDL_DYNAPI_entry
// intercepts work without an underlying native SDL2.  getMy() will set all
// my->* to NULL (dlsym(NULL,...) returns NULL for unknown symbols), and
// my2_SDL_DYNAPI_entry guards against that explicitly.
#define OPTIONAL_LIB

#define CUSTOM_INIT \
    box64->sdl2lib = lib;                   \
    getMy(lib);                             \
    box64->sdl2allocrw = my->SDL_AllocRW;   \
    box64->sdl2freerw  = my->SDL_FreeRW;

#define NEEDED_LIBS "libdl.so.2", "libm.so.6", "librt.so.1", "libpthread.so.0"

#define CUSTOM_FINI \
    my->SDL_Quit();                                             \
    if(my_glhandle) my_dlclose(thread_get_emu(), my_glhandle);  \
    my_glhandle = NULL;                                         \
    freeMy();                                                   \
    my_context->sdl2lib = NULL;                                 \
    my_context->sdl2allocrw = NULL;                             \
    my_context->sdl2freerw = NULL;

#include "wrappedlib_init.h"
