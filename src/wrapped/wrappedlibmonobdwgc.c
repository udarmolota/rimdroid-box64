/*
 * RimDroid experiment: x86_64 UnityPlayer running on native ARM64 Unity Mono.
 *
 * UnityPlayer opens RimWorldLinux_Data/MonoBleedingEdge/x86_64/libmonobdwgc-2.0.so by full path. When
 * RIMDROID_NATIVE_MONO_PATH names an ARM64 build of the same Unity Mono (Unity 2022.3.35f1, fork commit
 * c11bc9adba), library.c routes that load here and this wrapper exposes the native runtime instead of
 * emulating the x86 one. Without the variable nothing changes: the emulated x86 Mono is loaded as before.
 *
 * The bridge mechanics were proven one by one with tools/mono-arm64/box64-probe (docs/MONO_ARM64_SPIKE.md):
 *   - reverse internal calls: generated typed thunks, one per Unity internal call (wrappedlibmonobdwgc_icalls.h)
 *   - managed exceptions raised by x86 code: pending exception + early return, never a native unwind (P4)
 *   - x86 guest state as GC roots: emulated registers and guest stacks pushed into Boehm (P3)
 *   - SIGSEGV shared between Mono and Box64: signal chaining forced on, crash chaining off (P5)
 */
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wrappedlibs.h"

#include "debug.h"
#include "wrapper.h"
#include "bridge.h"
#include "librarian/library_private.h"
#include "x64emu.h"
#include "callback.h"
#include "box64context.h"
#include "emu/x64emu_private.h"
#include "regs.h"

const char* libmonobdwgcName = "libmonobdwgc-2.0.so";
#define LIBNAME libmonobdwgc

#include "generated/wrappedlibmonobdwgctypes.h"
#include "wrappercallback.h"

static void* rd_native_sym(const char* name)
{
    return (my_lib && my_lib->w.lib) ? dlsym(my_lib->w.lib, name) : NULL;
}

/*
 * GC roots held by x86 guest threads (P3, extended to every thread).
 *
 * Boehm scans native stacks and registers of the threads it stops, but an x86 guest keeps object
 * references in Box64 state: the emulated registers and a separately mapped guest stack. Every emulator
 * that runs Mono-facing guest code is registered here (the thread that initializes Mono, threads that call
 * mono_thread_attach, and any thread entering a reverse internal call), and a GC_push_other_roots hook
 * chained on top of Mono's own pushes their state on every collection.
 *
 * The whole guest stack is pushed, not only the part above RSP: while a thread runs translated code its
 * emulator RSP is only synchronized at exits from the dynarec, so it can be stale and too high. Scanning
 * dead frames only costs time and conservative retention.
 *
 * The hook runs with the world stopped and the GC allocation lock held: it must not lock, allocate or
 * print. The registry is only changed under the same lock, and an emulator is removed before Box64 frees
 * it (rd_emu_destroy_hook in threads.c).
 */
#define RD_GC_MAX_GUESTS 512

typedef void (*rd_gc_push_other_roots_t)(void);
static rd_gc_push_other_roots_t rd_gc_prev_push_other_roots;
static void (*rd_gc_push_all)(void*, void*);
static void* (*rd_gc_call_with_alloc_lock)(void* (*)(void*), void*);
static x64emu_t* rd_gc_guests[RD_GC_MAX_GUESTS];
static int rd_gc_guest_count;
static int rd_gc_ready;
static volatile unsigned long rd_gc_collections;
static volatile unsigned long rd_gc_bytes_last;
static __thread x64emu_t* rd_gc_registered_emu;

extern void (*rd_emu_destroy_hook)(x64emu_t* emu);

static void rd_gc_push_guest_roots(void)
{
    if (rd_gc_prev_push_other_roots)
        rd_gc_prev_push_other_roots();
    unsigned long bytes = 0;
    for (int i = 0; i < rd_gc_guest_count; ++i) {
        x64emu_t* emu = rd_gc_guests[i];
        if (!emu)
            continue;
        rd_gc_push_all(&emu->regs[0], &emu->regs[16]);
        rd_gc_push_all(&emu->xmm[0], &emu->xmm[16]);
        if (emu->init_stack && emu->size_stack) {
            rd_gc_push_all(emu->init_stack, (void*)((uintptr_t)emu->init_stack + emu->size_stack));
            bytes += emu->size_stack;
        }
    }
    rd_gc_bytes_last = bytes;
    rd_gc_collections++;
}

static void* rd_gc_add_locked(void* arg)
{
    x64emu_t* emu = (x64emu_t*)arg;
    int free_slot = -1;
    for (int i = 0; i < rd_gc_guest_count; ++i) {
        if (rd_gc_guests[i] == emu)
            return (void*)1;
        if (!rd_gc_guests[i] && free_slot < 0)
            free_slot = i;
    }
    if (free_slot < 0) {
        if (rd_gc_guest_count == RD_GC_MAX_GUESTS)
            return NULL;
        free_slot = rd_gc_guest_count++;
    }
    rd_gc_guests[free_slot] = emu;
    return (void*)1;
}

static void* rd_gc_remove_locked(void* arg)
{
    x64emu_t* emu = (x64emu_t*)arg;
    for (int i = 0; i < rd_gc_guest_count; ++i)
        if (rd_gc_guests[i] == emu)
            rd_gc_guests[i] = NULL;
    return NULL;
}

static void rd_gc_register_emu(x64emu_t* emu)
{
    if (!rd_gc_ready || !emu)
        return;
    if (!rd_gc_call_with_alloc_lock(rd_gc_add_locked, emu)) {
        printf_log(LOG_NONE, "[RD-MONO] GC guest registry full (%d threads): emu %p is NOT scanned\n",
            RD_GC_MAX_GUESTS, emu);
        return;
    }
    if (rd_gc_registered_emu != emu) {
        rd_gc_registered_emu = emu;
        printf_log(LOG_INFO, "[RD-MONO] GC guest roots: registered emu %p stack=[%p,+0x%x)\n",
            emu, emu->init_stack, emu->size_stack);
    }
}

static void rd_gc_emu_destroyed(x64emu_t* emu)
{
    if (rd_gc_ready)
        rd_gc_call_with_alloc_lock(rd_gc_remove_locked, emu);
    // Box64 destroys an emulator on its own thread; a new one may reuse the address.
    if (rd_gc_registered_emu == emu)
        rd_gc_registered_emu = NULL;
}

static void rd_gc_install(x64emu_t* main_emu)
{
    const char* off = getenv("RIMDROID_MONO_NO_GUEST_ROOTS");
    if (off && off[0] == '1') {
        printf_log(LOG_NONE, "[RD-MONO] GC guest roots DISABLED (RIMDROID_MONO_NO_GUEST_ROOTS=1)\n");
        return;
    }
    rd_gc_push_other_roots_t (*get_push)(void) = rd_native_sym("GC_get_push_other_roots");
    void (*set_push)(rd_gc_push_other_roots_t) = rd_native_sym("GC_set_push_other_roots");
    rd_gc_push_all = rd_native_sym("GC_push_all");
    rd_gc_call_with_alloc_lock = rd_native_sym("GC_call_with_alloc_lock");
    if (!get_push || !set_push || !rd_gc_push_all || !rd_gc_call_with_alloc_lock) {
        printf_log(LOG_NONE, "[RD-MONO] GC guest roots NOT installed: missing GC hooks (get=%p set=%p push=%p lock=%p)\n",
            get_push, set_push, rd_gc_push_all, rd_gc_call_with_alloc_lock);
        return;
    }
    rd_gc_prev_push_other_roots = get_push();
    set_push(rd_gc_push_guest_roots);
    rd_emu_destroy_hook = rd_gc_emu_destroyed;
    rd_gc_ready = 1;
    rd_gc_register_emu(main_emu);
    printf_log(LOG_NONE, "[RD-MONO] GC guest roots installed (prev hook %p)\n", rd_gc_prev_push_other_roots);
}

/*
 * Managed exceptions raised by x86 code (P4).
 *
 * A native mono_raise_exception unwinds straight to the managed catch block through Mono's LMF chain and
 * drops everything in between without running it: the reverse thunk, RunFunctionFmt, DynaCall and the
 * dynarec frames, leaving the guest registers, guest stack and emulator state inside dead x86 frames.
 *
 * So inside a reverse internal call an exception never unwinds natively. It becomes Mono's pending
 * exception and the emulator returns early: emu->quit makes the dynarec exit right after the bridge call,
 * DynaCall restores RBX/RDI/RSI/RBP/RSP/RIP, and the thunk restores R12-R15, which DynaCall does not save.
 * The internal call returns normally and Mono's wrapper for foreign internal calls throws the pending
 * exception at its checkpoint, inside managed code. Outside any internal call there is no managed frame
 * to reach and the raise stays native, as it would be on x86 Mono.
 */
static __thread int rd_icall_depth;
static __thread int rd_icall_raised;
static int (*rd_set_pending_exception)(void*, int);

#define RD_MONO_ICALL_BRIDGE_DEFINED
typedef struct rd_icall_frame_s {
    x64emu_t* emu;
    uint64_t r12, r13, r14, r15;
    int outer_raised;
} rd_icall_frame_t;

static inline void rd_icall_enter(rd_icall_frame_t* frame)
{
    x64emu_t* emu = thread_get_emu();
    frame->emu = emu;
    frame->r12 = emu->regs[_R12].q[0];
    frame->r13 = emu->regs[_R13].q[0];
    frame->r14 = emu->regs[_R14].q[0];
    frame->r15 = emu->regs[_R15].q[0];
    frame->outer_raised = rd_icall_raised;
    rd_icall_raised = 0;
    rd_icall_depth++;
    if (rd_gc_registered_emu != emu)
        rd_gc_register_emu(emu);
}

static inline int rd_icall_leave(rd_icall_frame_t* frame)
{
    rd_icall_depth--;
    int raised = rd_icall_raised;
    if (raised) {
        x64emu_t* emu = frame->emu;
        emu->regs[_R12].q[0] = frame->r12;
        emu->regs[_R13].q[0] = frame->r13;
        emu->regs[_R14].q[0] = frame->r14;
        emu->regs[_R15].q[0] = frame->r15;
    }
    rd_icall_raised = frame->outer_raised;
    return raised;
}

static inline float rd_icall_float_result(rd_icall_frame_t* frame)
{
    return frame->emu->xmm[0].f[0];
}

/* Returns 1 when the exception was turned into a pending exception for the innermost internal call. */
static int rd_defer_exception(x64emu_t* emu, void* exception)
{
    if (!exception || rd_icall_depth <= 0 || !rd_set_pending_exception)
        return 0;
    rd_set_pending_exception(exception, 1);
    rd_icall_raised = 1;
    emu->quit = 1;
    return 1;
}

#include "wrappedlibmonobdwgc_icalls.h"

static unsigned long rd_icall_bound;
static unsigned long rd_icall_missing;

EXPORT void my_mono_add_internal_call(x64emu_t* emu, const char* name, void* method)
{
    (void)emu;
    void* host = NULL;
    if (method) {
        host = GetNativeFnc((uintptr_t)method);
        if (!host && name)
            host = rd_mono_icall_bind(name, (uintptr_t)method);
    }
    if (!host) {
        // Registering the x86 address would make ARM64 code jump into x86 code. Leaving the call
        // unregistered makes Mono throw MissingMethodException if managed code ever calls it.
        rd_icall_missing++;
        printf_log(LOG_NONE, "[RD-MONO] no thunk for internal call %s (guest %p), not registered\n",
            name ? name : "(null)", method);
        return;
    }
    if ((++rd_icall_bound % 1000) == 0)
        printf_log(LOG_INFO, "[RD-MONO] internal calls bound: %lu (missing %lu)\n", rd_icall_bound, rd_icall_missing);
    my->mono_add_internal_call((void*)name, host);
}

EXPORT void my_mono_raise_exception(x64emu_t* emu, void* exception)
{
    if (rd_defer_exception(emu, exception))
        return;
    my->mono_raise_exception(exception);
}

/*
 * With a NULL exception slot these APIs raise a managed exception natively (the same unwind as
 * mono_raise_exception). Always pass a slot and route the exception through the bridge.
 */
static void* rd_finish_invoke(x64emu_t* emu, void* ret, void* exception)
{
    if (!exception)
        return ret;
    if (!rd_defer_exception(emu, exception))
        my->mono_raise_exception(exception);
    return NULL;
}

EXPORT void* my_mono_runtime_invoke(x64emu_t* emu, void* method, void* obj, void* params, void** exc)
{
    if (exc)
        return my->mono_runtime_invoke(method, obj, params, exc);
    void* exception = NULL;
    void* ret = my->mono_runtime_invoke(method, obj, params, &exception);
    return rd_finish_invoke(emu, ret, exception);
}

EXPORT void* my_mono_runtime_invoke_array(x64emu_t* emu, void* method, void* obj, void* params, void** exc)
{
    if (exc)
        return my->mono_runtime_invoke_array(method, obj, params, exc);
    void* exception = NULL;
    void* ret = my->mono_runtime_invoke_array(method, obj, params, &exception);
    return rd_finish_invoke(emu, ret, exception);
}

EXPORT void* my_mono_runtime_delegate_invoke(x64emu_t* emu, void* delegate, void* params, void** exc)
{
    if (exc)
        return my->mono_runtime_delegate_invoke(delegate, params, exc);
    void* exception = NULL;
    void* ret = my->mono_runtime_delegate_invoke(delegate, params, &exception);
    return rd_finish_invoke(emu, ret, exception);
}

EXPORT int my_mono_runtime_exec_main(x64emu_t* emu, void* method, void* args, void** exc)
{
    if (exc)
        return my->mono_runtime_exec_main(method, args, exc);
    void* exception = NULL;
    int ret = my->mono_runtime_exec_main(method, args, &exception);
    if (exception) {
        rd_finish_invoke(emu, NULL, exception);
        return 1;
    }
    return ret;
}

/*
 * Signal ownership (P5). Mono installs its SIGSEGV/SIGBUS/SIGILL/SIGABRT... handlers during JIT init and
 * replaces Box64's. It keeps the previous handler only with signal chaining on before init, and hands a
 * fault outside JIT code to it only with crash chaining off. Unity sets both itself, so its arguments are
 * overridden. RIMDROID_MONO_NO_SIGNAL_CHAINING=1 passes Unity's values through.
 */
static int rd_signal_override(void)
{
    const char* off = getenv("RIMDROID_MONO_NO_SIGNAL_CHAINING");
    return !(off && off[0] == '1');
}

EXPORT void my_mono_set_signal_chaining(x64emu_t* emu, int chain)
{
    (void)emu;
    int value = rd_signal_override() ? 1 : chain;
    printf_log(LOG_NONE, "[RD-MONO] mono_set_signal_chaining(%d) -> %d\n", chain, value);
    my->mono_set_signal_chaining(value);
}

EXPORT void my_mono_set_crash_chaining(x64emu_t* emu, int chain)
{
    (void)emu;
    int value = rd_signal_override() ? 0 : chain;
    printf_log(LOG_NONE, "[RD-MONO] mono_set_crash_chaining(%d) -> %d\n", chain, value);
    my->mono_set_crash_chaining(value);
}

EXPORT void* my_mono_jit_init_version(x64emu_t* emu, const char* domain_name, const char* runtime_version)
{
    if (rd_signal_override()) {
        my->mono_set_signal_chaining(1);
        my->mono_set_crash_chaining(0);
    }
    printf_log(LOG_NONE, "[RD-MONO] mono_jit_init_version(%s, %s) on native ARM64 Mono\n",
        domain_name ? domain_name : "(null)", runtime_version ? runtime_version : "(null)");
    void* domain = my->mono_jit_init_version((void*)domain_name, (void*)runtime_version);
    if (domain) {
        rd_set_pending_exception = rd_native_sym("mono_runtime_set_pending_exception");
        if (!rd_set_pending_exception)
            printf_log(LOG_NONE, "[RD-MONO] mono_runtime_set_pending_exception missing: exceptions from x86 internal calls will corrupt the guest\n");
        rd_gc_install(emu);
    }
    printf_log(LOG_NONE, "[RD-MONO] mono_jit_init_version -> domain %p\n", domain);
    return domain;
}

EXPORT void my_mono_unity_jit_cleanup(x64emu_t* emu, void* domain)
{
    (void)emu;
    printf_log(LOG_NONE, "[RD-MONO] jit cleanup: %lu collections scanned guest roots (last %lu stack bytes), %lu internal calls bound, %lu missing\n",
        rd_gc_collections, rd_gc_bytes_last, rd_icall_bound, rd_icall_missing);
    my->mono_unity_jit_cleanup(domain);
}

EXPORT void* my_mono_thread_attach(x64emu_t* emu, void* domain)
{
    void* thread = my->mono_thread_attach(domain);
    rd_gc_register_emu(emu);
    return thread;
}

/*
 * Callbacks from native Mono into x86 code.
 *
 * Synchronous iterators (foreach, stack walks) keep the guest function in a thread-local that is saved and
 * restored around the call, so nesting and concurrent threads stay correct. Callbacks that Mono stores and
 * calls later use a single static slot each: Unity installs each of them once.
 */
#define RD_GUEST_OR_NATIVE(fct) (GetNativeFnc((uintptr_t)(fct)))

static __thread uintptr_t rd_cb_gfunc;
static void rd_cb_gfunc_thunk(void* data, void* user_data)
{
    RunFunctionFmt(rd_cb_gfunc, "pp", data, user_data);
}

static __thread uintptr_t rd_cb_stackwalk;
static int rd_cb_stackwalk_thunk(void* method, int32_t native_offset, int32_t il_offset, int managed, void* data)
{
    return (int)RunFunctionFmt(rd_cb_stackwalk, "piiip", method, native_offset, il_offset, managed, data);
}

#define RD_WITH_GUEST(slot, fct, thunk, native_call, bridged_call) \
    do { \
        void* rd_native = (fct) ? RD_GUEST_OR_NATIVE(fct) : NULL; \
        if (!(fct) || rd_native) { native_call; break; } \
        uintptr_t rd_saved = slot; \
        slot = (uintptr_t)(fct); \
        bridged_call; \
        slot = rd_saved; \
    } while (0)

EXPORT void my_mono_stack_walk(x64emu_t* emu, void* func, void* user_data)
{
    (void)emu;
    RD_WITH_GUEST(rd_cb_stackwalk, func, rd_cb_stackwalk_thunk,
        my->mono_stack_walk(rd_native ? rd_native : func, user_data),
        my->mono_stack_walk(rd_cb_stackwalk_thunk, user_data));
}

EXPORT void my_mono_stack_walk_no_il(x64emu_t* emu, void* func, void* user_data)
{
    (void)emu;
    RD_WITH_GUEST(rd_cb_stackwalk, func, rd_cb_stackwalk_thunk,
        my->mono_stack_walk_no_il(rd_native ? rd_native : func, user_data),
        my->mono_stack_walk_no_il(rd_cb_stackwalk_thunk, user_data));
}

#define RD_GFUNC_2(NAME) \
EXPORT void my_##NAME(x64emu_t* emu, void* func, void* user_data) \
{ \
    (void)emu; \
    RD_WITH_GUEST(rd_cb_gfunc, func, rd_cb_gfunc_thunk, \
        my->NAME(rd_native ? rd_native : func, user_data), \
        my->NAME(rd_cb_gfunc_thunk, user_data)); \
}

#define RD_GFUNC_3(NAME) \
EXPORT void my_##NAME(x64emu_t* emu, void* first, void* func, void* user_data) \
{ \
    (void)emu; \
    RD_WITH_GUEST(rd_cb_gfunc, func, rd_cb_gfunc_thunk, \
        my->NAME(first, rd_native ? rd_native : func, user_data), \
        my->NAME(first, rd_cb_gfunc_thunk, user_data)); \
}

// GFunc, MonoFunc and ClassReportFunc all take (pointer, user_data).
RD_GFUNC_2(mono_assembly_foreach)
RD_GFUNC_2(mono_unity_gc_handles_foreach_get_target)
RD_GFUNC_2(mono_unity_gc_heap_foreach)
RD_GFUNC_2(mono_unity_image_set_mempool_chunk_foreach)
RD_GFUNC_2(mono_unity_root_domain_mempool_chunk_foreach)
RD_GFUNC_2(mono_unity_class_for_each)
RD_GFUNC_3(mono_unity_assembly_mempool_chunk_foreach)
RD_GFUNC_3(mono_unity_domain_mempool_chunk_foreach)
RD_GFUNC_3(mono_unity_type_get_name_full_chunked)

#undef RD_GFUNC_2
#undef RD_GFUNC_3

// MonoLogCallback: (log_domain, log_level, message, fatal, user_data)
static uintptr_t rd_cb_log;
static void rd_cb_log_thunk(const char* domain, const char* level, const char* message, int fatal, void* user_data)
{
    RunFunctionFmt(rd_cb_log, "pppip", domain, level, message, fatal, user_data);
}

EXPORT void my_mono_trace_set_log_handler(x64emu_t* emu, void* callback, void* user_data)
{
    (void)emu;
    void* native = callback ? GetNativeFnc((uintptr_t)callback) : NULL;
    if (callback && !native) {
        rd_cb_log = (uintptr_t)callback;
        native = rd_cb_log_thunk;
    }
    my->mono_trace_set_log_handler(native, user_data);
}

// UnityFindPluginCallback: const char* (const char*)
static uintptr_t rd_cb_find_plugin;
static const char* rd_cb_find_plugin_thunk(const char* name)
{
    return (const char*)RunFunctionFmt(rd_cb_find_plugin, "p", name);
}

EXPORT void my_mono_set_find_plugin_callback(x64emu_t* emu, void* find)
{
    (void)emu;
    void* native = find ? GetNativeFnc((uintptr_t)find) : NULL;
    if (find && !native) {
        rd_cb_find_plugin = (uintptr_t)find;
        native = rd_cb_find_plugin_thunk;
    }
    my->mono_set_find_plugin_callback(native);
}

/*
 * vprintf_func receives a native va_list that x86 code cannot read. Mono's own output is formatted here
 * and written to the Box64 log instead of Unity's handler.
 */
static int rd_mono_vprintf(const char* message, va_list args)
{
    char buffer[4096];
    int len = vsnprintf(buffer, sizeof(buffer), message, args);
    size_t end = strlen(buffer);
    if (end && buffer[end - 1] == '\n')
        buffer[end - 1] = 0;
    printf_log(LOG_NONE, "[RD-MONO] mono: %s\n", buffer);
    return len;
}

EXPORT void my_mono_unity_set_vprintf_func(x64emu_t* emu, void* func)
{
    (void)emu;
    printf_log(LOG_NONE, "[RD-MONO] mono_unity_set_vprintf_func(%p): Mono output goes to the Box64 log\n", func);
    my->mono_unity_set_vprintf_func(func ? rd_mono_vprintf : NULL);
}

/*
 * The dl fallback resolves P/Invoke libraries and symbols through x86 callbacks, which would hand native
 * Mono x86 handles and x86 code addresses. Not registered: native Mono loads native libraries itself.
 */
EXPORT void* my_mono_dl_fallback_register(x64emu_t* emu, void* load_func, void* symbol_func, void* close_func, void* user_data)
{
    (void)emu;
    printf_log(LOG_NONE, "[RD-MONO] mono_dl_fallback_register(load=%p symbol=%p close=%p data=%p) ignored\n",
        load_func, symbol_func, close_func, user_data);
    return NULL;
}

EXPORT void my_mono_dl_fallback_unregister(x64emu_t* emu, void* handler)
{
    (void)emu;
    if (handler)
        my->mono_dl_fallback_unregister(handler);
}

// Legacy profiler: only the shutdown callback is bridged, it runs once on the shutting-down thread.
static uintptr_t rd_cb_profiler_shutdown;
static void rd_cb_profiler_shutdown_thunk(void* prof)
{
    RunFunctionFmt(rd_cb_profiler_shutdown, "p", prof);
}

EXPORT void my_mono_profiler_install(x64emu_t* emu, void* prof, void* callback)
{
    (void)emu;
    void* native = callback ? GetNativeFnc((uintptr_t)callback) : NULL;
    if (callback && !native) {
        rd_cb_profiler_shutdown = (uintptr_t)callback;
        native = rd_cb_profiler_shutdown_thunk;
    }
    my->mono_profiler_install(prof, native);
}

/*
 * GC event callbacks can run while the world is stopped. Running x86 code then can block forever on a
 * Box64 lock owned by a stopped thread, so they are not installed.
 */
EXPORT void my_mono_profiler_install_gc(x64emu_t* emu, void* callback, void* heap_resize_callback)
{
    (void)emu;
    printf_log(LOG_NONE, "[RD-MONO] mono_profiler_install_gc(%p, %p) ignored\n", callback, heap_resize_callback);
}

EXPORT int my_mono_profiler_get_all_coverage_data(x64emu_t* emu, void* handle, void* callback)
{
    (void)emu;
    printf_log(LOG_NONE, "[RD-MONO] mono_profiler_get_all_coverage_data(%p, %p) ignored\n", handle, callback);
    return 0;
}

EXPORT int my_mono_profiler_get_coverage_data(x64emu_t* emu, void* handle, void* method, void* callback)
{
    (void)emu;
    printf_log(LOG_NONE, "[RD-MONO] mono_profiler_get_coverage_data(%p, %p, %p) ignored\n", handle, method, callback);
    return 0;
}

EXPORT void my_mono_profiler_set_coverage_filter_callback(x64emu_t* emu, void* handle, void* callback)
{
    (void)emu;
    printf_log(LOG_NONE, "[RD-MONO] mono_profiler_set_coverage_filter_callback(%p, %p) ignored\n", handle, callback);
}

/*
 * Liveness (Unity's memory profiler): the callbacks are stored in the state and called synchronously by
 * the calculation functions. One state is used at a time.
 */
static uintptr_t rd_cb_liveness_register;
static uintptr_t rd_cb_liveness_realloc;

static void rd_cb_liveness_register_thunk(void** array, int size, void* user_data)
{
    RunFunctionFmt(rd_cb_liveness_register, "pip", array, size, user_data);
}

static void* rd_cb_liveness_realloc_thunk(void* ptr, int size, void* user_data)
{
    return (void*)RunFunctionFmt(rd_cb_liveness_realloc, "pip", ptr, size, user_data);
}

EXPORT void* my_mono_unity_liveness_allocate_struct(x64emu_t* emu, void* filter, uint32_t max_count,
    void* callback, void* user_data, void* reallocate)
{
    (void)emu;
    void* native_callback = callback ? GetNativeFnc((uintptr_t)callback) : NULL;
    if (callback && !native_callback) {
        rd_cb_liveness_register = (uintptr_t)callback;
        native_callback = rd_cb_liveness_register_thunk;
    }
    void* native_realloc = reallocate ? GetNativeFnc((uintptr_t)reallocate) : NULL;
    if (reallocate && !native_realloc) {
        rd_cb_liveness_realloc = (uintptr_t)reallocate;
        native_realloc = rd_cb_liveness_realloc_thunk;
    }
    return my->mono_unity_liveness_allocate_struct(filter, max_count, native_callback, user_data, native_realloc);
}

// unitytls is a table of x86 function pointers used by Mono's TLS provider; not needed to reach the menu.
EXPORT void my_mono_unity_install_unitytls_interface(x64emu_t* emu, void* callbacks)
{
    (void)emu;
    printf_log(LOG_NONE, "[RD-MONO] mono_unity_install_unitytls_interface(%p) ignored\n", callbacks);
}

#ifndef STATICBUILD
#define PRE_INIT \
    if (1) { \
        const char* path = getenv("RIMDROID_NATIVE_MONO_PATH"); \
        if (!path || !*path) return -1; \
        lib->w.lib = dlopen(path, RTLD_NOW | RTLD_GLOBAL); \
        if (!lib->w.lib) { \
            printf_log(LOG_NONE, "[RD-MONO] cannot load native Mono %s: %s\n", path, dlerror()); \
            return -1; \
        } \
        printf_log(LOG_NONE, "[RD-MONO] using native ARM64 Mono %s\n", path); \
    } else
#endif

#include "wrappedlib_init.h"
