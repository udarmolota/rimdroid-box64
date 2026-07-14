#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#include <dlfcn.h>

#include "wrappedlibs.h"

#include "debug.h"
#include "wrapper.h"
#include "bridge.h"
#include "librarian/library_private.h"
#include "x64emu.h"
#include "emu/x64emu_private.h"
#include "elfloader.h"
#include "box64context.h"
#include "x64tls.h"

typedef struct my_tls_s {
    unsigned long int   i;
    unsigned long int   o;
} my_tls_t;

EXPORT void* my___tls_get_addr(x64emu_t* emu, void* p)
{
    my_tls_t *t = (my_tls_t*)p;
    // RimDroid: the RimWorld 1.6 save crash is a SEGV on the TLS path — a garbage descriptor
    // arrives from the guest (observed p=0x71`00000063: valid low 32 bits, junk high 32 bits =
    // the box64 "wrong register value" / un-zero-extended-32bit family, same as the qsort/IMT bug
    // but a new site). Validate before dereferencing; on ANY bad input LOG the guest CALLER (so we
    // can pin the miscompiled call site) and the hi/lo halves of p, then return a zeroed scratch
    // slot to keep the process ALIVE (was: survive only readable-but-out-of-range; an unreadable p
    // used to fall through and crash — that's the SEGV we just caught). Survival lets the save run
    // on and surfaces more callers. Capped, always-on (LOG_NONE).
    int p_aligned = !((uintptr_t)p & 0x7);
    int p_sane = ((uintptr_t)p >= 0x10000) && ((uintptr_t)p < 0x800000000000ULL) && p_aligned;
    unsigned long idx = p_sane ? t->i : ~0UL;
    if (!p_sane || idx >= (unsigned long)my_context->elfsize) {
        static int rd_tlsbad_n = 0;
        if (rd_tlsbad_n < 64) {
            rd_tlsbad_n++;
            uintptr_t sp = R_RSP;
            int sp_ok = (sp >= 0x10000) && !(sp & 0x7);
            uintptr_t caller = sp_ok ? *(uintptr_t*)sp : 0;   // guest return address = the call site
            printf_log(LOG_NONE, "RIMDROID TLSBAD p=%p (hi32=0x%08x lo32=0x%08x) i=0x%lx o=0x%lx "
                "elfsize=%d rip=%p(%s) caller=%p(%s)\n",
                p, (uint32_t)((uintptr_t)p>>32), (uint32_t)(uintptr_t)p,
                p_sane ? t->i : 0, p_sane ? t->o : 0, my_context->elfsize,
                (void*)R_RIP, getAddrFunctionName(R_RIP),
                (void*)caller, caller ? getAddrFunctionName(caller) : "?");
            fflush(NULL);
        }
        static __thread char rd_tls_scratch[512];
        memset(rd_tls_scratch, 0, sizeof(rd_tls_scratch));
        return rd_tls_scratch;
    }
    // Here p is valid and idx is in range, yet the RimWorld-1.6 save still SEGVs computing the slot
    // (fault deep in this expression, no TLSBAD). Two candidates: a STALE per-thread TLS block
    // (emu->tlsdata behind my_context after a module loaded on the save's worker thread) or a bad
    // elf slot (my_context->elfs[idx]). The stock code only refreshed when tlsdata==NULL; refresh
    // whenever it's stale (tlssize/n_elfs behind the context), then guard every dereference and, on
    // any anomaly, log exactly which sub-value is bad + return a scratch slot so the save survives.
    tlsdatasize_t* ptr = emu->tlsdata;
    if (!ptr || ptr->tlssize != my_context->tlssize || (int)ptr->n_elfs != my_context->elfsize) {
        refreshTLSData(emu);
        ptr = emu->tlsdata;
    }
    // Acquire-load of the elfs array pointer — pairs with AddElfHeader's release-publish of the
    // grown copy, so we never index a torn/freed array even if a dlopen grows it mid-call.
    elfheader_t** rd_elfs = __atomic_load_n(&my_context->elfs, __ATOMIC_ACQUIRE);
    elfheader_t* h = rd_elfs[idx];
    int slot_bad = (!ptr) || ((uintptr_t)ptr < 0x10000)
                 || (!h)   || ((uintptr_t)h < 0x10000) || ((uintptr_t)h & 0x7);
    if (slot_bad) {
        static int rd_tlsslot_n = 0;
        if (rd_tlsslot_n < 64) {
            rd_tlsslot_n++;
            printf_log(LOG_NONE, "RIMDROID TLSSLOT idx=0x%lx o=0x%lx elfsize=%d ptr=%p n_elfs=%d "
                "tlssize=%d elf=%p rip=%p(%s)\n", idx, t->o, my_context->elfsize, (void*)ptr,
                ptr ? ptr->n_elfs : -1, ptr ? (int)ptr->tlssize : -1, (void*)h,
                (void*)R_RIP, getAddrFunctionName(R_RIP));
            fflush(NULL);
        }
        static __thread char rd_tls_scratch2[512];
        memset(rd_tls_scratch2, 0, sizeof(rd_tls_scratch2));
        return rd_tls_scratch2;
    }
    return ptr->data + GetTLSBase(h) + t->o;
}

EXPORT void* my___libc_stack_end;
void stSetup(box64context_t* context)
{
    my___libc_stack_end = context->stack + context->stacksz;
}

#ifdef STATICBUILD
#include <link.h>
extern void* __libc_enable_secure;
#ifndef PPC64LE
extern void* __stack_chk_guard;
#endif
//extern void* __pointer_chk_guard;
//extern void* _rtld_global;
//extern void* _rtld_global_ro;
#endif

// don't try to load the actual ld-linux (because name is variable), just use box64 itself, as it's linked to ld-linux
const char* ldlinuxName = "ld-linux.so.2";
#define LIBNAME ldlinux

#ifndef STATICBUILD
#define PRE_INIT\
    if(1)                                                           \
        lib->w.lib = dlopen(NULL, RTLD_LAZY | RTLD_GLOBAL);    \
    else
#endif

#define CUSTOM_INIT         \
    stSetup(box64);         \

// define all standard library functions
#include "wrappedlib_init.h"

