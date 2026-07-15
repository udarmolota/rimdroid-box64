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
    // RimDroid: the RimWorld 1.6 save crash is a SEGV inside this wrapper — a garbage
    // descriptor (or module index) arrives from the guest. Validate before dereferencing
    // and log the guest caller: a Mono-JIT caller address = the known dynarec corruption
    // family; a libc/loader caller = a box64 TLS infra bug (the save runs on RimWorld's
    // freshly spawned Saver thread). Capped, always-on (LOG_NONE).
    int p_readable = ((uintptr_t)p >= 0x10000) && !((uintptr_t)p & 0x7);
    if (!p_readable || t->i >= (unsigned long)my_context->elfsize) {
        static int rd_tlsbad_n = 0;
        if (rd_tlsbad_n < 16) {
            rd_tlsbad_n++;
            printf_log(LOG_NONE, "RIMDROID TLSBAD p=%p i=0x%lx o=0x%lx elfsize=%d rip=%p(%s)\n",
                p, p_readable ? t->i : 0, p_readable ? t->o : 0,
                my_context->elfsize, (void*)R_RIP, getAddrFunctionName(R_RIP));
            fflush(NULL);
        }
        // For a decodable-but-out-of-range index keep the process alive on a scratch slot
        // so the save path can be observed further (the caller was doomed either way);
        // an unreadable p still falls through to the original crash.
        if (p_readable) {
            static __thread char rd_tls_scratch[512];
            memset(rd_tls_scratch, 0, sizeof(rd_tls_scratch));
            return rd_tls_scratch;
        }
    }
    tlsdatasize_t* ptr = emu->tlsdata;
    if (!ptr) {
        refreshTLSData(emu);
        ptr = emu->tlsdata;
    }
    return ptr->data+GetTLSBase(my_context->elfs[t->i])+t->o;
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

