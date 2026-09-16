#include <dlfcn.h>
#include <stdlib.h>

#include "wrappedlibs.h"

#include "debug.h"
#include "wrapper.h"
#include "bridge.h"
#include "librarian/library_private.h"
#include "x64emu.h"
#include "callback.h"
#include "box64context.h"

const char* rdmonoprobeName = "librdmonoprobe.so";
#define LIBNAME rdmonoprobe

#include "generated/wrappedrdmonoprobetypes.h"
#include "wrappercallback.h"

#define REVERSE_ICALL_SLOTS() \
    GO(0) \
    GO(1) \
    GO(2) \
    GO(3)

#define GO(A) \
static uintptr_t reverse_icall_iFii_fct_##A = 0; \
static int reverse_icall_iFii_##A(int left, int right) \
{ \
    return (int)RunFunctionFmt(reverse_icall_iFii_fct_##A, "ii", left, right); \
}
REVERSE_ICALL_SLOTS()
#undef GO

static void* find_reverse_icall_iFii(void* fct)
{
    if (!fct) return NULL;
    if (GetNativeFnc((uintptr_t)fct)) return GetNativeFnc((uintptr_t)fct);
#define GO(A) if (reverse_icall_iFii_fct_##A == (uintptr_t)fct) return reverse_icall_iFii_##A;
    REVERSE_ICALL_SLOTS()
#undef GO
#define GO(A) if (!reverse_icall_iFii_fct_##A) { reverse_icall_iFii_fct_##A = (uintptr_t)fct; return reverse_icall_iFii_##A; }
    REVERSE_ICALL_SLOTS()
#undef GO
    printf_log(LOG_NONE, "Warning, no free P2 reverse icall slot\n");
    return NULL;
}

#undef REVERSE_ICALL_SLOTS

EXPORT void my_mono_add_internal_call(x64emu_t* emu, const char* name, void* method)
{
    (void)emu;
    my->mono_add_internal_call((void*)name, find_reverse_icall_iFii(method));
}

/*
 * This wrapper exists solely for the ARM64 Mono feasibility probe.  Loading the
 * real Unity guest library by its pathname would make Box64 choose emulation,
 * so the probe asks for this synthetic name and explicitly supplies the native
 * library through RIMDROID_NATIVE_MONO_PATH.
 */
#ifndef STATICBUILD
#define PRE_INIT \
    if (1) { \
        const char* path = getenv("RIMDROID_NATIVE_MONO_PATH"); \
        if (!path || !*path) return -1; \
        lib->w.lib = dlopen(path, RTLD_NOW | RTLD_GLOBAL); \
        if (!lib->w.lib) return -1; \
    } else
#endif

#include "wrappedlib_init.h"
