#include <dlfcn.h>
#include <stdint.h>
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
    printf_log(LOG_NONE, "RIMDROID P2 reverse icall slot=%d left=0x%x right=0x%x\n", A, left, right); \
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

static uintptr_t reverse_icall_IFII_fct;
static int64_t reverse_icall_IFII(int64_t left, int64_t right)
{
    return (int64_t)RunFunctionFmt(reverse_icall_IFII_fct, "II", left, right);
}

static uintptr_t reverse_icall_pFp_fct;
static void* reverse_icall_pFp(void* value)
{
    return (void*)RunFunctionFmt(reverse_icall_pFp_fct, "p", value);
}

static uintptr_t reverse_icall_dFdd_fct;
static double reverse_icall_dFdd(double left, double right)
{
    return RunFunctionFmtD(reverse_icall_dFdd_fct, "dd", left, right);
}

static uintptr_t reverse_icall_dFff_fct;
static double reverse_icall_dFff(float left, float right)
{
    return RunFunctionFmtD(reverse_icall_dFff_fct, "ff", left, right);
}

static uintptr_t reverse_icall_IFIIIIIIIII_fct;
static int64_t reverse_icall_IFIIIIIIIII(
    int64_t a, int64_t b, int64_t c, int64_t d, int64_t e,
    int64_t f, int64_t g, int64_t h, int64_t i)
{
    return (int64_t)RunFunctionFmt(
        reverse_icall_IFIIIIIIIII_fct, "IIIIIIIII",
        a, b, c, d, e, f, g, h, i);
}

#define NARROW_IDENTITY(NAME, CTYPE, FMT) \
    static uintptr_t reverse_icall_##NAME##_fct; \
    static CTYPE reverse_icall_##NAME(CTYPE value) \
    { \
        return (CTYPE)RunFunctionFmt(reverse_icall_##NAME##_fct, FMT, value); \
    }

NARROW_IDENTITY(cFc, int8_t, "c")
NARROW_IDENTITY(CFC, uint8_t, "C")
NARROW_IDENTITY(wFw, int16_t, "w")
NARROW_IDENTITY(WFW, uint16_t, "W")

#undef NARROW_IDENTITY

static void* select_reverse_icall(const char* name, void* method)
{
    void* native = GetNativeFnc((uintptr_t)method);
    if (native) return native;

    if (strstr(name, "::NativeAddLong")) {
        reverse_icall_IFII_fct = (uintptr_t)method;
        return reverse_icall_IFII;
    }
    if (strstr(name, "::NativePointerIdentity")) {
        reverse_icall_pFp_fct = (uintptr_t)method;
        return reverse_icall_pFp;
    }
    if (strstr(name, "::NativeAddDouble")) {
        reverse_icall_dFdd_fct = (uintptr_t)method;
        return reverse_icall_dFdd;
    }
    if (strstr(name, "::NativeAddFloatArgs")) {
        reverse_icall_dFff_fct = (uintptr_t)method;
        return reverse_icall_dFff;
    }
    if (strstr(name, "::NativeSumNine")) {
        reverse_icall_IFIIIIIIIII_fct = (uintptr_t)method;
        return reverse_icall_IFIIIIIIIII;
    }
    if (strstr(name, "::NativeSByteIdentity")) {
        reverse_icall_cFc_fct = (uintptr_t)method;
        return reverse_icall_cFc;
    }
    if (strstr(name, "::NativeByteIdentity")) {
        reverse_icall_CFC_fct = (uintptr_t)method;
        return reverse_icall_CFC;
    }
    if (strstr(name, "::NativeInt16Identity")) {
        reverse_icall_wFw_fct = (uintptr_t)method;
        return reverse_icall_wFw;
    }
    if (strstr(name, "::NativeUInt16Identity")) {
        reverse_icall_WFW_fct = (uintptr_t)method;
        return reverse_icall_WFW;
    }
    return find_reverse_icall_iFii(method);
}

EXPORT void my_mono_add_internal_call(x64emu_t* emu, const char* name, void* method)
{
    (void)emu;
    void* host_method = select_reverse_icall(name, method);
    printf_log(LOG_NONE, "RIMDROID P2 register icall name=%s guest=%p host=%p\n",
        name ? name : "(null)", method, host_method);
    my->mono_add_internal_call((void*)name, host_method);
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
