#include <dlfcn.h>
#include <stdlib.h>

#include "wrappedlibs.h"

#include "librarian/library_private.h"

const char* rdmonoprobeName = "librdmonoprobe.so";
#define LIBNAME rdmonoprobe

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
