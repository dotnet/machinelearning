// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// ARM replacement for Intel MKL (libMklImports.so).
//
// This provides a small, self-contained libMklImports for arm/arm64 that
// covers exactly the symbols SymSGD needs, with no external BLAS dependency.
// That is important because the cross-compilation sysroots used in CI do not
// ship OpenBLAS (or any system BLAS), so linking against one is not an option.
//
// SymSGD uses only four CBLAS routines:
//   * cblas_sdot / cblas_saxpy   - dense single-precision dot and AXPY,
//   * cblas_sdoti / cblas_saxpyi - their sparse counterparts (MKL extensions).
// All four are implemented below as plain C loops. With -O3 the compiler
// autovectorizes the dense paths to NEON, matching hand-written BLAS closely.
//
// MKL DFTI (FFT) functions are stubbed — they are referenced by the managed
// MKL Components initializer but not used by SymSGD. The stubs return error
// codes so any actual FFT call fails cleanly rather than crashing.

// The native build is compiled with -fvisibility=hidden, so every symbol that
// must be visible to SymSgdNative (the CBLAS routines) or to the managed
// P/Invoke layer (DftiErrorMessage) has to be exported explicitly.
#if defined(_WIN32)
#define MKLIMPORTS_EXPORT __declspec(dllexport)
#else
#define MKLIMPORTS_EXPORT __attribute__((visibility("default")))
#endif

// --- Dense BLAS (CBLAS, level 1) ---

MKLIMPORTS_EXPORT float cblas_sdot(const int n, const float *x, const int incx,
                 const float *y, const int incy)
{
    float result = 0.0f;
    if (incx == 1 && incy == 1)
    {
        for (int i = 0; i < n; i++)
            result += x[i] * y[i];
    }
    else
    {
        int ix = incx < 0 ? (1 - n) * incx : 0;
        int iy = incy < 0 ? (1 - n) * incy : 0;
        for (int i = 0; i < n; i++, ix += incx, iy += incy)
            result += x[ix] * y[iy];
    }
    return result;
}

MKLIMPORTS_EXPORT void cblas_saxpy(const int n, const float a, const float *x, const int incx,
                 float *y, const int incy)
{
    if (a == 0.0f)
        return;
    if (incx == 1 && incy == 1)
    {
        for (int i = 0; i < n; i++)
            y[i] += a * x[i];
    }
    else
    {
        int ix = incx < 0 ? (1 - n) * incx : 0;
        int iy = incy < 0 ? (1 - n) * incy : 0;
        for (int i = 0; i < n; i++, ix += incx, iy += incy)
            y[iy] += a * x[ix];
    }
}

// --- Sparse BLAS (MKL extensions, not in standard BLAS) ---

MKLIMPORTS_EXPORT void cblas_saxpyi(const int nz, const float a,
                  const float *x, const int *indx, float *y)
{
    for (int i = 0; i < nz; i++)
        y[indx[i]] += a * x[i];
}

MKLIMPORTS_EXPORT float cblas_sdoti(const int nz, const float *x,
                  const int *indx, const float *y)
{
    float result = 0.0f;
    for (int i = 0; i < nz; i++)
        result += x[i] * y[indx[i]];
    return result;
}

// --- DFTI (FFT) stubs ---

MKLIMPORTS_EXPORT const char* DftiErrorMessage(long status)
{
    return "DFTI not available (arm64 MKL shim build)";
}

MKLIMPORTS_EXPORT long DftiCreateDescriptor(void **h, int precision, int domain, int dim, ...)
{
    *h = (void*)0;
    return -1;
}

MKLIMPORTS_EXPORT long DftiSetValue(void *h, int param, ...)
{
    return -1;
}

MKLIMPORTS_EXPORT long DftiCommitDescriptor(void *h) { return -1; }
MKLIMPORTS_EXPORT long DftiComputeForward(void *h, ...) { return -1; }
MKLIMPORTS_EXPORT long DftiComputeBackward(void *h, ...) { return -1; }
MKLIMPORTS_EXPORT long DftiFreeDescriptor(void **h) { return 0; }
