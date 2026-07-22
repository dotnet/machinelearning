// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// ARM replacement for Intel MKL (libMklImports.so).
//
// Standard CBLAS functions (sgemm, sgemv, saxpy, sdot, etc.) are
// forwarded to OpenBLAS, which exports them with identical signatures.
//
// Sparse CBLAS extensions (saxpyi, sdoti) are provided here since
// OpenBLAS does not include them.
//
// MKL DFTI (FFT) functions are stubbed — they are referenced by the
// managed MKL Components initializer but not used by SymSGD. The stubs
// return error codes so any actual FFT call fails cleanly rather than
// crashing.

// --- Sparse BLAS (MKL extensions, not in OpenBLAS) ---

void cblas_saxpyi(const int nz, const float a,
                  const float *x, const int *indx, float *y)
{
    for (int i = 0; i < nz; i++)
        y[indx[i]] += a * x[i];
}

float cblas_sdoti(const int nz, const float *x,
                  const int *indx, const float *y)
{
    float result = 0.0f;
    for (int i = 0; i < nz; i++)
        result += x[i] * y[indx[i]];
    return result;
}

// --- DFTI (FFT) stubs ---

const char* DftiErrorMessage(long status)
{
    return "DFTI not available (OpenBLAS arm64 build)";
}

long DftiCreateDescriptor(void **h, int precision, int domain, int dim, ...)
{
    *h = (void*)0;
    return -1;
}

long DftiSetValue(void *h, int param, ...)
{
    return -1;
}

long DftiCommitDescriptor(void *h) { return -1; }
long DftiComputeForward(void *h, ...) { return -1; }
long DftiComputeBackward(void *h, ...) { return -1; }
long DftiFreeDescriptor(void **h) { return 0; }
