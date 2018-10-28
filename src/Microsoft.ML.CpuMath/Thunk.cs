// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Security;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    internal static unsafe class Thunk
    {
        internal const string NativePath = "CpuMathNative";

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMul(/*const*/ float* pmat, /*const*/ float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulP(/*const*/ float* pmat, /*const*/ int* pposSrc, /*const*/ float* psrc,
            int posMin, int iposMin, int iposLim, float* pdst, int crow, int ccol);

        // These treat pmat as if it is stored in column-major order. Thus, crow and ccol are the numbers of rows
        // and columns from that perspective. Alternatively, crow is the number of rows in the transpose of pmat
        // (thought of as row-major order).
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTran(/*const*/ float* pmat, /*const*/ float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void Scale(float a, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleSrcU(float a, /*const*/ float* ps, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleAddU(float a, float b, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleU(float a, /*const*/ float* ps, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleSU(float a, /*const*/ float* ps, /*const*/ int* pi, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleCopyU(float a, /*const*/ float* ps, /*const*/ float* pd, float* pr, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScalarU(float a, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddU(/*const*/ float* ps, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddSU(/*const*/ float* ps, /*const*/ int* pi, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float Sum(/*const*/ float* pValues, int length);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumSqU(/*const*/ float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumSqDiffU(float mean, /*const*/ float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumAbsU(/*const*/ float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumAbsDiffU(float mean, /*const*/ float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float MulElementWiseU(/*const*/ float* ps1, /*const*/float* ps2, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float MaxAbsU(/*const*/ float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float MaxAbsDiffU(float mean, /*const*/ float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float DotU(/*const*/ float* pa, /*const*/ float* pb, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float DotSU(/*const*/ float* pa, /*const*/ float* pb, /*const*/ int* pi, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float Dist2(/*const*/ float* px, /*const*/ float* py, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ZeroItemsU(float* pd, int c, /*const*/ int* pindices, int cindices);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ZeroMatrixItemsCore(float* pd, int c, int ccol, int cfltRow, /*const*/ int* pindices, int cindices);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]

        public static extern void SdcaL1UpdateU(float primalUpdate, /*const*/ float* ps, float threshold, float* pd1, float* pd2, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void SdcaL1UpdateSU(float primalUpdate, /*const*/ float* ps, /*const*/ int* pi, float threshold, float* pd1, float* pd2, int c);

#if !CORECLR
        // In CoreCLR we use Buffer.MemoryCopy directly instead of
        // plumbing our own version.
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MemCpy(void* dst, /*const*/ void* src, long count);
#endif
    }
}
