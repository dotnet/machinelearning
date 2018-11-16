// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// The exported function names need to be unique (can't be disambiguated based on signature), hence
// we introduce suffix letters to indicate the general patterns used.
// * A suffix means aligned and padded for SSE operations.
// * U suffix means unaligned and unpadded.
// * S suffix means sparse (unaligned) vector.
// * P suffix means sparse (unaligned) partial vector - the vector is only part of a larger sparse vector.
// * R suffix means sparse matrix.
// * C suffix means convolution matrix.
// * D suffix means convolution matrix, with implicit source padding.
// * Tran means the matrix is transposed.

using System.Runtime.InteropServices;
using System.Security;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    internal static class CpuMathNativeUtils
    {
        [DllImport("CpuMathNative", EntryPoint = "AddScalarU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float AddScalarU(float a, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "Scale"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void Scale(float a, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "ScaleSrcU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void ScaleSrcU(float a, /*_In_ const*/ float* ps, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "ScaleAddU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void ScaleAddU(float a, float b, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddScaleU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void AddScaleU(float a, /*_In_ const*/ float* ps, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddScaleSU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void AddScaleSU(float a, /*_In_ const*/ float* ps, /*_In_ const*/ int* pi, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddScaleCopyU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void AddScaleCopyU(float a, /*_In_ const*/ float* ps, /*_In_ const*/ float* pd, /*_Inout_*/ float* pr, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void AddU(/*_In_ const*/ float* ps, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddSU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void AddSU(/*_In_ const*/ float* ps, /*_In_ const*/ int* pi, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "MulElementWiseU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void MulElementWiseU(/*_In_ const*/ float* ps1, /*_In_ const*/ float* ps2, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "Sum"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float Sum(/*const*/ float* pValues, int length);

        [DllImport("CpuMathNative", EntryPoint = "SumSqU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float SumSqU(/*const*/ float* ps, int c);

        [DllImport("CpuMathNative", EntryPoint = "SumSqDiffU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float SumSqDiffU(float mean, /*const*/ float* ps, int c);

        [DllImport("CpuMathNative", EntryPoint = "SumAbsU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float SumAbsU(/*const*/ float* ps, int c);

        [DllImport("CpuMathNative", EntryPoint = "SumAbsDiffU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float SumAbsDiffU(float mean, /*const*/ float* ps, int c);

        [DllImport("CpuMathNative", EntryPoint = "MaxAbsU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float MaxAbsU(/*const*/ float* ps, int c);

        [DllImport("CpuMathNative", EntryPoint = "MaxAbsDiffU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float MaxAbsDiffU(float mean, /*const*/ float* ps, int c);

        [DllImport("CpuMathNative", EntryPoint = "DotU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float DotU(/*const*/ float* pa, /*const*/ float* pb, int c);

        [DllImport("CpuMathNative", EntryPoint = "DotSU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float DotSU(/*const*/ float* pa, /*const*/ float* pb, /*const*/ int* pi, int c);

        [DllImport("CpuMathNative", EntryPoint = "Dist2"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float Dist2(/*const*/ float* px, /*const*/ float* py, int c);

        [DllImport("CpuMathNative", EntryPoint = "SdcaL1UpdateU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void SdcaL1UpdateU(float primalUpdate, /*_In_ const*/ float* ps, float threshold, /*_Inout_*/ float* pd1, /*_Inout_*/ float* pd2, int c);

        [DllImport("CpuMathNative", EntryPoint = "SdcaL1UpdateSU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void SdcaL1UpdateSU(float primalUpdate, /*_In_ const*/ float* ps, /*_In_ const*/ int* pi, float threshold, /*_Inout_*/ float* pd1, /*_Inout_*/ float* pd2, int c);
    }
}
