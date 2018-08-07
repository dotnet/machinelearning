// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.InteropServices;
using System.Security;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    internal static class CpuMathNativeUtils
    {
        [DllImport("CpuMathNative", EntryPoint = "DotU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float DotU(/*const*/ float* pa, /*const*/ float* pb, int c);

        [DllImport("CpuMathNative", EntryPoint = "DotSU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float DotSU(/*const*/ float* pa, /*const*/ float* pb, /*const*/ int* pi, int c);

        [DllImport("CpuMathNative", EntryPoint = "SumSqU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float SumSqU(/*const*/ float* ps, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void AddU(/*_In_ const*/ float* ps, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddSU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void AddSU(/*_In_ const*/ float* ps, /*_In_ const*/ int* pi, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddScaleU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void AddScaleU(float a, /*_In_ const*/ float* ps, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "AddScaleSU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void AddScaleSU(float a, /*_In_ const*/ float* ps, /*_In_ const*/ int* pi, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "ScaleU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void ScaleU(float a, /*_Inout_*/ float* pd, int c);

        [DllImport("CpuMathNative", EntryPoint = "Dist2"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float Dist2(/*const*/ float* px, /*const*/ float* py, int c);

        [DllImport("CpuMathNative", EntryPoint = "SumAbsU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe float SumAbsU(/*const*/ float* ps, int c);

        [DllImport("CpuMathNative", EntryPoint = "MulElementWiseU"), SuppressUnmanagedCodeSecurity]
        internal static extern unsafe void MulElementWiseU(/*_In_ const*/ float* ps1, /*_In_ const*/ float* ps2, /*_Inout_*/ float* pd, int c);
    }
}
