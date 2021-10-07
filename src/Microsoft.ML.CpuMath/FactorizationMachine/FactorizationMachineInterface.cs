// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.InteropServices;
using System.Security;
using Microsoft.ML.Internal.CpuMath.Core;

namespace Microsoft.ML.Internal.CpuMath.FactorizationMachine
{
    internal static unsafe partial class FieldAwareFactorizationMachineInterface
    {
        private const string NativePath = "CpuMathNative";
        private const int CbAlign = 16;

        private static bool Compat(AlignedArray a)
        {
            Contracts.AssertValue(a);
            Contracts.Assert(a.Size > 0);
            return a.CbAlign == CbAlign;
        }

        private static unsafe float* Ptr(AlignedArray a, float* p)
        {
            Contracts.AssertValue(a);
            float* q = p + a.GetBase((long)p);
            Contracts.Assert(((long)q & (CbAlign - 1)) == 0);
            return q;
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        private static extern void CalculateIntermediateVariablesNative(int fieldCount, int latentDim, int count, int* /*const*/ fieldIndices, int* /*const*/ featureIndices,
            float* /*const*/ featureValues, float* /*const*/ linearWeights, float* /*const*/ latentWeights, float* latentSum, float* response);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        private static extern void CalculateGradientAndUpdateNative(float lambdaLinear, float lambdaLatent, float learningRate, int fieldCount, int latentDim, float weight,
            int count, int* /*const*/ fieldIndices, int* /*const*/ featureIndices, float* /*const*/ featureValues, float* /*const*/ latentSum, float slope,
            float* linearWeights, float* latentWeights, float* linearAccumulatedSquaredGrads, float* latentAccumulatedSquaredGrads);
    }
}
