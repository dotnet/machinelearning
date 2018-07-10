// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using System.Runtime.InteropServices;

using System.Security;

namespace Microsoft.ML.Runtime.FactorizationMachine
{
    internal unsafe static class FieldAwareFactorizationMachineInterface
    {
        internal const string NativePath = "FactorizationMachineNative";
        public const int CbAlign = 16;

        private static bool Compat(AlignedArray a)
        {
            Contracts.AssertValue(a);
            Contracts.Assert(a.Size > 0);
            return a.CbAlign == CbAlign;
        }

        private unsafe static float* Ptr(AlignedArray a, float* p)
        {
            Contracts.AssertValue(a);
            float* q = p + a.GetBase((long)p);
            Contracts.Assert(((long)q & (CbAlign - 1)) == 0);
            return q;
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void CalculateIntermediateVariablesNative(int fieldCount, int latentDim, int count, int* fieldIndices, int* featureIndices,
            float* featureValues, float* linearWeights, float* latentWeights, float* latentSum, float* response);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void CalculateGradientAndUpdateNative(float lambdaLinear, float lambdaLatent, float learningRate, int fieldCount, int latentDim, float weight,
            int count, int* fieldIndices, int* featureIndices, float* featureValues, float* latentSum, float slope,
            float* linearWeights, float* latentWeights, float* linearAccumulatedSquaredGrads, float* latentAccumulatedSquaredGrads);

        public static void CalculateIntermediateVariables(int fieldCount, int latentDim, int count, int[] fieldIndices, int[] featureIndices, float[] featureValues,
            float[] linearWeights, AlignedArray latentWeights, AlignedArray latentSum, ref float response)
        {
            Contracts.AssertNonEmpty(fieldIndices);
            Contracts.AssertNonEmpty(featureValues);
            Contracts.AssertNonEmpty(featureIndices);
            Contracts.AssertNonEmpty(linearWeights);
            Contracts.Assert(Compat(latentWeights));
            Contracts.Assert(Compat(latentSum));

            unsafe
            {
                fixed (int* pf = &fieldIndices[0])
                fixed (int* pi = &featureIndices[0])
                fixed (float* px = &featureValues[0])
                fixed (float* pw = &linearWeights[0])
                fixed (float* pv = &latentWeights.Items[0])
                fixed (float* pq = &latentSum.Items[0])
                fixed (float* pr = &response)
                    CalculateIntermediateVariablesNative(fieldCount, latentDim, count, pf, pi, px, pw, Ptr(latentWeights, pv), Ptr(latentSum, pq), pr);
            }
        }

        public static void CalculateGradientAndUpdate(float lambdaLinear, float lambdaLatent, float learningRate, int fieldCount, int latentDim,
            float weight, int count, int[] fieldIndices, int[] featureIndices, float[] featureValues, AlignedArray latentSum, float slope,
            float[] linearWeights, AlignedArray latentWeights, float[] linearAccumulatedSquaredGrads, AlignedArray latentAccumulatedSquaredGrads)
        {
            Contracts.AssertNonEmpty(fieldIndices);
            Contracts.AssertNonEmpty(featureIndices);
            Contracts.AssertNonEmpty(featureValues);
            Contracts.Assert(Compat(latentSum));
            Contracts.AssertNonEmpty(linearWeights);
            Contracts.Assert(Compat(latentWeights));
            Contracts.AssertNonEmpty(linearAccumulatedSquaredGrads);
            Contracts.Assert(Compat(latentAccumulatedSquaredGrads));

            unsafe
            {
                fixed (int* pf = &fieldIndices[0])
                fixed (int* pi = &featureIndices[0])
                fixed (float* px = &featureValues[0])
                fixed (float* pq = &latentSum.Items[0])
                fixed (float* pw = &linearWeights[0])
                fixed (float* pv = &latentWeights.Items[0])
                fixed (float* phw = &linearAccumulatedSquaredGrads[0])
                fixed (float* phv = &latentAccumulatedSquaredGrads.Items[0])
                    CalculateGradientAndUpdateNative(lambdaLinear, lambdaLatent, learningRate, fieldCount, latentDim, weight, count, pf, pi, px,
                        Ptr(latentSum, pq), slope, pw, Ptr(latentWeights, pv), phw, Ptr(latentAccumulatedSquaredGrads, phv));
            }

        }
    }
}
