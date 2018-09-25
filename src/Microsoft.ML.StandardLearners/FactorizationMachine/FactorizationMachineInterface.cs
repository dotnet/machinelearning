// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using System.Runtime.InteropServices;

using System.Security;

namespace Microsoft.ML.Runtime.FactorizationMachine
{
    internal static unsafe class FieldAwareFactorizationMachineInterface
    {
        internal const string NativePath = "FactorizationMachineNative";

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void CalculateIntermediateVariablesNative(int fieldCount, int latentDim, int count, int* /*const*/ fieldIndices, int* /*const*/ featureIndices,
            float* /*const*/ featureValues, float* /*const*/ linearWeights, float* /*const*/ latentWeights, float* latentSum, float* response);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void CalculateGradientAndUpdateNative(float lambdaLinear, float lambdaLatent, float learningRate, int fieldCount, int latentDim, float weight,
            int count, int* /*const*/ fieldIndices, int* /*const*/ featureIndices, float* /*const*/ featureValues, float* /*const*/ latentSum, float slope,
            float* linearWeights, float* latentWeights, float* linearAccumulatedSquaredGrads, float* latentAccumulatedSquaredGrads);

        public static void CalculateIntermediateVariables(int fieldCount, int latentDim, int count, int[] fieldIndices, int[] featureIndices, float[] featureValues,
            float[] linearWeights, float[] latentWeights, float[] latentSum, ref float response)
        {
            Contracts.AssertNonEmpty(fieldIndices);
            Contracts.AssertNonEmpty(featureValues);
            Contracts.AssertNonEmpty(featureIndices);
            Contracts.AssertNonEmpty(linearWeights);

            unsafe
            {
                fixed (int* pf = &fieldIndices[0])
                fixed (int* pi = &featureIndices[0])
                fixed (float* px = &featureValues[0])
                fixed (float* pw = &linearWeights[0])
                fixed (float* pv = &latentWeights[0])
                fixed (float* pq = &latentSum[0])
                fixed (float* pr = &response)
                    CalculateIntermediateVariablesNative(fieldCount, latentDim, count, pf, pi, px, pw, pv, pq, pr);
            }
        }

        public static void CalculateGradientAndUpdate(float lambdaLinear, float lambdaLatent, float learningRate, int fieldCount, int latentDim,
            float weight, int count, int[] fieldIndices, int[] featureIndices, float[] featureValues, float[] latentSum, float slope,
            float[] linearWeights, float[] latentWeights, float[] linearAccumulatedSquaredGrads, float[] latentAccumulatedSquaredGrads)
        {
            Contracts.AssertNonEmpty(fieldIndices);
            Contracts.AssertNonEmpty(featureIndices);
            Contracts.AssertNonEmpty(featureValues);
            Contracts.AssertNonEmpty(linearWeights);
            Contracts.AssertNonEmpty(linearAccumulatedSquaredGrads);

            unsafe
            {
                fixed (int* pf = &fieldIndices[0])
                fixed (int* pi = &featureIndices[0])
                fixed (float* px = &featureValues[0])
                fixed (float* pq = &latentSum[0])
                fixed (float* pw = &linearWeights[0])
                fixed (float* pv = &latentWeights[0])
                fixed (float* phw = &linearAccumulatedSquaredGrads[0])
                fixed (float* phv = &latentAccumulatedSquaredGrads[0])
                    CalculateGradientAndUpdateNative(lambdaLinear, lambdaLatent, learningRate, fieldCount, latentDim, weight, count, pf, pi, px,
                        pq, slope, pw, pv, phw, phv);
            }

        }
    }
}
