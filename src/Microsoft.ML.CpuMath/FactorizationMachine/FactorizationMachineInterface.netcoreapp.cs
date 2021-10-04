// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.Intrinsics.X86;
using Microsoft.ML.Internal.CpuMath.Core;

namespace Microsoft.ML.Internal.CpuMath.FactorizationMachine
{
    [BestFriend]
    internal static unsafe partial class FieldAwareFactorizationMachineInterface
    {
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
                {
                    if (Avx.IsSupported)
                        AvxIntrinsics.CalculateIntermediateVariables(pf, pi, px, pw, Ptr(latentWeights, pv), Ptr(latentSum, pq), pr, fieldCount, latentDim, count);
                    else
                        CalculateIntermediateVariablesNative(fieldCount, latentDim, count, pf, pi, px, pw, Ptr(latentWeights, pv), Ptr(latentSum, pq), pr);
                }
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
                {
                    if (Avx.IsSupported)
                        AvxIntrinsics.CalculateGradientAndUpdate(pf, pi, px, Ptr(latentSum, pq), pw, Ptr(latentWeights, pv),
                            phw, Ptr(latentAccumulatedSquaredGrads, phv), lambdaLinear, lambdaLatent, learningRate, fieldCount, latentDim, weight, count, slope);
                    else
                        CalculateGradientAndUpdateNative(lambdaLinear, lambdaLatent, learningRate, fieldCount, latentDim, weight, count, pf, pi, px,
                            Ptr(latentSum, pq), slope, pw, Ptr(latentWeights, pv), phw, Ptr(latentAccumulatedSquaredGrads, phv));
                }
            }
        }
    }
}
