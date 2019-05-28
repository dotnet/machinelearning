using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers
{
    internal static unsafe class FieldAwareFactorizationMachineInterface
    {
        internal const string NativePath = "FactorizationMachineNative";
        public const int CbAlign = 16;

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
        public static extern void CalculateIntermediateVariablesNativeSSE(int fieldCount, int latentDim, int count, int* /*const*/ fieldIndices, int* /*const*/ featureIndices,
            float* /*const*/ featureValues, float* /*const*/ linearWeights, float* /*const*/ latentWeights, float* latentSum, float* response);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void CalculateIntermediateVariablesNativeAVX(int fieldCount, int latentDim, int count, int* /*const*/ fieldIndices, int* /*const*/ featureIndices,
            float* /*const*/ featureValues, float* /*const*/ linearWeights, float* /*const*/ latentWeights, float* latentSum, float* response);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void CalculateIntermediateVariablesNativeFMA(int fieldCount, int latentDim, int count, int* /*const*/ fieldIndices, int* /*const*/ featureIndices,
            float* /*const*/ featureValues, float* /*const*/ linearWeights, float* /*const*/ latentWeights, float* latentSum, float* response);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void CalculateGradientAndUpdateNativeSSE(float lambdaLinear, float lambdaLatent, float learningRate, int fieldCount, int latentDim, float weight,
            int count, int* /*const*/ fieldIndices, int* /*const*/ featureIndices, float* /*const*/ featureValues, float* /*const*/ latentSum, float slope,
            float* linearWeights, float* latentWeights, float* linearAccumulatedSquaredGrads, float* latentAccumulatedSquaredGrads);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void CalculateGradientAndUpdateNativeAVX(float lambdaLinear, float lambdaLatent, float learningRate, int fieldCount, int latentDim, float weight,
            int count, int* /*const*/ fieldIndices, int* /*const*/ featureIndices, float* /*const*/ featureValues, float* /*const*/ latentSum, float slope,
            float* linearWeights, float* latentWeights, float* linearAccumulatedSquaredGrads, float* latentAccumulatedSquaredGrads);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void CalculateGradientAndUpdateNativeFMA(float lambdaLinear, float lambdaLatent, float learningRate, int fieldCount, int latentDim, float weight,
            int count, int* /*const*/ fieldIndices, int* /*const*/ featureIndices, float* /*const*/ featureValues, float* /*const*/ latentSum, float slope,
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
                {
                    if (Fma.IsSupported)
                        CalculateIntermediateVariablesNativeFMA(fieldCount, latentDim, count, pf, pi, px, pw, Ptr(latentWeights, pv), Ptr(latentSum, pq), pr);
                    else if (Avx.IsSupported)
                        CalculateIntermediateVariablesNativeAVX(fieldCount, latentDim, count, pf, pi, px, pw, Ptr(latentWeights, pv), Ptr(latentSum, pq), pr);
                    else
                        CalculateIntermediateVariablesNativeSSE(fieldCount, latentDim, count, pf, pi, px, pw, Ptr(latentWeights, pv), Ptr(latentSum, pq), pr);
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
                    if (Fma.IsSupported)
                        CalculateGradientAndUpdateNativeFMA(lambdaLinear, lambdaLatent, learningRate, fieldCount, latentDim, weight, count, pf, pi, px,
                            Ptr(latentSum, pq), slope, pw, Ptr(latentWeights, pv), phw, Ptr(latentAccumulatedSquaredGrads, phv));
                    else if (Avx.IsSupported)
                        CalculateGradientAndUpdateNativeAVX(lambdaLinear, lambdaLatent, learningRate, fieldCount, latentDim, weight, count, pf, pi, px,
                            Ptr(latentSum, pq), slope, pw, Ptr(latentWeights, pv), phw, Ptr(latentAccumulatedSquaredGrads, phv));
                    else
                        CalculateGradientAndUpdateNativeSSE(lambdaLinear, lambdaLatent, learningRate, fieldCount, latentDim, weight, count, pf, pi, px,
                            Ptr(latentSum, pq), slope, pw, Ptr(latentWeights, pv), phw, Ptr(latentAccumulatedSquaredGrads, phv));
                }
            }
        }
    }
}
