// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Microsoft.ML.Internal.CpuMath.Core;

namespace Microsoft.ML.Internal.CpuMath.FactorizationMachine
{
    internal static class AvxIntrinsics
    {
        private static readonly Vector256<float> _point5 = Vector256.Create(0.5f);

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> MultiplyAdd(Vector256<float> src1, Vector256<float> src2, Vector256<float> src3)
        {
            if (Fma.IsSupported)
            {
                return Fma.MultiplyAdd(src1, src2, src3);
            }
            else
            {
                Vector256<float> product = Avx.Multiply(src1, src2);
                return Avx.Add(product, src3);
            }
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> MultiplyAddNegated(Vector256<float> src1, Vector256<float> src2, Vector256<float> src3)
        {
            if (Fma.IsSupported)
            {
                return Fma.MultiplyAddNegated(src1, src2, src3);
            }
            else
            {
                Vector256<float> product = Avx.Multiply(src1, src2);
                return Avx.Subtract(src3, product);
            }
        }

        // This function implements Algorithm 1 in https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf.
        // Compute the output value of the field-aware factorization, as the sum of the linear part and the latent part.
        // The linear part is the inner product of linearWeights and featureValues.
        // The latent part is the sum of all intra-field interactions in one field f, for all fields possible
        public static unsafe void CalculateIntermediateVariables(int* fieldIndices, int* featureIndices, float* featureValues,
            float* linearWeights, float* latentWeights, float* latentSum, float* response, int fieldCount, int latentDim, int count)
        {
            Contracts.Assert(Avx.IsSupported);

            // The number of all possible fields.
            int m = fieldCount;
            int d = latentDim;
            int c = count;
            int* pf = fieldIndices;
            int* pi = featureIndices;
            float* px = featureValues;
            float* pw = linearWeights;
            float* pv = latentWeights;
            float* pq = latentSum;
            float linearResponse = 0;
            float latentResponse = 0;

            Unsafe.InitBlock(pq, 0, (uint)(m * m * d * sizeof(float)));

            Vector256<float> y = Vector256<float>.Zero;
            Vector256<float> tmp = Vector256<float>.Zero;

            for (int i = 0; i < c; i++)
            {
                int f = pf[i];
                int j = pi[i];
                linearResponse += pw[j] * px[i];

                Vector256<float> x = Avx.BroadcastScalarToVector256(px + i);
                Vector256<float> xx = Avx.Multiply(x, x);

                // tmp -= <v_j,f, v_j,f> * x * x
                int vBias = j * m * d + f * d;

                // j-th feature's latent vector in the f-th field hidden space.
                float* vjf = pv + vBias;

                for (int k = 0; k + 8 <= d; k += 8)
                {
                    Vector256<float> vjfBuffer = Avx.LoadVector256(vjf + k);
                    tmp = MultiplyAddNegated(Avx.Multiply(vjfBuffer, vjfBuffer), xx, tmp);
                }

                for (int fprime = 0; fprime < m; fprime++)
                {
                    vBias = j * m * d + fprime * d;
                    int qBias = f * m * d + fprime * d;
                    float* vjfprime = pv + vBias;
                    float* qffprime = pq + qBias;

                    // q_f,f' += v_j,f' * x
                    for (int k = 0; k + 8 <= d; k += 8)
                    {
                        Vector256<float> vjfprimeBuffer = Avx.LoadVector256(vjfprime + k);
                        Vector256<float> q = Avx.LoadVector256(qffprime + k);
                        q = MultiplyAdd(vjfprimeBuffer, x, q);
                        Avx.Store(qffprime + k, q);
                    }
                }
            }

            for (int f = 0; f < m; f++)
            {
                // tmp += <q_f,f, q_f,f>
                float* qff = pq + f * m * d + f * d;
                for (int k = 0; k + 8 <= d; k += 8)
                {
                    Vector256<float> qffBuffer = Avx.LoadVector256(qff + k);

                    // Intra-field interactions.
                    tmp = MultiplyAdd(qffBuffer, qffBuffer, tmp);
                }

                // y += <q_f,f', q_f',f>, f != f'
                // Whis loop handles inter - field interactions because f != f'.
                for (int fprime = f + 1; fprime < m; fprime++)
                {
                    float* qffprime = pq + f * m * d + fprime * d;
                    float* qfprimef = pq + fprime * m * d + f * d;
                    for (int k = 0; k + 8 <= d; k += 8)
                    {
                        // Inter-field interaction.
                        Vector256<float> qffprimeBuffer = Avx.LoadVector256(qffprime + k);
                        Vector256<float> qfprimefBuffer = Avx.LoadVector256(qfprimef + k);
                        y = MultiplyAdd(qffprimeBuffer, qfprimefBuffer, y);
                    }
                }
            }

            y = MultiplyAdd(_point5, tmp, y);
            tmp = Avx.Add(y, Avx.Permute2x128(y, y, 1));
            tmp = Avx.HorizontalAdd(tmp, tmp);
            y = Avx.HorizontalAdd(tmp, tmp);
            Sse.StoreScalar(&latentResponse, y.GetLower()); // The lowest slot is the response value.
            *response = linearResponse + latentResponse;
        }

        // This function implements Algorithm 2 in https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
        // Calculate the stochastic gradient and update the model.
        public static unsafe void CalculateGradientAndUpdate(int* fieldIndices, int* featureIndices, float* featureValues, float* latentSum, float* linearWeights,
            float* latentWeights, float* linearAccumulatedSquaredGrads, float* latentAccumulatedSquaredGrads, float lambdaLinear, float lambdaLatent, float learningRate,
            int fieldCount, int latentDim, float weight, int count, float slope)
        {
            Contracts.Assert(Avx.IsSupported);

            int m = fieldCount;
            int d = latentDim;
            int c = count;
            int* pf = fieldIndices;
            int* pi = featureIndices;
            float* px = featureValues;
            float* pq = latentSum;
            float* pw = linearWeights;
            float* pv = latentWeights;
            float* phw = linearAccumulatedSquaredGrads;
            float* phv = latentAccumulatedSquaredGrads;

            Vector256<float> wei = Vector256.Create(weight);
            Vector256<float> s = Vector256.Create(slope);
            Vector256<float> lr = Vector256.Create(learningRate);
            Vector256<float> lambdav = Vector256.Create(lambdaLatent);

            for (int i = 0; i < count; i++)
            {
                int f = pf[i];
                int j = pi[i];

                // Calculate gradient of linear term w_j.
                float g = weight * (lambdaLinear * pw[j] + slope * px[i]);

                // Accumulate the gradient of the linear term.
                phw[j] += g * g;

                // Perform ADAGRAD update rule to adjust linear term.
                pw[j] -= learningRate / MathF.Sqrt(phw[j]) * g;

                // Update latent term, v_j,f', f'=1,...,m.
                Vector256<float> x = Avx.BroadcastScalarToVector256(px + i);

                for (int fprime = 0; fprime < m; fprime++)
                {
                    float* vjfprime = pv + j * m * d + fprime * d;
                    float* hvjfprime = phv + j * m * d + fprime * d;
                    float* qfprimef = pq + fprime * m * d + f * d;
                    Vector256<float> sx = Avx.Multiply(s, x);

                    for (int k = 0; k + 8 <= d; k += 8)
                    {
                        Vector256<float> v = Avx.LoadVector256(vjfprime + k);
                        Vector256<float> q = Avx.LoadVector256(qfprimef + k);

                        // Calculate L2-norm regularization's gradient.
                        Vector256<float> gLatent = Avx.Multiply(lambdav, v);

                        Vector256<float> tmp = q;

                        // Calculate loss function's gradient.
                        if (fprime == f)
                            tmp = MultiplyAddNegated(v, x, q);
                        gLatent = MultiplyAdd(sx, tmp, gLatent);
                        gLatent = Avx.Multiply(wei, gLatent);

                        // Accumulate the gradient of latent vectors.
                        Vector256<float> h = MultiplyAdd(gLatent, gLatent, Avx.LoadVector256(hvjfprime + k));

                        // Perform ADAGRAD update rule to adjust latent vector.
                        v = MultiplyAddNegated(lr, Avx.Multiply(Avx.ReciprocalSqrt(h), gLatent), v);
                        Avx.Store(vjfprime + k, v);
                        Avx.Store(hvjfprime + k, h);
                    }
                }
            }
        }
    }
}
