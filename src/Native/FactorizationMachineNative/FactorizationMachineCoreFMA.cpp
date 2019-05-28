// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include "../Stdafx.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <immintrin.h>

EXPORT_API(void) CalculateIntermediateVariablesNativeFMA(int fieldCount, int latentDim, int count, _In_ int * fieldIndices, _In_ int * featureIndices, _In_ float * featureValues,
    _In_ float * linearWeights, _In_ float * latentWeights, _Inout_ float * latentSum, _Out_ float * response)
{
    // The number of all possible fields.
    const int m = fieldCount;
    const int d = latentDim;
    const int c = count;
    const int * pf = fieldIndices;
    const int * pi = featureIndices;
    const float * px = featureValues;
    const float * pw = linearWeights;
    const float * pv = latentWeights;
    float * pq = latentSum;
    float linearResponse = 0;
    float latentResponse = 0;

    memset(pq, 0, sizeof(float) * m * m * d);
    __m256 _y = _mm256_setzero_ps();
    __m256 _tmp = _mm256_setzero_ps();

    for (int i = 0; i < c; i++)
    {
        const int f = pf[i];
        const int j = pi[i];
        linearResponse += pw[j] * px[i];

        const __m256 _x = _mm256_broadcast_ss(px + i);
        const __m256 _xx = _mm256_mul_ps (_x, _x);

         // tmp -= <v_j,f, v_j,f> * x * x
        const int vBias = j * m * d + f * d;

        // j-th feature's latent vector in the f-th field hidden space.
        const float * vjf = pv + vBias;

        for (int k = 0; k + 8 <= d; k += 8)
        {
            const __m256 _v = _mm256_load_ps(vjf + k);
            _tmp = _mm256_fnmadd_ps(_mm256_mul_ps(_v, _v), _xx, _tmp);
        }

        for (int fprime = 0; fprime < m; fprime++)
        {
            const int vBias = j * m * d + fprime * d;
            const int qBias = f * m * d + fprime * d;
            const float * vjfprime = pv + vBias;
            float * qffprime = pq + qBias;

            // q_f,f' += v_j,f' * x
            for (int k = 0; k + 8 <= d; k += 8)
            {
                const __m256 _v = _mm256_load_ps(vjfprime + k);
                __m256 _q = _mm256_load_ps(qffprime + k);
                _q = _mm256_fmadd_ps(_v, _x, _q);
                _mm256_store_ps(qffprime + k, _q);
            }
        }
    }

    for (int f = 0; f < m; f++)
    {
        // tmp += <q_f,f, q_f,f>
        const float * qff = pq + f * m * d + f * d;
        for (int k = 0; k + 8 <= d; k += 8)
        {
            __m256 _qff = _mm256_load_ps(qff + k);

            // Intra-field interactions.
            _tmp = _mm256_fmadd_ps(_qff, _qff, _tmp);

        }

        // y += <q_f,f', q_f',f>, f != f'
        // Whis loop handles inter - field interactions because f != f'.
        for (int fprime = f + 1; fprime < m; fprime++)
        {
            const float * qffprime = pq + f * m * d + fprime * d;
            const float * qfprimef = pq + fprime * m * d + f * d;
            for (int k = 0; k + 8 <= d; k += 8)
            {
                // Inter-field interaction.
                __m256 _qffprime = _mm256_load_ps(qffprime + k);
                __m256 _qfprimef = _mm256_load_ps(qfprimef + k);
                _y = _mm256_fmadd_ps(_qffprime, _qfprimef, _y);
            }
        }
    }

    _y = _mm256_fmadd_ps(_mm256_set1_ps(0.5f), _tmp, _y);
    _tmp = _mm256_add_ps(_y, _mm256_permute2f128_ps(_y, _y, 1));
    _tmp = _mm256_hadd_ps(_tmp, _tmp);
    _y = _mm256_hadd_ps(_tmp, _tmp);
    _mm_store_ss(&latentResponse, _mm256_castps256_ps128(_y));
    *response = linearResponse + latentResponse;

}

// Calculate the stochastic gradient and update the model.
// The /*const*/ comment on the parameters of the function means that their values should not get altered by this function.
EXPORT_API(void) CalculateGradientAndUpdateNativeFMA(float lambdaLinear, float lambdaLatent, float learningRate, int fieldCount, int latentDim, float weight, int count,
    _In_ int* /*const*/ fieldIndices, _In_ int* /*const*/ featureIndices, _In_ float* /*const*/ featureValues, _In_ float* /*const*/ latentSum, float slope,
    _Inout_ float* linearWeights, _Inout_ float* latentWeights, _Inout_ float* linearAccumulatedSquaredGrads, _Inout_ float* latentAccumulatedSquaredGrads)
{
    const int m = fieldCount;
    const int d = latentDim;
    const int c = count;
    const int * pf = fieldIndices;
    const int * pi = featureIndices;
    const float * px = featureValues;
    const float * pq = latentSum;
    float * pw = linearWeights;
    float * pv = latentWeights;
    float * phw = linearAccumulatedSquaredGrads;
    float * phv = latentAccumulatedSquaredGrads;

    const __m256 _wei = _mm256_set1_ps(weight);
    const __m256 _s = _mm256_set1_ps(slope);
    const __m256 _lr = _mm256_set1_ps(learningRate);
    const __m256 _lambdav = _mm256_set1_ps(lambdaLatent);

    for (int i = 0; i < count; i++)
    {
        const int f = pf[i];
        const int j = pi[i];

        // Calculate gradient of linear term w_j.
        float g = weight * (lambdaLinear * pw[j] + slope * px[i]);

        // Accumulate the gradient of the linear term.
        phw[j] += g * g;

        // Perform ADAGRAD update rule to adjust linear term.
        pw[j] -= learningRate / sqrt(phw[j]) * g;

        // Update latent term, v_j,f', f'=1,...,m.
        const __m256 _x = _mm256_broadcast_ss(px + i);
        for (int fprime = 0; fprime < m; fprime++)
        {
            float * vjfprime = pv + j * m * d + fprime * d;
            float * hvjfprime = phv + j * m * d + fprime * d;
            const float * qfprimef = pq + fprime * m * d + f * d;
            const __m256 _sx = _mm256_mul_ps(_s, _x);

            for (int k = 0; k + 8 <= d; k += 8)
            {
                __m256 _v = _mm256_load_ps(vjfprime + k);
                __m256 _q = _mm256_load_ps(qfprimef + k);

                // Calculate L2-norm regularization's gradient.
                __m256 _g = _mm256_mul_ps(_lambdav, _v);

                // Calculate loss function's gradient.
                if (fprime != f)
                   _g = _mm256_fmadd_ps(_sx, _q, _g);
                else
                   _g = _mm256_fmadd_ps(_sx, _mm256_sub_ps(_q, _mm256_mul_ps(_v, _x)), _g);
                _g = _mm256_mul_ps(_wei, _g);

                // Accumulate the gradient of latent vectors.
                const __m256 _h = _mm256_fmadd_ps(_g, _g, _mm256_load_ps(hvjfprime + k));

                // Perform ADAGRAD update rule to adjust latent vector.
                _v = _mm256_fnmadd_ps(_lr, _mm256_mul_ps(_mm256_rsqrt_ps(_h), _g), _v);
                _mm256_store_ps(vjfprime + k, _v);
                _mm256_store_ps(hvjfprime + k, _h);
            }
        }
    }
}