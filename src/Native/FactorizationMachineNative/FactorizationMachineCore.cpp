// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include "../Stdafx.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <pmmintrin.h>

// Compute the output value of the field-aware factorization, as the sum of the linear part and the latent part. 
// The linear part is the inner product of linearWeights and featureValues.
// The latent part is the sum of all intra-field interactions in one field f, for all fields possible. 
EXPORT_API(void) CalculateIntermediateVariablesNative(int fieldCount, int latentDim, int count, _In_ int * fieldIndices, _In_ int * featureIndices, _In_ float * featureValues,
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
    __m128 _y = _mm_setzero_ps();
    __m128 _tmp = _mm_setzero_ps();

    for (int i = 0; i < c; i++)
    {
        const int f = pf[i];
        const int j = pi[i];
        linearResponse += pw[j] * px[i];

        const __m128 _x = _mm_load1_ps(px + i);
        const __m128 _xx = _mm_mul_ps(_x, _x);

        // tmp -= <v_j,f, v_j,f> * x * x
        const int vBias = j * m * d + f * d;

        // j-th feature's latent vector in the f-th field hidden space.
        const float * vjf = pv + vBias;

        for (int k = 0; k + 4 <= d; k += 4)
        {
            const __m128 _v = _mm_load_ps(vjf + k);
            _tmp = _mm_sub_ps(_tmp, _mm_mul_ps(_mm_mul_ps(_v, _v), _xx));
        }

        for (int fprime = 0; fprime < m; fprime++)
        {
            const int vBias = j * m * d + fprime * d;
            const int qBias = f * m * d + fprime * d;
            const float * vjfprime = pv + vBias;
            float * qffprime = pq + qBias;

            // q_f,f' += v_j,f' * x
            for (int k = 0; k + 4 <= d; k += 4)
            {
                const __m128 _v = _mm_load_ps(vjfprime + k);
                __m128 _q = _mm_load_ps(qffprime + k);
                _q = _mm_add_ps(_q, _mm_mul_ps(_v, _x));
                _mm_store_ps(qffprime + k, _q);
            }
        }
    }

    for (int f = 0; f < m; f++)
    {
        // tmp += <q_f,f, q_f,f>
        const float * qff = pq + f * m * d + f * d;
        for (int k = 0; k + 4 <= d; k += 4)
        {
            __m128 _qff = _mm_load_ps(qff + k);

            // Intra-field interactions. 
            _tmp = _mm_add_ps(_tmp, _mm_mul_ps(_qff, _qff));
        }

        // y += <q_f,f', q_f',f>, f != f'
        // Whis loop handles inter - field interactions because f != f'.
        for (int fprime = f + 1; fprime < m; fprime++)
        {
            const float * qffprime = pq + f * m * d + fprime * d;
            const float * qfprimef = pq + fprime * m * d + f * d;
            for (int k = 0; k + 4 <= d; k += 4)
            {
                // Inter-field interaction.
                __m128 _qffprime = _mm_load_ps(qffprime + k);
                __m128 _qfprimef = _mm_load_ps(qfprimef + k);
                _y = _mm_add_ps(_y, _mm_mul_ps(_qffprime, _qfprimef));
            }
        }
    }

    _y = _mm_add_ps(_y, _mm_mul_ps(_mm_set_ps1(0.5f), _tmp));
    _tmp = _mm_add_ps(_y, _mm_movehl_ps(_y, _y));
    _y = _mm_add_ps(_tmp, _mm_shuffle_ps(_tmp, _tmp, 1)); // The lowest slot is the response value.
    _mm_store_ss(&latentResponse, _y);
    *response = linearResponse + latentResponse;
}

// Calculate the stochastic gradient and update the model. 
// The /*const*/ comment on the parameters of the function means that their values should not get altered by this function.
EXPORT_API(void) CalculateGradientAndUpdateNative(float lambdaLinear, float lambdaLatent, float learningRate, int fieldCount, int latentDim, float weight, int count,
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

    const __m128 _wei = _mm_set_ps1(weight);
    const __m128 _s = _mm_set_ps1(slope);
    const __m128 _lr = _mm_set_ps1(learningRate);
    const __m128 _lambdav = _mm_set_ps1(lambdaLatent);

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
        const __m128 _x = _mm_load1_ps(px + i);
        for (int fprime = 0; fprime < m; fprime++)
        {
            float * vjfprime = pv + j * m * d + fprime * d;
            float * hvjfprime = phv + j * m * d + fprime * d;
            const float * qfprimef = pq + fprime * m * d + f * d;
            const __m128 _sx = _mm_mul_ps(_s, _x);

            for (int k = 0; k + 4 <= d; k += 4)
            {
                __m128 _v = _mm_load_ps(vjfprime + k);
                __m128 _q = _mm_load_ps(qfprimef + k);

                // Calculate L2-norm regularization's gradient.
                __m128 _g = _mm_mul_ps(_lambdav, _v);

                // Calculate loss function's gradient.
                if (fprime != f)
                    _g = _mm_add_ps(_g, _mm_mul_ps(_sx, _q));
                else
                    _g = _mm_add_ps(_g, _mm_mul_ps(_sx, _mm_sub_ps(_q, _mm_mul_ps(_v, _x))));
                _g = _mm_mul_ps(_wei, _g);

                // Accumulate the gradient of latent vectors.
                const __m128 _h = _mm_add_ps(_mm_load_ps(hvjfprime + k), _mm_mul_ps(_g, _g));

                // Perform ADAGRAD update rule to adjust latent vector.
                _v = _mm_sub_ps(_v, _mm_mul_ps(_lr, _mm_mul_ps(_mm_rsqrt_ps(_h), _g)));
                _mm_store_ps(vjfprime + k, _v);
                _mm_store_ps(hvjfprime + k, _h);
            }
        }
    }
}
