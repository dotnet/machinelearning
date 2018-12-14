// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Internal.Internallearn;

namespace Microsoft.ML.Runtime
{
    public sealed class QuantileStatistics : IQuantileDistribution<float>
    {
        private readonly float[] _data;
        private readonly float[] _weights;

        //This holds the cumulative sum of _weights to search the rank easily by binary search.
        private float[] _weightedSums;
        private SummaryStatistics _summaryStatistics;

        float IDistribution<float>.Minimum
        {
            get
            {
                if (_data.Length == 0)
                    return float.NaN;

                return _data[0];
            }
        }

        float IDistribution<float>.Maximum
        {
            get
            {
                if (_data.Length == 0)
                    return float.NaN;

                return _data[_data.Length - 1];
            }
        }

        float IQuantileDistribution<float>.Median { get { return ((IQuantileDistribution<float>)this).GetQuantile(0.5F); } }

        float IDistribution<float>.Mean { get { return (float)SummaryStatistics.Mean; } }

        float IDistribution<float>.StandardDeviation { get { return (float)SummaryStatistics.SampleStdDev; } }

        /// <summary>
        /// data array will be modified because of sorting if it is not already sorted yet and this class owns the data.
        /// Modifying the data outside will lead to erroneous output by this class
        /// </summary>
        public QuantileStatistics(float[] data, float[] weights = null, bool isSorted = false)
        {
            Contracts.CheckValue(data, nameof(data));
            Contracts.Check(weights == null || weights.Length == data.Length, "weights");

            _data = data;
            _weights = weights;

            if (!isSorted)
                Array.Sort(_data);
            else
                Contracts.Assert(Utils.IsSorted(_data));
        }

        /// <summary>
        /// There are many ways to estimate quantile. This implementations is based on R-8, SciPy-(1/3,1/3)
        /// https://en.wikipedia.org/wiki/Quantile#Estimating_the_quantiles_of_a_population
        /// </summary>
        float IQuantileDistribution<float>.GetQuantile(float p)
        {
            Contracts.CheckParam(0 <= p && p <= 1, nameof(p), "Probablity argument for Quantile function should be between 0 to 1 inclusive");

            if (_data.Length == 0)
                return float.NaN;

            if (p == 0 || _data.Length == 1)
                return _data[0];
            if (p == 1)
                return _data[_data.Length - 1];

            float h = GetRank(p);

            if (h <= 1)
                return _data[0];

            if (h >= _data.Length)
                return _data[_data.Length - 1];

            var hf = (int)h;
            return (float)(_data[hf - 1] + (h - hf) * (_data[hf] - _data[hf - 1]));
        }

        float[] ISampleableDistribution<float>.GetSupportSample(out float[] weights)
        {
            var result = new float[_data.Length];
            Array.Copy(_data, result, _data.Length);
            if (_weights == null)
            {
                weights = null;
            }
            else
            {
                weights = new float[_data.Length];
                Array.Copy(_weights, weights, _weights.Length);
            }

            return result;
        }

        private float GetRank(float p)
        {
            const float oneThird = (float)1 / 3;

            // holds length of the _data array if the weights is null or holds the sum of weights
            float weightedLength = _data.Length;

            if (_weights != null)
            {
                if (_weightedSums == null)
                {
                    _weightedSums = new float[_weights.Length];
                    _weightedSums[0] = _weights[0];
                    for (int i = 1; i < _weights.Length; i++)
                        _weightedSums[i] = _weights[i] + _weightedSums[i - 1];
                }

                weightedLength = _weightedSums[_weightedSums.Length - 1];
            }

            // This implementations is based on R-8, SciPy-(1/3,1/3)
            // https://en.wikipedia.org/wiki/Quantile#Estimating_the_quantiles_of_a_population
            var h = (_weights == null) ? (weightedLength + oneThird) * p + oneThird : weightedLength * p;

            if (_weights == null)
                return h;

            return _weightedSums.FindIndexSorted(h);
        }

        private SummaryStatistics SummaryStatistics
        {
            get
            {
                if (_summaryStatistics == null)
                {
                    _summaryStatistics = new SummaryStatistics();
                    if (_weights != null)
                    {
                        for (int i = 0; i < _data.Length; i++)
                            _summaryStatistics.Add(_data[i], _weights[i]);
                    }
                    else
                    {
                        for (int i = 0; i < _data.Length; i++)
                            _summaryStatistics.Add(_data[i]);
                    }
                }

                return _summaryStatistics;
            }
        }
    }
}
