// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Internal.CpuMath;

namespace Microsoft.ML.Runtime.Sweeper.Algorithms
{
    public sealed class SweeperProbabilityUtils
    {
        private readonly Random _rng;

        public SweeperProbabilityUtils(IHost host)
        {
            _rng = new Random(host.Rand.Next());
        }

        public static double Sum(double[] a)
        {
            double total = 0;
            foreach (double d in a)
                total += d;
            return total;
        }

        public static double NormalPdf(double x, double mean, double variance)
        {
            const double minVariance = 1e-200;
            variance = Math.Max(variance, minVariance);
            return 1 / Math.Sqrt(2 * Math.PI * variance) * Math.Exp(-Math.Pow(x - mean, 2) / (2 * variance));
        }

        public static double NormalCdf(double x, double mean, double variance)
        {
            double centered = x - mean;
            double ztrans = centered / (Math.Sqrt(variance) * Math.Sqrt(2));

            return 0.5 * (1 + ProbabilityFunctions.Erf(ztrans));
        }

        public static double StdNormalPdf(double x)
        {
            return 1 / Math.Sqrt(2 * Math.PI) * Math.Exp(-Math.Pow(x, 2) / 2);
        }

        public static double StdNormalCdf(double x)
        {
            return 0.5 * (1 + ProbabilityFunctions.Erf(x * 1 / Math.Sqrt(2)));
        }

        /// <summary>
        /// Samples from a Gaussian Normal with mean mu and std dev sigma.
        /// </summary>
        /// <param name="numRVs">Number of samples</param>
        /// <param name="mu">mean</param>
        /// <param name="sigma">standard deviation</param>
        /// <returns></returns>
        public double[] NormalRVs(int numRVs, double mu, double sigma)
        {
            List<double> rvs = new List<double>();
            double u1;
            double u2;

            for (int i = 0; i < numRVs; i++)
            {
                u1 = _rng.NextDouble();
                u2 = _rng.NextDouble();
                rvs.Add(mu + sigma * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
            }

            return rvs.ToArray();
        }

        /// <summary>
        /// This performs (slow) roulette-wheel sampling of a categorical distribution. Should be swapped for other
        /// method as soon as one is available. 
        /// </summary>
        /// <param name="numSamples">Number of samples to draw.</param>
        /// <param name="weights">Weights for distribution (should sum to 1).</param>
        /// <returns>A set of indicies indicating which element was chosen for each sample.</returns>
        public int[] SampleCategoricalDistribution(int numSamples, double[] weights)
        {
            // Normalize weights if necessary.
            double total = Sum(weights);
            if (Math.Abs(1.0 - total) > 0.0001)
                weights = Normalize(weights);

            // Build roulette wheel.
            double[] rw = new double[weights.Length];
            double cs = 0.0;
            for (int i = 0; i < weights.Length; i++)
            {
                cs += weights[i];
                rw[i] = cs;
            }

            // Draw samples.
            int[] results = new int[numSamples];
            for (int i = 0; i < results.Length; i++)
            {
                double u = _rng.NextDouble();
                results[i] = BinarySearch(rw, u, 0, rw.Length - 1);
            }

            return results;
        }

        public double SampleUniform()
        {
            return _rng.NextDouble();
        }

        /// <summary>
        /// Simple binary search method for finding smallest index in array where value 
        /// meets or exceeds what you're looking for.
        /// </summary>
        /// <param name="a">Array to search</param>
        /// <param name="u">Value to search for</param>
        /// <param name="low">Left boundary of search</param>
        /// <param name="high">Right boundary of search</param>
        /// <returns></returns>
        private int BinarySearch(double[] a, double u, int low, int high)
        {
            int diff = high - low;
            if (diff < 2)
                return a[low] >= u ? low : high;
            int mid = low + (diff / 2);
            return a[mid] >= u ? BinarySearch(a, u, low, mid) : BinarySearch(a, u, mid, high);
        }

        public static double[] Normalize(double[] weights)
        {
            double total = Sum(weights);

            // If all weights equal zero, set to 1 (to avoid divide by zero).
            if (total <= Double.Epsilon)
            {
                weights.SetValue(1, 0, weights.Length);
                total = weights.Length;
            }

            for (int i = 0; i < weights.Length; i++)
                weights[i] /= total;
            return weights;
        }

        public static double[] InverseNormalize(double[] weights)
        {
            weights = Normalize(weights);

            for (int i = 0; i < weights.Length; i++)
                weights[i] = 1 - weights[i];

            return Normalize(weights);
        }

        public static Float[] ParameterSetAsFloatArray(IHost host, IValueGenerator[] sweepParams, ParameterSet ps, bool expandCategoricals = true)
        {
            host.Assert(ps.Count == sweepParams.Length);

            var result = new List<Float>();

            for (int i = 0; i < sweepParams.Length; i++)
            {
                // This allows us to query possible values of this parameter.
                var sweepParam = sweepParams[i];

                // This holds the actual value for this parameter, chosen in this parameter set.
                var pset = ps[sweepParam.Name];
                host.AssertValue(pset);

                var parameterDiscrete = sweepParam as DiscreteValueGenerator;
                if (parameterDiscrete != null)
                {
                    int hotIndex = -1;
                    for (int j = 0; j < parameterDiscrete.Count; j++)
                    {
                        if (parameterDiscrete[j].Equals(pset))
                        {
                            hotIndex = j;
                            break;
                        }
                    }
                    host.Assert(hotIndex >= 0);

                    if (expandCategoricals)
                        for (int j = 0; j < parameterDiscrete.Count; j++)
                            result.Add(j == hotIndex ? 1 : 0);
                    else
                        result.Add(hotIndex);
                }
                else if (sweepParam is LongValueGenerator lvg)
                {
                    // Normalizing all numeric parameters to [0,1] range.
                    result.Add(lvg.NormalizeValue(new LongParameterValue(pset.Name, long.Parse(pset.ValueText))));
                }
                else if (sweepParam is FloatValueGenerator fvg)
                {
                    // Normalizing all numeric parameters to [0,1] range.
                    result.Add(fvg.NormalizeValue(new FloatParameterValue(pset.Name, float.Parse(pset.ValueText))));
                }
                else
                    throw host.Except("Smart sweeper can only sweep over discrete and numeric parameters");
            }

            return result.ToArray();
        }

        public static ParameterSet FloatArrayAsParameterSet(IHost host, IValueGenerator[] sweepParams, Float[] array, bool expandedCategoricals = true)
        {
            Contracts.Assert(array.Length == sweepParams.Length);

            List<IParameterValue> parameters = new List<IParameterValue>();
            int currentArrayIndex = 0;
            for (int i = 0; i < sweepParams.Length; i++)
            {
                var parameterDiscrete = sweepParams[i] as DiscreteValueGenerator;
                if (parameterDiscrete != null)
                {
                    if (expandedCategoricals)
                    {
                        int hotIndex = -1;
                        for (int j = 0; j < parameterDiscrete.Count; j++)
                        {
                            if (array[i + j] > 0)
                            {
                                hotIndex = j;
                                break;
                            }
                        }
                        host.Assert(hotIndex >= i);
                        parameters.Add(new StringParameterValue(sweepParams[i].Name, parameterDiscrete[hotIndex].ValueText));
                        currentArrayIndex += parameterDiscrete.Count;
                    }
                    else
                    {
                        parameters.Add(new StringParameterValue(sweepParams[i].Name, parameterDiscrete[(int)array[currentArrayIndex]].ValueText));
                        currentArrayIndex++;
                    }
                }
                else
                {
                    parameters.Add(sweepParams[i].CreateFromNormalized(array[currentArrayIndex]));
                    currentArrayIndex++;
                }
            }

            return new ParameterSet(parameters);
        }
    }
}
