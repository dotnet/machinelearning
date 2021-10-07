// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using Microsoft.ML.Internal.CpuMath;

namespace Microsoft.ML.AutoML
{
    internal sealed class SweeperProbabilityUtils
    {
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
                u1 = AutoMlUtils.Random.Value.NextDouble();
                u2 = AutoMlUtils.Random.Value.NextDouble();
                rvs.Add(mu + sigma * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
            }

            return rvs.ToArray();
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

        public static float[] ParameterSetAsFloatArray(IValueGenerator[] sweepParams, ParameterSet ps, bool expandCategoricals = true)
        {
            Runtime.Contracts.Assert(ps.Count == sweepParams.Length);

            var result = new List<float>();

            for (int i = 0; i < sweepParams.Length; i++)
            {
                // This allows us to query possible values of this parameter.
                var sweepParam = sweepParams[i];

                // This holds the actual value for this parameter, chosen in this parameter set.
                var pset = ps[sweepParam.Name];
                Runtime.Contracts.Assert(pset != null);

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
                    Runtime.Contracts.Assert(hotIndex >= 0);

                    if (expandCategoricals)
                        for (int j = 0; j < parameterDiscrete.Count; j++)
                            result.Add(j == hotIndex ? 1 : 0);
                    else
                        result.Add(hotIndex);
                }
                else if (sweepParam is LongValueGenerator lvg)
                {
                    var longValue = GetIfIParameterValueOfT<long>(pset) ?? long.Parse(pset.ValueText, CultureInfo.InvariantCulture);
                    // Normalizing all numeric parameters to [0,1] range.
                    result.Add(lvg.NormalizeValue(new LongParameterValue(pset.Name, longValue)));
                }
                else if (sweepParam is FloatValueGenerator fvg)
                {
                    var floatValue = GetIfIParameterValueOfT<float>(pset) ?? float.Parse(pset.ValueText, CultureInfo.InvariantCulture);
                    // Normalizing all numeric parameters to [0,1] range.
                    result.Add(fvg.NormalizeValue(new FloatParameterValue(pset.Name, floatValue)));
                }
                else
                {
                    throw new InvalidOperationException("Smart sweeper can only sweep over discrete and numeric parameters");
                }
            }

            return result.ToArray();
        }

        private static T? GetIfIParameterValueOfT<T>(IParameterValue parameterValue)
            where T : struct =>
            parameterValue is IParameterValue<T> pvt ? pvt.Value : default(T?);

        public static ParameterSet FloatArrayAsParameterSet(IValueGenerator[] sweepParams, float[] array, bool expandedCategoricals = true)
        {
            Runtime.Contracts.Assert(array.Length == sweepParams.Length);

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
                        Runtime.Contracts.Assert(hotIndex >= i);
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
