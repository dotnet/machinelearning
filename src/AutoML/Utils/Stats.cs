// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Auto
{
    internal static class Stats
    {
        /// <summary>
        /// Generates a beta-distributed random variable
        /// </summary>
        /// <param name="alpha1">first parameter</param>
        /// <param name="alpha2">second parameter</param>
        /// <returns>Sample from distribution</returns>
        public static double SampleFromBeta(double alpha1, double alpha2)
        {
            double gamma1 = SampleFromGamma(alpha1);
            double gamma2 = SampleFromGamma(alpha2);
            return gamma1 / (gamma1 + gamma2);
        }

        /// <summary>
        /// Returns a sample from the gamma distribution with scale parameter 1, shape parameter alpha
        /// </summary>
        /// <param name="alpha">Shape parameter</param>
        /// <returns>Sample from gamma distribution</returns>
        /// <remarks>Uses Marsaglia and Tsang's fast algorithm</remarks>
        public static double SampleFromGamma(double alpha)
        {
            //Contracts.CheckParam(alpha > 0, nameof(alpha), "alpha must be positive");

            if (alpha < 1)
                return SampleFromGamma(alpha + 1) * Math.Pow(AutoMlUtils.Random.NextDouble(), 1.0 / alpha);

            double d = alpha - 1.0 / 3;
            double c = 1 / Math.Sqrt(9 * d);
            double x;
            double u;
            double v;
            while (true)
            {
                do
                {
                    x = SampleFromGaussian();
                    v = Math.Pow(1.0 + c * x, 3);
                } while (v <= 0);
                u = AutoMlUtils.Random.NextDouble();
                double xSqr = x * x;
                if (u < 1.0 - 0.0331 * xSqr * xSqr ||
                    Math.Log(u) < 0.5 * xSqr + d * (1.0 - v + Math.Log(v)))
                {
                    return d * v;
                }
            }
        }

        /// <summary>
        /// Returns a number sampled from a zero-mean, unit variance Gaussian
        /// </summary>
        /// <returns>a sample</returns>
        /// <remarks>uses Joseph L. Leva's algorithm from "A fast normal random number generator", 1992</remarks>
        public static double SampleFromGaussian()
        {
            double u;
            double v;
            double q;
            do
            {
                u = AutoMlUtils.Random.NextDouble();
                v = _vScale * (AutoMlUtils.Random.NextDouble() - 0.5);
                double x = u - 0.449871;
                double y = Math.Abs(v) + 0.386595;
                q = x * x + y * (0.19600 * y - 0.25472 * x);
            } while (q > 0.27597 && (q > 0.27846 || v * v > -4 * u * u * Math.Log(u)));

            return v / u;
        }

        private static double _vScale = 2 * Math.Sqrt(2 / Math.E);
    }
}
