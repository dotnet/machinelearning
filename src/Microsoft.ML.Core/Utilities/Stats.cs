// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// A class containing common statistical functions
    /// </summary>
    public static class Stats
    {
        /// <summary>
        /// Returns a number uniformly sampled from 0...(rangeSize-1)
        /// </summary>
        /// <param name="rangeSize">Size of range to sample from, between 0 and int.MaxValue^2</param>
        /// <param name="rand">Random number generator</param>
        /// <returns>Sampled value</returns>
        public static long SampleLong(long rangeSize, IRandom rand)
        {
            Contracts.CheckParam(rangeSize > 0, nameof(rangeSize), "rangeSize must be positive.");

            if (rangeSize < int.MaxValue)
                return rand.Next((int)rangeSize);

            Contracts.Check(rangeSize <= (long)int.MaxValue * int.MaxValue,
                "rangeSize must be no more than int.MaxValue^2");

            int max = (int)Math.Ceiling(Math.Sqrt(rangeSize));
            while ((long)max * max < rangeSize)
                max++; // might happen due to rounding

            long result;
            do
            {
                result = (long)max * rand.Next(max) + rand.Next(max);
            } while (result >= rangeSize);

            return result;
        }

        private static double _vScale = 2 * Math.Sqrt(2 / Math.E);

        /// <summary>
        /// Returns a number sampled from a zero-mean, unit variance Gaussian
        /// </summary>
        /// <param name="rand">A Random to use for the sampling</param>
        /// <returns>a sample</returns>
        /// <remarks>uses Joseph L. Leva's algorithm from "A fast normal random number generator", 1992</remarks>
        public static double SampleFromGaussian(IRandom rand)
        {
            double u;
            double v;
            double q;
            do
            {
                u = rand.NextDouble();
                v = _vScale * (rand.NextDouble() - 0.5);
                double x = u - 0.449871;
                double y = Math.Abs(v) + 0.386595;
                q = x * x + y * (0.19600 * y - 0.25472 * x);
            } while (q > 0.27597 && (q > 0.27846 || v * v > -4 * u * u * Math.Log(u)));

            return v / u;
        }

        /// <summary>
        /// Returns a sample from the gamma distribution with scale parameter 1, shape parameter alpha
        /// </summary>
        /// <param name="alpha">Shape parameter</param>
        /// <param name="r">The random number generator to use</param>
        /// <returns>Sample from gamma distribution</returns>
        /// <remarks>Uses Marsaglia and Tsang's fast algorithm</remarks>
        public static double SampleFromGamma(IRandom r, double alpha)
        {
            Contracts.CheckParam(alpha > 0, nameof(alpha), "alpha must be positive");

            if (alpha < 1)
                return SampleFromGamma(r, alpha + 1) * Math.Pow(r.NextDouble(), 1.0 / alpha);

            double d = alpha - 1.0 / 3;
            double c = 1 / Math.Sqrt(9 * d);
            double x;
            double u;
            double v;
            while (true)
            {
                do
                {
                    x = SampleFromGaussian(r);
                    v = Math.Pow(1.0 + c * x, 3);
                } while (v <= 0);
                u = r.NextDouble();
                double xSqr = x * x;
                if (u < 1.0 - 0.0331 * xSqr * xSqr ||
                    Math.Log(u) < 0.5 * xSqr + d * (1.0 - v + Math.Log(v)))
                {
                    return d * v;
                }
            }
        }

        /// <summary>
        /// Generates a beta-distributed random variable
        /// </summary>
        /// <param name="rand">Random generator to use</param>
        /// <param name="alpha1">first parameter</param>
        /// <param name="alpha2">second parameter</param>
        /// <returns>Sample from distribution</returns>
        public static double SampleFromBeta(IRandom rand, double alpha1, double alpha2)
        {
            double gamma1 = SampleFromGamma(rand, alpha1);
            double gamma2 = SampleFromGamma(rand, alpha2);
            return gamma1 / (gamma1 + gamma2);
        }

        /// <summary>
        /// Generates a dirichlet-distributed random variable
        /// </summary>
        /// <param name="rand">Random generator to use</param>
        /// <param name="alphas">array of parameters</param>
        /// <param name="result">array in which to store resulting sample</param>
        public static void SampleFromDirichlet(IRandom rand, double[] alphas, double[] result)
        {
            Contracts.Check(alphas.Length == result.Length,
                "Dirichlet parameters must have the same dimensionality as sample space.");

            double total = 0;
            for (int i = 0; i < alphas.Length; i++)
            {
                total += result[i] = SampleFromGamma(rand, alphas[i]);
            }

            for (int i = 0; i < alphas.Length; i++)
            {
                result[i] /= total;
            }
        }

        public static int SampleFromPoisson(IRandom rand, double lambda)
        {
            if (lambda < 5)
            {
                double expLam = Math.Exp(-lambda);
                int k = 0;
                double t = rand.NextDouble();
                while (t > expLam)
                {
                    k++;
                    t *= rand.NextDouble();
                }
                return k;
            }
            else
            {
                double sqrtLam = Math.Sqrt(lambda);
                double logLam = double.NaN;
                while (true)
                {
                    double u = 0.64 * rand.NextDouble();
                    double v = -0.68 + 1.28 * rand.NextDouble();
                    double sqrV = 0;
                    if (lambda > 13.5)
                    {
                        sqrV = v * v;
                        if (v >= 0)
                        {
                            if (sqrV > 6.5 * u * (0.64 - u) * (u + 0.2))
                                continue;
                        }
                        else if (sqrV > 9.6 * u * (0.66 - u) * (u + 0.07))
                            continue;
                    }
                    int k = (int)(sqrtLam * v / u + lambda + 0.5);
                    if (k < 0)
                        continue;
                    double sqrU = u * u;
                    if (lambda > 13.5)
                    {
                        if (v >= 0)
                        {
                            if (sqrV < 15.2 * sqrU * (0.61 - u) * (0.8 - u))
                                return k;
                        }
                        else if (sqrV < 6.76 * sqrU * (0.62 - u) * (1.4 - u))
                            return k;
                    }
                    double logkFac = MathUtils.LogFactorial(k);
                    if (double.IsNaN(logLam))
                        logLam = Math.Log(lambda);
                    double p = sqrtLam * Math.Exp(-lambda + k * logLam - logkFac);
                    if (sqrU < p)
                        return k;
                }
            }
        }

        // Mean refers to the mu parameter. Scale refers to the b parameter.
        // https://en.wikipedia.org/wiki/Laplace_distribution
        public static Float SampleFromLaplacian(IRandom rand, Float mean, Float scale)
        {
            Float u = rand.NextSingle();
            u = u - 0.5f;
            Float ret = mean;
            if (u >= 0)
                ret -= scale * MathUtils.Log(1 - 2 * u);
            else
                ret += scale * MathUtils.Log(1 + 2 * u);

            return ret;
        }

        /// <summary>
        /// Sample from a standard Cauchy distribution:
        /// https://en.wikipedia.org/wiki/Lorentzian_function
        /// </summary>
        /// <param name="rand"></param>
        /// <returns></returns>
        public static Float SampleFromCauchy(IRandom rand)
        {
            return (Float)Math.Tan(Math.PI * (rand.NextSingle() - 0.5));
        }

        /// <summary>
        /// Returns a number sampled from the binomial distribution with parameters n and p
        /// </summary>
        /// <param name="r">Random generator to use</param>
        /// <param name="n">Parameter N of binomial</param>
        /// <param name="p">Parameter p of binomial</param>
        /// <returns></returns>
        /// <remarks>Should be robust for all values of n, p</remarks>
        public static int SampleFromBinomial(IRandom r, int n, double p)
        {
            return BinoRand.Next(r, n, p);
        }

        private static class BinoRand
        {
            // n*p at which we switch algorithms
            private const int NPThresh = 10;

            private static double[] _fctab = new double[] {
                0.08106146679532726, 0.04134069595540929, 0.02767792568499834,
                0.02079067210376509, 0.01664469118982119, 0.01387612882307075,
                0.01189670994589177, 0.01041126526197209, 0.009255462182712733,
                0.008330563433362871
            };

            private const double One12 = 1.0 / 12;
            private const double One360 = 1.0 / 360;
            private const double One1260 = 1.0 / 1260;

            private static double Fc(int k)
            {
                if (k < _fctab.Length)
                    return _fctab[k];
                else
                {
                    long k1 = k + 1;
                    long k1Sq = k1 * k1;
                    return (One12 - (One360 - One1260 / k1Sq) / k1Sq) / k1;
                }
            }

            public static int Next(IRandom rand, int n, double p)
            {
                int x;
                double pin = Math.Min(p, 1 - p);
                if (n * pin < NPThresh)
                    x = InvTransform(n, pin, rand);
                else
                    x = GenerateLarge(n, pin, rand);

                if (pin != p)
                    return n - x;
                else
                    return x;
            }

            // For small n
            // Inverse transformation algorithm
            // Described in Kachitvichyanukul and Schmeiser: "Binomial Random Variate Generation"
            private static int InvTransform(int n, double p, IRandom rn)
            {
                int x = 0;
                double u = rn.NextDouble();
                double q = 1 - p;
                double s = Math.Pow(q, n);
                double r = p / q;
                double a = (n + 1) * r;

                for (;;)
                {
                    if (u <= s)
                        break;
                    u -= s;
                    x++;
                    s *= ((a / x) - r);
                }

                return x;
            }

            // For large n
            // Algorithm from W. Hormann: "The Generation of Binomial Random Variables"
            // This is algorithm BTRD
            private static int GenerateLarge(int n, double p, IRandom rn)
            {
                double np = n * p;
                double q = 1 - p;
                double npq = np * q;
                double spq = Math.Sqrt(npq);
                double b = 1.15 + 2.53 * spq;
                double vr = 0.92 - 4.2 / b;
                double urvr = 0.86 * vr;
                double a = -0.0873 + 0.0248 * b + 0.01 * p;
                double a2 = 2.0 * a;
                double c = np + 0.5;

                // don't initialize these yet, becuase we may not need them
                double alpha = double.NaN;
                double r = double.NaN;
                double nr = double.NaN;
                int m = 0;

                // especially this one: it requires a log
                double h = double.NaN;

                // Step 1
                for (;;)
                {
                    double v = rn.NextDouble();
                    double u;
                    if (v <= urvr)
                    {
                        u = v / vr - 0.43;
                        return (int)Math.Floor((a2 / (0.5 - Math.Abs(u)) + b) * u + c);
                    }

                    // Step 2
                    if (v >= vr)
                    {
                        u = rn.NextDouble() - 0.5;
                    }
                    else
                    {
                        u = v / vr - 0.93;
                        u = Math.Sign(u) * 0.5 - u;
                        v = rn.NextDouble() * vr;
                    }

                    // Step 3.0
                    double us = 0.5 - Math.Abs(u);
                    double kd = Math.Floor((a2 / us + b) * u + c);
                    if (kd < 0 || kd > n)
                        continue;
                    int k = (int)kd;

                    if (double.IsNaN(alpha))
                    {
                        alpha = (2.83 + 5.1 / b) * spq;
                        m = (int)Math.Floor((n + 1) * p);
                        r = p / q;
                        nr = (n + 1) * r;
                    }

                    v *= alpha / (a / (us * us) + b);
                    int km = Math.Abs(k - m);
                    if (km <= 15)
                    {
                        // Step 3.1
                        // Recursive evaluation of f(k)
                        double f = 1;
                        if (m < k)
                        {
                            for (int i = m + 1; i <= k; ++i)
                            {
                                f *= (nr / (double)i - r);
                            }
                        }
                        else
                        {
                            for (int i = k + 1; i <= m; ++i)
                            {
                                v *= (nr / (double)i - r);
                            }
                        }
                        if (v <= f)
                        {
                            return k;
                        }
                        else
                            continue;
                    }

                    // Step 3.2: squeeze-acceptance or rejection
                    v = Math.Log(v);
                    double rho = (km / npq) * (((km / 3.0 + 0.625) * km + 0.1666666667) / npq + 0.5);
                    double t = -0.5 * km * km / npq;
                    if (v < t - rho)
                    {
                        return k;
                    }
                    if (v > t + rho)
                        continue;

                    // Step 3.3: setup for 3.4, moved to initialization

                    // Step 3.4: final acceptance-rejection test
                    int nm = (n - m + 1);
                    if (double.IsNaN(h))
                        h = (m + 0.5) * Math.Log((m + 1) / (r * nm)) + Fc(m) + Fc(n - m);
                    int nk = n - k + 1;
                    double vval = h + (n + 1) * Math.Log((double)nm / (double)nk) + (k + 0.5) * Math.Log(nk * r / (double)(k + 1)) - Fc(k) - Fc(n - k);
                    if (v <= vval)
                    {
                        return k;
                    }
                }
            }
        }
    }
}
