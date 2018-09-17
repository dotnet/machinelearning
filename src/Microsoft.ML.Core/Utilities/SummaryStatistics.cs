// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    public abstract class SummaryStatisticsBase
    {
        // Sum of squared difference from the current mean.
        protected double M2;

        internal SummaryStatisticsBase()
        {
            Max = double.NegativeInfinity;
            Min = double.PositiveInfinity;
        }

        public override bool Equals(object obj)
        {
            SummaryStatisticsBase s = obj as SummaryStatisticsBase;
            if (s == null)
                return false;
            return s.RawCount == RawCount && s.Count == Count &&
                s.Mean == Mean && s.M2 == M2 && s.Min == Min
                && s.Max == Max && s.NonzeroCount == NonzeroCount;
        }

        public override int GetHashCode()
        {
            return Hashing.CombinedHash(RawCount.GetHashCode(),
                Count, Mean, M2, Min, Max, NonzeroCount);
        }

        /// <summary>
        /// The running (unweighted) count of elements added to this object.
        /// </summary>
        public long RawCount { get; private set; }

        /// <summary>
        /// The running (weighted) count of elements.
        /// </summary>
        public double Count { get; private set; }

        /// <summary>
        /// The running count of non-zero elements.
        /// </summary>
        public double NonzeroCount { get; private set; }

        /// <summary>
        /// The running weight of non-zero elements.
        /// </summary>
        public double NonzeroWeight { get; private set; }

        /// <summary>
        /// The running arithmetic mean.
        /// </summary>
        public double Mean { get; private set; }

        /// <summary>
        /// Thes sample variance.
        /// </summary>
        public double SampleVariance => (M2 / Count) * RawCount / (RawCount - 1);

        /// <summary>
        ///  The sample standard deviation.
        /// </summary>
        public double SampleStdDev => Math.Sqrt(SampleVariance);

        /// <summary>
        /// Returns the standard error of the mean.
        /// </summary>
        public double StandardErrorMean => SampleStdDev / Math.Sqrt(RawCount);

        /// <summary>
        /// The maximum.
        /// </summary>
        public double Max { get; private set; }

        /// <summary>
        /// The minimum.
        /// </summary>
        public double Min { get; private set; }

        public override string ToString()
        {
            return string.Format(@"Stats[Count={0}, Mean={1}, StdDev={2}]",
                Count, Mean, SampleStdDev);
        }

        /// <summary>
        /// Accumulates one more value, optionally weighted.
        /// This accumulation procedure is based on the following,
        /// with adjustments as appropriate for weighted instances:
        /// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        /// </summary>
        /// <param name="v">The value</param>
        /// <param name="w">The weight given to this value</param>
        /// <param name="c">Amount of appereance of this value</param>
        public virtual void Add(double v, double w = 1.0, long c = 1)
        {
            double temp = w + Count;
            double delta = v - Mean;
            double deltaN = delta / temp;
            double deltaN2 = deltaN * deltaN;
            double r = delta * w / temp;
            double term1 = w * delta * deltaN * Count;

            M2 += Count * delta * r;
            Mean += r;
            Count = temp;
            RawCount = RawCount + c;
            if (v != 0.0)
            {
                NonzeroCount += c;
                NonzeroWeight += w;
            }

            if (Max < v)
                Max = v;
            if (Min > v)
                Min = v;
        }

        /// <summary>
        /// Adds a stats object with another type of stats object. The result
        /// should be equivalent, up to the effects of numerical imprecision,
        /// as if in addition to all the values this object has seen, it has
        /// also seen the values added to the other object.
        /// </summary>
        /// <param name="s"></param>
        public void Add(SummaryStatisticsBase s)
        {
            double delta = s.Mean - Mean;
            double na = Count;
            double nb = s.Count;
            double nx = na + nb;
            if (nb == 0.0)
                return; // No-op.
            M2 += s.M2 + delta * delta * na * nb / nx;
            Count += s.Count;
            Mean += delta * nb / nx;
            RawCount += s.RawCount;
            NonzeroCount += s.NonzeroCount;
            NonzeroWeight += s.NonzeroWeight;

            if (Max < s.Max)
                Max = s.Max;
            if (Min > s.Min)
                Min = s.Min;
        }
    }

    public sealed class SummaryStatisticsUpToSecondOrderMoments : SummaryStatisticsBase
    {
        /// <summary>
        /// A convenient way to combine the observations of two Stats objects
        /// into a new Stats object.
        /// </summary>
        /// <param name="a">The first operand</param>
        /// <param name="b">The second operand</param>
        /// <returns></returns>
        public static SummaryStatisticsUpToSecondOrderMoments operator +(SummaryStatisticsUpToSecondOrderMoments a, SummaryStatisticsUpToSecondOrderMoments b)
        {
            SummaryStatisticsUpToSecondOrderMoments result = new SummaryStatisticsUpToSecondOrderMoments();
            result.Add(a);
            result.Add(b);
            return result;
        }
    }

    /// <summary>
    /// A class for one-pass accumulation of weighted summary statistics, up
    /// to the fourth moment. The accumulative algorithms used here may be
    /// reviewed at
    /// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    /// All quantities are weighted, except for <c>RawCount</c>.
    /// </summary>
    public sealed class SummaryStatistics : SummaryStatisticsBase
    {
        // Sum of cubed difference from the current mean.
        private double _m3;
        // Sum of (tessaracted?) difference from the current mean.
        private double _m4;

        public SummaryStatistics()
            : base()
        {
        }

        public override bool Equals(object obj)
        {
            if (!base.Equals(obj))
                return false;
            SummaryStatistics s = obj as SummaryStatistics;
            if (s == null)
                return false;
            return s._m3 == _m3 && s._m4 == _m4;
        }

        public override int GetHashCode()
        {
            return Hashing.CombinedHash<double>(base.GetHashCode(),
                _m3, _m4);
        }

        /// <summary>
        /// The running count of non-zero elements.
        /// </summary>
        public double Nonzero => NonzeroCount;

        /// <summary>
        /// The sample skewness.
        /// </summary>
        public double Skewness => Math.Sqrt(Count) * _m3 / Math.Pow(M2, 1.5);

        /// <summary>
        /// The sample kurtosis.
        /// </summary>
        public double Kurtosis => Count * _m4 / (M2 * M2) - 3;

        /// <summary>
        /// Z-test value for a normality test based on the skew.
        /// Under the null hypothesis of normality this quantity will
        /// itself be approximately normally distributed.
        /// </summary>
        public double SkewnessZ
        {
            get
            {
                double b1 = Skewness; // Actually Sqrt(B1)...
                // REVIEW: Should n be based on the weighted or unweighted count?
                // My first thought was unweighted, but then my thought was, the Z score
                // is not a p-value, and when transforming we'd incorporate information on
                // the degrees of freedom, but not before.
                double n = Count;

                double y = b1 * Math.Sqrt((n + 1) * (n + 3) / (6 * (n - 2)));
                double beta = 3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9));
                double w2 = -1.0 + Math.Sqrt(2.0 * (beta - 1.0));
                double delta = 1.0 / Math.Sqrt(Math.Log(Math.Sqrt(w2)));
                double alpha = Math.Sqrt(2.0 / (w2 - 1.0));
                double yDivAlpha = y / alpha;
                double z1 = delta * Math.Log(yDivAlpha + Math.Sqrt(yDivAlpha * yDivAlpha + 1.0));

                return z1;
            }
        }

        /// <summary>
        /// Z-test value for a normality test based on the kurtosis.
        /// Under the null hypothesis of normality this quantity will
        /// itself be approximately normally distributed.
        /// </summary>
        public double KurtosisZ
        {
            get
            {
                double b2 = Kurtosis;
                if (double.IsNaN(b2))
                {
                    // This can happen if the values fed in are all the same.
                    return double.NaN;
                }
                double n = Count;

                double e = -6 / (n + 1);
                double var = 24 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1) * (n + 3) * (n + 5));
                if (var == 0)
                    return 0.0;
                double x = (b2 - e) / Math.Sqrt(var); // Standardized b2.

                double ibeta = (n + 7) * (n + 9) / (6 * (n * n - 5 * n + 2)) *
                    Math.Sqrt(n * (n - 2) * (n - 3) / (6 * (n + 3) * (n + 5)));
                double a = 6 + 8 * ibeta * (2 * ibeta + Math.Sqrt(1 + 4 * ibeta * ibeta));
                double toCubeRoot = (1 - 2 / a) / (1 + x * Math.Sqrt(2 / (a - 4)));
                double z2 = (1 - 2 / (9 * a) - Math.Sign(toCubeRoot) * Math.Pow(Math.Abs(toCubeRoot), 1.0 / 3.0)) /
                    Math.Sqrt(2 / (9 * a));

                return z2;
            }
        }

        /// <summary>
        /// Omnibus K2 unifying the skew and kurtosis Z-tests. Under the
        /// null hypothesis of normality this quantity will be approximately
        /// chi-squared distributed.
        ///
        /// D'Agostino, Ralph B.; Albert Belanger; Ralph B. D'Agostino, Jr (1990). "A suggestion
        /// for using powerful and informative tests of normality". The American Statistician 44
        /// (4): 316–321. JSTOR 2684359.
        /// </summary>
        public double OmnibusK2
        {
            get
            {
                double z1 = SkewnessZ;
                double z2 = KurtosisZ;
                return z1 * z1 + z2 * z2;
            }
        }

        public override string ToString()
        {
            return string.Format(@"Stats[Count={0}, Mean={1}, StdDev={2}, Skew={3}, Kurt={4}, SkewZ={5}, KurtZ={6}, OmniK={7}]",
                Count, Mean, SampleStdDev, Skewness, Kurtosis, SkewnessZ, KurtosisZ, OmnibusK2);
        }

        /// <summary>
        /// Accumulates one more value, optionally weighted.
        /// This accumulation procedure is based on the following,
        /// with adjustments as appropriate for weighted instances:
        /// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        /// </summary>
        /// <param name="v">The value</param>
        /// <param name="w">The weight given to this value</param>
        /// <param name="c">Amount of appereance of this value</param>
        public override void Add(double v, double w = 1.0, long c = 1)
        {
            double temp = w + Count;
            double delta = v - Mean;
            double deltaN = delta / temp;
            double deltaN2 = deltaN * deltaN;
            double r = delta * w / temp;
            double term1 = w * delta * deltaN * Count;

            _m4 += term1 * deltaN2 * (Count * Count - Count * w + w * w)
                + 6 * deltaN2 * w * w * M2
                - 4 * deltaN * w * _m3;
            _m3 += term1 * deltaN * (Count - w) - 3 * w * deltaN * M2;

            base.Add(v, w, c);
        }

        /// <summary>
        /// Adds a stats object with another type of stats object. The result
        /// should be equivalent, up to the effects of numerical imprecision,
        /// as if in addition to all the values this object has seen, it has
        /// also seen the values added to the other object.
        /// </summary>
        /// <param name="s"></param>
        public void Add(SummaryStatistics s)
        {
            double delta = s.Mean - Mean;
            double na = Count;
            double nb = s.Count;
            double nx = na + nb;
            if (nb == 0.0)
                return; // No-op.
            _m4 += s._m4 + delta *
                (delta * (delta * delta * na * nb * (na * na - na * nb + nb * nb) / nx +
                6 * (na * na * s.M2 + nb * nb * M2)) / nx +
                4 * (na * s._m3 - nb * _m3)) / nx;
            _m3 += s._m3 + delta * (delta * delta * na * nb * (na - nb) / nx + 3 * (na * s.M2 - nb * M2)) / nx;

            base.Add(s);
        }

        /// <summary>
        /// A convenient way to combine the observations of two Stats objects
        /// into a new Stats object.
        /// </summary>
        /// <param name="a">The first operand</param>
        /// <param name="b">The second operand</param>
        /// <returns></returns>
        public static SummaryStatistics operator +(SummaryStatistics a, SummaryStatistics b)
        {
            SummaryStatistics result = new SummaryStatistics();
            result.Add(a);
            result.Add(b);
            return result;
        }
    }
}
