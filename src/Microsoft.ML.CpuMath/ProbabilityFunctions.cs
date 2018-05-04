// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    /// <summary>
    /// Probability Functions.
    /// </summary>
    public sealed class ProbabilityFunctions
    {
        /// <summary>
        /// The approximate complimentary error function (i.e., 1-erf).
        /// </summary>
        /// <param name="x">The input parameter, of infinite range.</param>
        /// <returns>Evaluation of the function</returns>
        public static double Erfc(double x)
        {
            if (Double.IsInfinity(x))
                return Double.IsPositiveInfinity(x) ? 0 : 2.0;

            const double p = 0.3275911;
            const double a1 = 0.254829592;
            const double a2 = -0.284496736;
            const double a3 = 1.421413741;
            const double a4 = -1.453152027;
            const double a5 = 1.061405429;
            double t = 1.0 / (1.0 + p * Math.Abs(x));
            double ev = ((((((((a5 * t) + a4) * t) + a3) * t) + a2) * t + a1) * t) * Math.Exp(-(x * x));
            return x >= 0 ? ev : 2 - ev;
        }

        /// <summary>
        /// The approximate error function.
        /// </summary>
        /// <param name="x">The input parameter, of infinite range.</param>
        /// <returns>Evaluation of the function</returns>
        public static double Erf(double x)
        {
            if (Double.IsInfinity(x))
                return Double.IsPositiveInfinity(x) ? 1.0 : -1.0;

            const double p = 0.3275911;
            const double a1 = 0.254829592;
            const double a2 = -0.284496736;
            const double a3 = 1.421413741;
            const double a4 = -1.453152027;
            const double a5 = 1.061405429;
            double t = 1.0 / (1.0 + p * Math.Abs(x));
            double ev = 1.0 - ((((((((a5 * t) + a4) * t) + a3) * t) + a2) * t + a1) * t) * Math.Exp(-(x * x));
            return x >= 0 ? ev : -ev;
        }

        /// <summary>
        /// The inverse error function.
        /// </summary>
        /// <param name="x">Parameter in the range 1 to -1.</param>
        /// <returns>Evaluation of the function.</returns>
        public static double Erfinv(double x)
        {
            if (x > 1 || x < -1)
                return Double.NaN;

            if (x == 1)
                return Double.PositiveInfinity;

            if (x == -1.0)
                return Double.NegativeInfinity;

            // This is very inefficient... fortunately we only need to compute it very infrequently.
            double[] c = new double[1000];
            c[0] = 1;
            for (int k = 1; k < c.Length; ++k)
            {
                for (int m = 0; m < k; ++m)
                    c[k] += c[m] * c[k - 1 - m] / (m + 1) / (m + m + 1);
            }

            double cc = Math.Sqrt(Math.PI) / 2.0;
            double ccinc = Math.PI / 4.0;
            double zz = x;
            double zzinc = x * x;
            double ans = 0.0;
            for (int k = 0; k < c.Length; ++k)
            {
                ans += c[k] * cc * zz / (2 * k + 1);
                cc *= ccinc;
                zz *= zzinc;
            }

            return ans;
        }

        private static readonly double[] _probA = new double[] { 3.3871328727963666080e0, 1.3314166789178437745e+2, 1.9715909503065514427e+3, 1.3731693765509461125e+4,
                4.5921953931549871457e+4, 6.7265770927008700853e+4, 3.3430575583588128105e+4, 2.5090809287301226727e+3 };
        private static readonly double[] _probB = new double[] { 4.2313330701600911252e+1, 6.8718700749205790830e+2, 5.3941960214247511077e+3, 2.1213794301586595867e+4,
                3.9307895800092710610e+4, 2.8729085735721942674e+4, 5.2264952788528545610e+3 };

        private static readonly double[] _probC = new double[] { 1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0, 3.64784832476320460504e0,
                1.27045825245236838258e0, 2.41780725177450611770e-1, 2.27238449892691845833e-2, 7.74545014278341407640e-4 };
        private static readonly double[] _probD = new double[] { 2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1, 1.48103976427480074590e-1,
                1.51986665636164571966e-2, 5.47593808499534494600e-4, 1.05075007164441684324e-9 };

        private static readonly double[] _probE = new double[] { 6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0, 2.96560571828504891230e-1,
                2.65321895265761230930e-2, 1.24266094738807843860e-3, 2.71155556874348757815e-5, 2.01033439929228813265e-7 };
        private static readonly double[] _probF = new double[] { 5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2, 7.86869131145613259100e-4,
                1.84631831751005468180e-5, 1.42151175831644588870e-7, 2.04426310338993978564e-15 };

        /// <summary>
        /// The probit function. This has many applications, the most familiar being perhaps
        /// that this is the point "x" at which the standard normal CDF evaluates to the indicated
        /// p value. It is used in establishing confidence intervals.
        /// </summary>
        /// <param name="p">The input p value, so in the range 0 to 1.</param>
        /// <returns>One intepretation is, the value at which the standard normal CDF evaluates to p.</returns>
        public static double Probit(double p)
        {
            double q = p - 0.5;
            double r = 0.0;
            if (Math.Abs(q) <= 0.425)
            {
                // Input value is close-ish to 0.5 (0.075 to 0.925)
                r = 0.180625 - q * q;
                return q * (((((((_probA[7] * r + _probA[6]) * r + _probA[5]) * r + _probA[4]) * r + _probA[3]) * r + _probA[2]) * r + _probA[1]) * r + _probA[0]) /
                    (((((((_probB[6] * r + _probB[5]) * r + _probB[4]) * r + _probB[3]) * r + _probB[2]) * r + _probB[1]) * r + _probB[0]) * r + 1.0);
            }
            else
            {
                if (q < 0)
                    r = p;
                else
                    r = 1 - p;

                Contracts.CheckParam(r >= 0, nameof(p), "Illegal input value");

                r = Math.Sqrt(-Math.Log(r));
                double retval = 0.0;
                if (r < 5)
                {
                    r = r - 1.6;
                    retval = (((((((_probC[7] * r + _probC[6]) * r + _probC[5]) * r + _probC[4]) * r + _probC[3]) * r + _probC[2]) * r + _probC[1]) * r + _probC[0]) /
                        (((((((_probD[6] * r + _probD[5]) * r + _probD[4]) * r + _probD[3]) * r + _probD[2]) * r + _probD[1]) * r + _probD[0]) * r + 1.0);
                }
                else
                {
                    r = r - 5;
                    retval = (((((((_probE[7] * r + _probE[6]) * r + _probE[5]) * r + _probE[4]) * r + _probE[3]) * r + _probE[2]) * r + _probE[1]) * r + _probE[0]) /
                        (((((((_probF[6] * r + _probF[5]) * r + _probF[4]) * r + _probF[3]) * r + _probF[2]) * r + _probF[1]) * r + _probF[0]) * r + 1.0);
                }
                return q >= 0 ? retval : -retval;
            }
        }
    }
}
