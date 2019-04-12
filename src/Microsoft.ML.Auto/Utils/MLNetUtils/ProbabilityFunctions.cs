// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Auto
{
    internal static class ProbabilityFunctions
    {
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
    }
}
