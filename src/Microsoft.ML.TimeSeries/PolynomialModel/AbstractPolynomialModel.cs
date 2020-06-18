using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// This class calculates f(x) = k0 + k1 * x + k2 * x^2 + k3 * x^3 ... kn * x^n with given x and coefs
    /// </summary>
    internal abstract class AbstractPolynomialModel
    {
        protected IReadOnlyList<double> Coeffs;

        public AbstractPolynomialModel(IReadOnlyList<double> coeffs)
        {
            Contracts.CheckValue(coeffs, nameof(coeffs));
            Coeffs = coeffs;
        }

        public abstract double Y(double x);
    }
}
