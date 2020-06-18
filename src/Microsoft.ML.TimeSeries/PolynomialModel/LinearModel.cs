using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// This class calculates f(x) = b0 + b1 * x
    /// </summary>
    internal sealed class LinearModel : AbstractPolynomialModel
    {
        /// <summary>
        /// Store the coefficients in member variables for better performance.
        /// </summary>
        private readonly double _b0;
        private readonly double _b1;

        public LinearModel(IReadOnlyList<double> coeffs)
            : base(coeffs)
        {
            Contracts.CheckParam(coeffs.Count == 2, nameof(coeffs), "must contain exact 2 elements.");

            _b0 = coeffs[0];
            _b1 = coeffs[1];
        }

        public override double Y(double x)
        {
            return _b0 + (_b1 * x);
        }
    }
}
