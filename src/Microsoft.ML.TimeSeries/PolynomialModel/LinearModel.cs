using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    internal class LinearModel : AbstractPolynomialModel
    {
        public LinearModel(IReadOnlyList<double> coeffs)
            : base(coeffs)
        {
            Contracts.CheckParam(coeffs.Count == 2, nameof(coeffs), "must contain exact 2 elements.");
        }

        public override double Y(double x)
        {
            return Coeffs[0] + (Coeffs[1] * x);
        }
    }
}
