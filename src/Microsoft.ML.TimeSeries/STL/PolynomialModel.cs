// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
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

    /// <summary>
    /// A general polynomial model
    /// </summary>
    internal sealed class PolynomialModel : AbstractPolynomialModel
    {
        public PolynomialModel(IReadOnlyList<double> coeffs)
            : base(coeffs)
        {
        }

        /// <summary>
        /// This function calculates the y value by given the x value, under this model
        /// </summary>
        /// <param name="x">The specific x value</param>
        public override double Y(double x)
        {
            double result = Coeffs[0];
            double p = 1.0;
            for (int i = 1; i < Coeffs.Count; i++)
            {
                p *= x;
                result += Coeffs[i] * p;
            }
            return result;
        }
    }

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
