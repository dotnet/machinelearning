// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data
{
    public sealed class RegressionMetrics
    {
        /// <summary>
        /// Gets the absolute loss of the model.
        /// </summary>
        /// <remarks>
        /// The absolute loss is defined as
        /// L1 = (1/m) * sum( abs( yi - y&apos;i))
        /// where m is the number of instances in the test set.
        /// y'i are the predicted labels for each instance.
        /// yi are the correct labels of each instance.
        /// </remarks>
        public double L1 { get; }

        /// <summary>
        /// Gets the squared loss of the model.
        /// </summary>
        /// <remarks>
        /// The squared loss is defined as
        /// L2 = (1/m) * sum(( yi - y&apos;i)^2)
        /// where m is the number of instances in the test set.
        /// y'i are the predicted labels for each instance.
        /// yi are the correct labels of each instance.
        /// </remarks>
        public double L2 { get; }

        /// <summary>
        /// Gets the root mean square loss (or RMS) which is the square root of the L2 loss.
        /// </summary>
        public double Rms { get; }

        /// <summary>
        /// Gets the result of user defined loss function.
        /// </summary>
        /// <remarks>
        /// This is the average of a loss function defined by the user,
        /// computed over all the instances in the test set.
        /// </remarks>
        public double LossFn { get; }

        /// <summary>
        /// Gets the R squared value of the model, which is also known as
        /// the coefficient of determination​.
        /// </summary>
        public double RSquared { get; }

        internal RegressionMetrics(IExceptionContext ectx, IRow overallResult)
        {
            double Fetch(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);
            L1 = Fetch(RegressionEvaluator.L1);
            L2 = Fetch(RegressionEvaluator.L2);
            Rms = Fetch(RegressionEvaluator.Rms);
            LossFn = Fetch(RegressionEvaluator.Loss);
            RSquared = Fetch(RegressionEvaluator.RSquared);
        }

        [BestFriend]
        internal RegressionMetrics(double l1, double l2, double rms, double lossFunction, double rSquared)
        {
            L1 = l1;
            L2 = l2;
            Rms = rms;
            LossFn = lossFunction;
            RSquared = rSquared;
        }
    }
}