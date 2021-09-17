// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Evaluation results regression algorithms (supervised learning algorithm).
    /// </summary>
    public sealed class RegressionMetrics
    {
        /// <summary>
        /// Gets the absolute loss of the model.
        /// </summary>
        /// <remarks>
        /// <format type="text/markdown"><![CDATA[
        /// The absolute loss is defined as
        /// $L1 = \frac{1}{m} \sum_{i = 1}^m | y_i - \hat{y}_i |$,
        /// where $m$ is the number of instances in the test set,
        /// $\hat{y}_i$ are the predicted labels for each instance,
        /// and $y_i$ are the correct labels of each instance.
        ///
        /// L1 loss is a non-negative, decreasing metric. Smaller values indicate a better model with respect to this metric.
        /// ]]>
        /// </format>
        /// </remarks>
        public double MeanAbsoluteError { get; }

        /// <summary>
        /// Gets the squared loss of the model.
        /// </summary>
        /// <remarks>
        /// <format type="text/markdown"><![CDATA[
        /// The squared loss is defined as
        /// $L2 = \frac{1}{m} \sum_{i = 1}^m (y_i - \hat{y}_i)^2$,
        /// where $m$ is the number of instances in the test set,
        /// \hat{y}_i are the predicted labels for each instance,
        /// and y_i are the correct labels of each instance.
        ///
        /// L2 loss is a non-negative, decreasing metric. Smaller values indicate a better model with respect to this metric.
        /// ]]>
        /// </format>
        /// </remarks>
        public double MeanSquaredError { get; }

        /// <summary>
        /// Gets the root mean square loss (or RMS) which is the square root of the L2 loss <see cref="MeanSquaredError"/>.
        /// </summary>
        public double RootMeanSquaredError { get; }

        /// <summary>
        /// Gets the result of user defined loss function.
        /// </summary>
        /// <remarks>
        /// This is the average of a loss function defined by the user,
        /// computed over all the instances in the test set.
        /// </remarks>
        public double LossFunction { get; }

        /// <summary>
        /// Gets the R-squared value of the model, which is also known as
        /// the <a href="https://en.wikipedia.org/wiki/Coefficient_of_determination">coefficient of determination​</a>.
        /// R-Squared closer to 1 indicates a better fitted model.
        /// </summary>
        public double RSquared { get; }

        internal RegressionMetrics(IExceptionContext ectx, DataViewRow overallResult)
        {
            double Fetch(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);
            MeanAbsoluteError = Fetch(RegressionEvaluator.L1);
            MeanSquaredError = Fetch(RegressionEvaluator.L2);
            RootMeanSquaredError = Fetch(RegressionEvaluator.Rms);
            LossFunction = Fetch(RegressionEvaluator.Loss);
            RSquared = Fetch(RegressionEvaluator.RSquared);
        }

        [BestFriend]
        internal RegressionMetrics(double l1, double l2, double rms, double lossFunction, double rSquared)
        {
            MeanAbsoluteError = l1;
            MeanSquaredError = l2;
            RootMeanSquaredError = rms;
            LossFunction = lossFunction;
            RSquared = rSquared;
        }
    }
}
