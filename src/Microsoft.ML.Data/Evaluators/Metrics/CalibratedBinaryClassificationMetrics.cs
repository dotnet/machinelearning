// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Evaluation results for binary classifiers, including probabilistic metrics.
    /// </summary>
    public sealed class CalibratedBinaryClassificationMetrics : BinaryClassificationMetrics
    {
        /// <summary>
        /// Gets the log-loss of the classifier.
        /// </summary>
        /// <remarks>
        /// The log-loss metric, is computed as follows:
        /// LL = - (1/m) * sum( log(p[i]))
        /// where m is the number of instances in the test set.
        /// p[i] is the probability returned by the classifier if the instance belongs to class 1,
        /// and 1 minus the probability returned by the classifier if the instance belongs to class 0.
        /// </remarks>
        public double LogLoss { get; }

        /// <summary>
        /// Gets the log-loss reduction (also known as relative log-loss, or reduction in information gain - RIG)
        /// of the classifier.
        /// </summary>
        /// <remarks>
        /// The log-loss reduction is scaled relative to a classifier that predicts the prior for every example:
        /// (LL(prior) - LL(classifier)) / LL(prior)
        /// This metric can be interpreted as the advantage of the classifier over a random prediction.
        /// For example, if the RIG equals 20, it can be interpreted as &quot;the probability of a correct prediction is
        /// 20% better than random guessing.&quot;
        /// </remarks>
        public double LogLossReduction { get; }

        /// <summary>
        /// Gets the test-set entropy (prior Log-Loss/instance) of the classifier.
        /// </summary>
        public double Entropy { get; }

        internal CalibratedBinaryClassificationMetrics(IExceptionContext ectx, DataViewRow overallResult)
            : base(ectx, overallResult)
        {
            double Fetch(string name) => Fetch<double>(ectx, overallResult, name);
            LogLoss = Fetch(BinaryClassifierEvaluator.LogLoss);
            LogLossReduction = Fetch(BinaryClassifierEvaluator.LogLossReduction);
            Entropy = Fetch(BinaryClassifierEvaluator.Entropy);
        }
    }
}