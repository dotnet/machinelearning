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
        /// Gets the log-loss of the classifier. Log-loss measures the performance of a classifier
        /// with respect to how much the predicted probabilities diverge from the true class label. Lower
        /// log-loss indicates a better model. A perfect model, which predicts a probability of 1 for the
        /// true class, will have a log-loss of 0.
        /// </summary>
        /// <remarks>
        /// <format type="text/markdown"><![CDATA[
        /// The log-loss metric, is computed as follows:
        /// $LogLoss = - \frac{1}{m} \sum{i = 1}^m ln(p_i)$
        /// where m is the number of instances in the test set and
        /// $p_i$ is the probability returned by the classifier if the instance belongs to class 1,
        /// and 1 minus the probability returned by the classifier if the instance belongs to class 0.
        /// ]]>
        /// </format>
        /// </remarks>
        public double LogLoss { get; }

        /// <summary>
        /// Gets the log-loss reduction (also known as relative log-loss, or reduction in information gain - RIG)
        /// of the classifier. It gives a measure of how much a model improves on a model that gives random predictions.
        /// Log-loss reduction closer to 1 indicates a better model.
        /// </summary>
        /// <remarks>
        /// <format type="text/markdown"><![CDATA[
        /// The log-loss reduction is scaled relative to a classifier that predicts the prior for every example:
        /// $LogLossReduction = \frac{LogLoss(prior) - LogLoss(classifier)}{LogLoss(prior)}$
        /// This metric can be interpreted as the advantage of the classifier over a random prediction.
        /// For example, if the RIG equals 0.2, it can be interpreted as "the probability of a correct prediction is
        /// 20% better than random guessing".
        /// ]]>
        /// </format>
        /// </remarks>
        public double LogLossReduction { get; }

        /// <summary>
        /// Gets the test-set entropy, which is the prior log-loss based on the proportion of positive
        /// and negative instances in the test set. A classifier's <see cref="LogLoss"/> lower than
        /// the entropy indicates that a classifier does better than predicting the proportion of positive
        /// instances as the probability for each instance.
        /// </summary>
        /// <remarks>
        /// <format type="text/markdown"><![CDATA[
        /// $Entropy = -p log_2(p) - (1 - p) log_2(1 - p)$, where $p$ is the proportion of the positive class
        /// in the test set.
        /// ]]>
        /// </format>
        /// </remarks>
        public double Entropy { get; }

        internal CalibratedBinaryClassificationMetrics(IHost host, DataViewRow overallResult, IDataView confusionMatrix)
            : base(host, overallResult, confusionMatrix)
        {
            double Fetch(string name) => Fetch<double>(host, overallResult, name);
            LogLoss = Fetch(BinaryClassifierEvaluator.LogLoss);
            LogLossReduction = Fetch(BinaryClassifierEvaluator.LogLossReduction);
            Entropy = Fetch(BinaryClassifierEvaluator.Entropy);
        }
    }
}
