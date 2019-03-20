// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Collections.Immutable;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Evaluation results for multi-class classification trainers.
    /// </summary>
    public sealed class MulticlassClassificationMetrics
    {
        /// <summary>
        /// Gets the average log-loss of the classifier.
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
        /// For example, if the RIG equals 20, it can be interpreted as "the probability of a correct prediction is
        /// 20% better than random guessing".
        /// </remarks>
        public double LogLossReduction { get; private set; }

        /// <summary>
        /// Gets the macro-average accuracy of the model.
        /// </summary>
        /// <remarks>
        /// The macro-average is the average accuracy at the class level. The accuracy for each class is computed
        /// and the macro-accuracy is the average of these accuracies.
        ///
        /// The macro-average metric gives the same weight to each class, no matter how many instances from
        /// that class the dataset contains.
        /// </remarks>
        public double MacroAccuracy { get; }

        /// <summary>
        /// Gets the micro-average accuracy of the model.
        /// </summary>
        /// <remarks>
        /// The micro-average is the fraction of instances predicted correctly.
        /// The micro-average does not take class membership into account.
        /// </remarks>
        public double MicroAccuracy { get; }

        /// <summary>
        /// If <see cref="TopK"/> is positive, this is the relative number of examples where
        /// the true label is one of the top-k predicted labels by the predictor.
        /// </summary>
        public double TopKAccuracy { get; }

        /// <summary>
        /// If positive, this is the top-K for which the <see cref="TopKAccuracy"/> is calculated.
        /// </summary>
        public int TopK { get; }

        /// <summary>
        /// Gets the log-loss of the classifier for each class.
        /// </summary>
        /// <remarks>
        /// The log-loss metric, is computed as follows:
        /// LL = - (1/m) * sum( log(p[i]))
        /// where m is the number of instances in the test set.
        /// p[i] is the probability returned by the classifier if the instance belongs to the class,
        /// and 1 minus the probability returned by the classifier if the instance does not belong to the class.
        /// </remarks>
        public IReadOnlyList<double> PerClassLogLoss { get; }

        internal MulticlassClassificationMetrics(IExceptionContext ectx, DataViewRow overallResult, int topK)
        {
            double FetchDouble(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);
            MicroAccuracy = FetchDouble(MulticlassClassificationEvaluator.AccuracyMicro);
            MacroAccuracy = FetchDouble(MulticlassClassificationEvaluator.AccuracyMacro);
            LogLoss = FetchDouble(MulticlassClassificationEvaluator.LogLoss);
            LogLossReduction = FetchDouble(MulticlassClassificationEvaluator.LogLossReduction);
            TopK = topK;
            if (topK > 0)
                TopKAccuracy = FetchDouble(MulticlassClassificationEvaluator.TopKAccuracy);

            var perClassLogLoss = RowCursorUtils.Fetch<VBuffer<double>>(ectx, overallResult, MulticlassClassificationEvaluator.PerClassLogLoss);
            PerClassLogLoss = perClassLogLoss.DenseValues().ToImmutableArray();
        }

        internal MulticlassClassificationMetrics(double accuracyMicro, double accuracyMacro, double logLoss, double logLossReduction,
            int topK, double topKAccuracy, double[] perClassLogLoss)
        {
            MicroAccuracy = accuracyMicro;
            MacroAccuracy = accuracyMacro;
            LogLoss = logLoss;
            LogLossReduction = logLossReduction;
            TopK = topK;
            TopKAccuracy = topKAccuracy;
            PerClassLogLoss = perClassLogLoss.ToImmutableArray();
        }
    }
}