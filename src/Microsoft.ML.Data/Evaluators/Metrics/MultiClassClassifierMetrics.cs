// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;

namespace Microsoft.ML.Data
{
    public sealed class MultiClassClassifierMetrics
    {
        /// <summary>
        /// Gets the micro-average accuracy of the model.
        /// </summary>
        /// <remarks>
        /// The micro-average is the fraction of instances predicted correctly.
        ///
        /// The micro-average metric weighs each class according to the number of instances that belong
        /// to it in the dataset.
        /// </remarks>
        public double AccuracyMicro { get; }

        /// <summary>
        /// Gets the macro-average accuracy of the model.
        /// </summary>
        /// <remarks>
        /// The macro-average is computed by taking the average over all the classes of the fraction
        /// of correct predictions in this class (the number of correctly predicted instances in the class,
        /// divided by the total number of instances in the class).
        ///
        /// The macro-average metric gives the same weight to each class, no matter how many instances from
        /// that class the dataset contains.
        /// </remarks>
        public double AccuracyMacro { get; }

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
        /// If positive, this is the top-K for which the <see cref="TopKAccuracy"/> is calculated.
        /// </summary>
        public int TopK { get; }

        /// <summary>
        /// If <see cref="TopK"/> is positive, this is the relative number of examples where
        /// the true label is one of the top k predicted labels by the predictor.
        /// </summary>
        public double TopKAccuracy { get; }

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
        public double[] PerClassLogLoss { get; }

        internal MultiClassClassifierMetrics(IExceptionContext ectx, Row overallResult, int topK)
        {
            double FetchDouble(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);
            AccuracyMicro = FetchDouble(MultiClassClassifierEvaluator.AccuracyMicro);
            AccuracyMacro = FetchDouble(MultiClassClassifierEvaluator.AccuracyMacro);
            LogLoss = FetchDouble(MultiClassClassifierEvaluator.LogLoss);
            LogLossReduction = FetchDouble(MultiClassClassifierEvaluator.LogLossReduction);
            TopK = topK;
            if (topK > 0)
                TopKAccuracy = FetchDouble(MultiClassClassifierEvaluator.TopKAccuracy);

            var perClassLogLoss = RowCursorUtils.Fetch<VBuffer<double>>(ectx, overallResult, MultiClassClassifierEvaluator.PerClassLogLoss);
            PerClassLogLoss = new double[perClassLogLoss.Length];
            perClassLogLoss.CopyTo(PerClassLogLoss);
        }

        internal MultiClassClassifierMetrics(double accuracyMicro, double accuracyMacro, double logLoss, double logLossReduction,
            int topK, double topKAccuracy, double[] perClassLogLoss)
        {
            AccuracyMicro = accuracyMicro;
            AccuracyMacro = accuracyMacro;
            LogLoss = logLoss;
            LogLossReduction = logLossReduction;
            TopK = topK;
            TopKAccuracy = topKAccuracy;
            PerClassLogLoss = new double[perClassLogLoss.Length];
            perClassLogLoss.CopyTo(PerClassLogLoss, 0);
        }
    }
}