// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using static Microsoft.ML.Runtime.Data.MetricKinds;

namespace Microsoft.ML.Models
{
    /// <summary>
    /// This class contains the overall metrics computed by binary classification evaluators.
    /// </summary>
    public sealed class BinaryClassificationMetrics
    {
        private BinaryClassificationMetrics()
        {
        }

        internal static List<BinaryClassificationMetrics> FromMetrics(IHostEnvironment env, IDataView overallMetrics, IDataView confusionMatrix, int confusionMatriceStartIndex = 0)
        {
            Contracts.AssertValue(env);
            env.AssertValue(overallMetrics);
            env.AssertValue(confusionMatrix);

            var metricsEnumerable = overallMetrics.AsEnumerable<SerializationClass>(env, true, ignoreMissingColumns: true);
            if (!metricsEnumerable.GetEnumerator().MoveNext())
            {
                throw env.Except("The overall RegressionMetrics didn't have any rows.");
            }

            List<BinaryClassificationMetrics> metrics = new List<BinaryClassificationMetrics>();
            var confusionMatrices = ConfusionMatrix.Create(env, confusionMatrix).GetEnumerator();

            int index = 0;
            foreach (var metric in metricsEnumerable)
            {

                if (index++ >= confusionMatriceStartIndex && !confusionMatrices.MoveNext())
                {
                    throw env.Except("Confusion matrices didn't have enough matrices.");
                }

                metrics.Add(
                    new BinaryClassificationMetrics()
                    {
                        Auc = metric.Auc,
                        Accuracy = metric.Accuracy,
                        PositivePrecision = metric.PositivePrecision,
                        PositiveRecall = metric.PositiveRecall,
                        NegativePrecision = metric.NegativePrecision,
                        NegativeRecall = metric.NegativeRecall,
                        LogLoss = metric.LogLoss,
                        LogLossReduction = metric.LogLossReduction,
                        Entropy = metric.Entropy,
                        F1Score = metric.F1Score,
                        Auprc = metric.Auprc,
                        RowTag = metric.RowTag,
                        ConfusionMatrix = confusionMatrices.Current,
                    });

            }

            return metrics;
        }

        /// <summary>
        /// Gets the area under the ROC curve.
        /// </summary>
        /// <remarks>
        /// The area under the ROC curve is equal to the probability that the classifier ranks
        /// a randomly chosen positive instance higher than a randomly chosen negative one
        /// (assuming 'positive' ranks higher than 'negative').
        /// </remarks>
        public double Auc { get; private set; }

        /// <summary>
        /// Gets the accuracy of a classifier which is the proportion of correct predictions in the test set.
        /// </summary>
        public double Accuracy { get; private set; }

        /// <summary>
        /// Gets the positive precision of a classifier which is the proportion of correctly predicted
        /// positive instances among all the positive predictions (i.e., the number of positive instances
        /// predicted as positive, divided by the total number of instances predicted as positive).
        /// </summary>
        public double PositivePrecision { get; private set; }

        /// <summary>
        /// Gets the positive recall of a classifier which is the proportion of correctly predicted
        /// positive instances among all the positive instances (i.e., the number of positive instances
        /// predicted as positive, divided by the total number of positive instances).
        /// </summary>
        public double PositiveRecall { get; private set; }

        /// <summary>
        /// Gets the negative precision of a classifier which is the proportion of correctly predicted
        /// negative instances among all the negative predictions (i.e., the number of negative instances
        /// predicted as negative, divided by the total number of instances predicted as negative).
        /// </summary>
        public double NegativePrecision { get; private set; }

        /// <summary>
        /// Gets the negative recall of a classifier which is the proportion of correctly predicted
        /// negative instances among all the negative instances (i.e., the number of negative instances
        /// predicted as negative, divided by the total number of negative instances).
        /// </summary>
        public double NegativeRecall { get; private set; }

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
        public double LogLoss { get; private set; }

        /// <summary>
        /// Gets the log-loss reduction (also known as relative log-loss, or reduction in information gain - RIG)
        /// of the classifier.
        /// </summary>
        /// <remarks>
        /// The log-loss reduction is scaled relative to a classifier that predicts the prior for every example:
        /// (LL(prior) - LL(classifier)) / LL(prior)
        /// This metric can be interpreted as the advantage of the classifier over a random prediction.
        /// E.g., if the RIG equals 20, it can be interpreted as "the probability of a correct prediction is
        /// 20% better than random guessing".
        /// </remarks>
        public double LogLossReduction { get; private set; }

        /// <summary>
        /// Gets the test-set entropy (prior Log-Loss/instance) of the classifier.
        /// </summary>
        public double Entropy { get; private set; }

        /// <summary>
        /// Gets the F1 score of the classifier.
        /// </summary>
        /// <remarks>
        /// F1 score is the harmonic mean of precision and recall: 2 * precision * recall / (precision + recall).
        /// </remarks>
        public double F1Score { get; private set; }

        /// <summary>
        /// Gets the area under the precision/recall curve of the classifier.
        /// </summary>
        /// <remarks>
        /// The area under the precision/recall curve is a single number summary of the information in the
        /// precision/recall curve. It is increasingly used in the machine learning community, particularly
        /// for imbalanced datasets where one class is observed more frequently than the other. On these
        /// datasets, AUPRC can highlight performance differences that are lost with AUC.
        /// </remarks>
        public double Auprc { get; private set; }

        /// <summary>
        /// Gets the confusion matrix, or error matrix, of the classifier.
        /// </summary>
        public ConfusionMatrix ConfusionMatrix { get; private set; }

        /// <summary>
        /// For cross-validation, this is equal to "Fold N" for per-fold metric rows, "Overall" for the average metrics and "STD" for standard deviation.
        /// For non-CV scenarios, this is equal to null
        /// </summary>
        public string RowTag { get; private set; }

        /// <summary>
        /// This class contains the public fields necessary to deserialize from IDataView.
        /// </summary>
        private sealed class SerializationClass
        {
#pragma warning disable 649 // never assigned
            [ColumnName(BinaryClassifierEvaluator.Auc)]
            public Double Auc;

            [ColumnName(BinaryClassifierEvaluator.Accuracy)]
            public Double Accuracy;

            [ColumnName(BinaryClassifierEvaluator.PosPrecName)]
            public Double PositivePrecision;

            [ColumnName(BinaryClassifierEvaluator.PosRecallName)]
            public Double PositiveRecall;

            [ColumnName(BinaryClassifierEvaluator.NegPrecName)]
            public Double NegativePrecision;

            [ColumnName(BinaryClassifierEvaluator.NegRecallName)]
            public Double NegativeRecall;

            [ColumnName(BinaryClassifierEvaluator.LogLoss)]
            public Double LogLoss;

            [ColumnName(BinaryClassifierEvaluator.LogLossReduction)]
            public Double LogLossReduction;

            [ColumnName(BinaryClassifierEvaluator.Entropy)]
            public Double Entropy;

            [ColumnName(BinaryClassifierEvaluator.F1)]
            public Double F1Score;

            [ColumnName(BinaryClassifierEvaluator.AuPrc)]
            public Double Auprc;

            [ColumnName(ColumnNames.FoldIndex)]
            public string RowTag;
#pragma warning restore 649 // never assigned
        }
    }
}
