// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Evaluation results for anomaly detection(unsupervised learning algorithm).
    /// </summary>
    public sealed class AnomalyDetectionMetrics
    {
        /// <summary>
        /// Gets the area under the ROC curve.
        /// </summary>
        /// <remarks>
        /// The area under the ROC curve is equal to the probability that the algorithm ranks
        /// a randomly chosen positive instance higher than a randomly chosen negative one
        /// (assuming 'positive' ranks higher than 'negative').
        /// </remarks>
        public double AreaUnderRocCurve { get; }

        /// <summary>
        /// Detection rate at K false positives.
        /// </summary>
        /// <remarks>
        /// This is computed as follows:
        /// 1.Sort the test examples by the output of the anomaly detector in descending order of scores.
        /// 2.Among the top K False Positives,  compute ratio :  (True Positive @ K)  / (Total anomalies in test data)
        /// Example confusion matrix for anomaly detection:
        ///                            Anomalies (in test data)  | Non-Anomalies (in test data)
        ///  Predicted Anomalies     :         TP                |           FP
        ///  Predicted Non-Anomalies :         FN                |           TN
        ///  </remarks>
        public double DetectionRateAtKFalsePositives { get; }

        internal AnomalyDetectionMetrics(IExceptionContext ectx, DataViewRow overallResult)
        {
            double FetchDouble(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);
            AreaUnderRocCurve = FetchDouble(BinaryClassifierEvaluator.Auc);
            DetectionRateAtKFalsePositives = FetchDouble(AnomalyDetectionEvaluator.OverallMetrics.DrAtK);
        }
    }
}
