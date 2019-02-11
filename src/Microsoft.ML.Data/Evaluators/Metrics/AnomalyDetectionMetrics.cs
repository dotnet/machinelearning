// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Data.DataView;

namespace Microsoft.ML.Data.Evaluators.Metrics
{
    /// <summary>
    /// Evaluation results for anomaly detection.
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
        public double Auc { get; }

        /// <summary>
        /// Detection rate at k false positives.
        /// </summary>
        public double DrAtK { get; }
        /// <summary>
        /// Detection rate at fraction p false positives.
        /// </summary>
        public double DrAtPFpr { get; }
        /// <summary>
        /// Detection rate at number of anomalies.
        /// </summary>
        public double DrAtNumPos { get; }

        internal AnomalyDetectionMetrics(IExceptionContext ectx, Row overallResult)
        {
            double FetchDouble(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);

            Auc = FetchDouble(BinaryClassifierEvaluator.Auc);
            DrAtK = FetchDouble(AnomalyDetectionEvaluator.OverallMetrics.DrAtK);
            DrAtPFpr = FetchDouble(AnomalyDetectionEvaluator.OverallMetrics.DrAtPFpr);
            DrAtNumPos = FetchDouble(AnomalyDetectionEvaluator.OverallMetrics.DrAtNumPos);
        }
    }
}
