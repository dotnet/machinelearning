// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Data.DataView;

namespace Microsoft.ML.Data.Evaluators.Metrics
{
    public sealed class AnomalyDetectionMetrics
    {
        public double Auc { get; }
        public double DrAtK { get; }
        public double DrAtPFpr { get; }
        public double DrAtNumPos { get; }
        public double NumAnomalies { get; }
        public double ThreshAtK { get; }
        public double ThreshAtP { get; }
        public double ThreshAtNumPos { get; }

        internal AnomalyDetectionMetrics(IExceptionContext ectx, Row overallResult)
        {
            long FetchInt(string name) => RowCursorUtils.Fetch<long>(ectx, overallResult, name);
            float FetchFloat(string name) => RowCursorUtils.Fetch<float>(ectx, overallResult, name);
            double FetchDouble(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);

            Auc = FetchDouble(BinaryClassifierEvaluator.Auc);
            DrAtK = FetchDouble(AnomalyDetectionEvaluator.OverallMetrics.DrAtK);
            DrAtPFpr = FetchDouble(AnomalyDetectionEvaluator.OverallMetrics.DrAtPFpr);
            DrAtNumPos = FetchDouble(AnomalyDetectionEvaluator.OverallMetrics.DrAtNumPos);
            NumAnomalies = FetchInt(AnomalyDetectionEvaluator.OverallMetrics.NumAnomalies);
            ThreshAtK = FetchFloat(AnomalyDetectionEvaluator.OverallMetrics.ThreshAtK);
            ThreshAtP = FetchFloat(AnomalyDetectionEvaluator.OverallMetrics.ThreshAtP);
            ThreshAtNumPos = FetchFloat(AnomalyDetectionEvaluator.OverallMetrics.ThreshAtNumPos);
        }
    }
}
