// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal class BinaryMetricsAgent : IMetricsAgent<BinaryClassificationMetrics>
    {
        private readonly MLContext _mlContext;
        private readonly BinaryClassificationMetric _optimizingMetric;

        public BinaryMetricsAgent(MLContext mlContext,
            BinaryClassificationMetric optimizingMetric)
        {
            _mlContext = mlContext;
            _optimizingMetric = optimizingMetric;
        }

        public double GetScore(BinaryClassificationMetrics metrics)
        {
            if (metrics == null)
            {
                return double.NaN;
            }

            switch (_optimizingMetric)
            {
                case BinaryClassificationMetric.Accuracy:
                    return metrics.Accuracy;
                case BinaryClassificationMetric.AreaUnderRocCurve:
                    return metrics.AreaUnderRocCurve;
                case BinaryClassificationMetric.AreaUnderPrecisionRecallCurve:
                    return metrics.AreaUnderPrecisionRecallCurve;
                case BinaryClassificationMetric.F1Score:
                    return metrics.F1Score;
                case BinaryClassificationMetric.NegativePrecision:
                    return metrics.NegativePrecision;
                case BinaryClassificationMetric.NegativeRecall:
                    return metrics.NegativeRecall;
                case BinaryClassificationMetric.PositivePrecision:
                    return metrics.PositivePrecision;
                case BinaryClassificationMetric.PositiveRecall:
                    return metrics.PositiveRecall;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public bool IsModelPerfect(double score)
        {
            if (double.IsNaN(score))
            {
                return false;
            }

            switch (_optimizingMetric)
            {
                case BinaryClassificationMetric.Accuracy:
                    return score == 1;
                case BinaryClassificationMetric.AreaUnderRocCurve:
                    return score == 1;
                case BinaryClassificationMetric.AreaUnderPrecisionRecallCurve:
                    return score == 1;
                case BinaryClassificationMetric.F1Score:
                    return score == 1;
                case BinaryClassificationMetric.NegativePrecision:
                    return score == 1;
                case BinaryClassificationMetric.NegativeRecall:
                    return score == 1;
                case BinaryClassificationMetric.PositivePrecision:
                    return score == 1;
                case BinaryClassificationMetric.PositiveRecall:
                    return score == 1;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public BinaryClassificationMetrics EvaluateMetrics(IDataView data, string labelColumn, string groupIdColumn)
        {
            return _mlContext.BinaryClassification.EvaluateNonCalibrated(data, labelColumn);
        }
    }
}
