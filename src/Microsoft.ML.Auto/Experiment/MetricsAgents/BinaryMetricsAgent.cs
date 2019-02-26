// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class BinaryMetricsAgent : IMetricsAgent<BinaryClassificationMetrics>
    {
        private readonly BinaryClassificationMetric _optimizingMetric;

        public BinaryMetricsAgent(BinaryClassificationMetric optimizingMetric)
        {
            this._optimizingMetric = optimizingMetric;
        }

        public double GetScore(BinaryClassificationMetrics metrics)
        {
            switch(_optimizingMetric)
            {
                case BinaryClassificationMetric.Accuracy:
                    return metrics.Accuracy;
                case BinaryClassificationMetric.Auc:
                    return metrics.Auc;
                case BinaryClassificationMetric.Auprc:
                    return metrics.Auprc;
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

        public bool IsModelPerfect(BinaryClassificationMetrics metrics)
        {
            if (metrics == null)
            {
                return false;
            }

            switch (_optimizingMetric)
            {
                case BinaryClassificationMetric.Accuracy:
                    return metrics.Accuracy == 1;
                case BinaryClassificationMetric.Auc:
                    return metrics.Auc == 1;
                case BinaryClassificationMetric.Auprc:
                    return metrics.Auprc == 1;
                case BinaryClassificationMetric.F1Score:
                    return metrics.F1Score == 1;
                case BinaryClassificationMetric.NegativePrecision:
                    return metrics.NegativePrecision == 1;
                case BinaryClassificationMetric.NegativeRecall:
                    return metrics.NegativeRecall == 1;
                case BinaryClassificationMetric.PositivePrecision:
                    return metrics.PositivePrecision == 1;
                case BinaryClassificationMetric.PositiveRecall:
                    return metrics.PositiveRecall == 1;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }
    }
}
