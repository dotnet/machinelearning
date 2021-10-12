// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal class MultiMetricsAgent : IMetricsAgent<MulticlassClassificationMetrics>
    {
        private readonly MLContext _mlContext;
        private readonly MulticlassClassificationMetric _optimizingMetric;

        public MultiMetricsAgent(MLContext mlContext,
            MulticlassClassificationMetric optimizingMetric)
        {
            _mlContext = mlContext;
            _optimizingMetric = optimizingMetric;
        }

        public double GetScore(MulticlassClassificationMetrics metrics)
        {
            if (metrics == null)
            {
                return double.NaN;
            }

            switch (_optimizingMetric)
            {
                case MulticlassClassificationMetric.MacroAccuracy:
                    return metrics.MacroAccuracy;
                case MulticlassClassificationMetric.MicroAccuracy:
                    return metrics.MicroAccuracy;
                case MulticlassClassificationMetric.LogLoss:
                    return metrics.LogLoss;
                case MulticlassClassificationMetric.LogLossReduction:
                    return metrics.LogLossReduction;
                case MulticlassClassificationMetric.TopKAccuracy:
                    return metrics.TopKAccuracy;
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
                case MulticlassClassificationMetric.MacroAccuracy:
                    return score == 1;
                case MulticlassClassificationMetric.MicroAccuracy:
                    return score == 1;
                case MulticlassClassificationMetric.LogLoss:
                    return score == 0;
                case MulticlassClassificationMetric.LogLossReduction:
                    return score == 1;
                case MulticlassClassificationMetric.TopKAccuracy:
                    return score == 1;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public MulticlassClassificationMetrics EvaluateMetrics(IDataView data, string labelColumn, string groupIdColumn)
        {
            return _mlContext.MulticlassClassification.Evaluate(data, labelColumn);
        }
    }
}
