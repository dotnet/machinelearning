// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class MultiMetricsAgent : IMetricsAgent<MulticlassClassificationMetrics>
    {
        private readonly MulticlassClassificationMetric _optimizingMetric;

        public MultiMetricsAgent(MulticlassClassificationMetric optimizingMetric)
        {
            this._optimizingMetric = optimizingMetric;
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

        public bool IsModelPerfect(MulticlassClassificationMetrics metrics)
        {
            if (metrics == null)
            {
                return false;
            }

            switch (_optimizingMetric)
            {
                case MulticlassClassificationMetric.MacroAccuracy:
                    return metrics.MacroAccuracy == 1;
                case MulticlassClassificationMetric.MicroAccuracy:
                    return metrics.MicroAccuracy == 1;
                case MulticlassClassificationMetric.LogLoss:
                    return metrics.LogLoss == 0;
                case MulticlassClassificationMetric.LogLossReduction:
                    return metrics.LogLossReduction == 1;
                case MulticlassClassificationMetric.TopKAccuracy:
                    return metrics.TopKAccuracy == 1;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }
    }
}