// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class MultiMetricsAgent : IMetricsAgent<MultiClassClassifierMetrics>
    {
        private readonly MulticlassClassificationMetric _optimizingMetric;

        public MultiMetricsAgent(MulticlassClassificationMetric optimizingMetric)
        {
            this._optimizingMetric = optimizingMetric;
        }

        public double GetScore(MultiClassClassifierMetrics metrics)
        {
            switch (_optimizingMetric)
            {
                case MulticlassClassificationMetric.MacroAccuracy:
                    return metrics.AccuracyMacro;
                case MulticlassClassificationMetric.MicroAccuracy:
                    return metrics.AccuracyMicro;
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

        public bool IsModelPerfect(MultiClassClassifierMetrics metrics)
        {
            if (metrics == null)
            {
                return false;
            }

            switch (_optimizingMetric)
            {
                case MulticlassClassificationMetric.MacroAccuracy:
                    return metrics.AccuracyMacro == 1;
                case MulticlassClassificationMetric.MicroAccuracy:
                    return metrics.AccuracyMicro == 1;
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