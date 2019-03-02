// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class RegressionMetricsAgent : IMetricsAgent<RegressionMetrics>
    {
        private readonly RegressionMetric _optimizingMetric;

        public RegressionMetricsAgent(RegressionMetric optimizingMetric)
        {
            this._optimizingMetric = optimizingMetric;
        }

        public double GetScore(RegressionMetrics metrics)
        {
            switch (_optimizingMetric)
            {
                case RegressionMetric.MeanAbsoluteError:
                    return metrics.L1;
                case RegressionMetric.MeanSquaredError:
                    return metrics.L2;
                case RegressionMetric.RootMeanSquaredError:
                    return metrics.Rms;
                case RegressionMetric.RSquared:
                    return metrics.RSquared;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public bool IsModelPerfect(RegressionMetrics metrics)
        {
            if (metrics == null)
            {
                return false;
            }

            switch (_optimizingMetric)
            {
                case RegressionMetric.MeanAbsoluteError:
                    return metrics.L1 == 0;
                case RegressionMetric.MeanSquaredError:
                    return metrics.L2 == 0;
                case RegressionMetric.RootMeanSquaredError:
                    return metrics.Rms == 0;
                case RegressionMetric.RSquared:
                    return metrics.RSquared == 1;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }
    }
}
