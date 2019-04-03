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
            if (metrics == null)
            {
                return double.NaN;
            }

            switch (_optimizingMetric)
            {
                case RegressionMetric.MeanAbsoluteError:
                    return metrics.MeanAbsoluteError;
                case RegressionMetric.MeanSquaredError:
                    return metrics.MeanSquaredError;
                case RegressionMetric.RootMeanSquaredError:
                    return metrics.RootMeanSquaredError;
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
                    return metrics.MeanAbsoluteError == 0;
                case RegressionMetric.MeanSquaredError:
                    return metrics.MeanSquaredError == 0;
                case RegressionMetric.RootMeanSquaredError:
                    return metrics.RootMeanSquaredError == 0;
                case RegressionMetric.RSquared:
                    return metrics.RSquared == 1;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }
    }
}
