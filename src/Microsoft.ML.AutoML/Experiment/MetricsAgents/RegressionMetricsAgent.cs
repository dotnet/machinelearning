// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal class RegressionMetricsAgent : IMetricsAgent<RegressionMetrics>
    {
        private readonly MLContext _mlContext;
        private readonly RegressionMetric _optimizingMetric;

        public RegressionMetricsAgent(MLContext mlContext, RegressionMetric optimizingMetric)
        {
            _mlContext = mlContext;
            _optimizingMetric = optimizingMetric;
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

        public bool IsModelPerfect(double score)
        {
            if (double.IsNaN(score))
            {
                return false;
            }

            switch (_optimizingMetric)
            {
                case RegressionMetric.MeanAbsoluteError:
                    return score == 0;
                case RegressionMetric.MeanSquaredError:
                    return score == 0;
                case RegressionMetric.RootMeanSquaredError:
                    return score == 0;
                case RegressionMetric.RSquared:
                    return score == 1;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public RegressionMetrics EvaluateMetrics(IDataView data, string labelColumn, string groupIdColumn)
        {
            return _mlContext.Regression.Evaluate(data, labelColumn);
        }
    }
}
