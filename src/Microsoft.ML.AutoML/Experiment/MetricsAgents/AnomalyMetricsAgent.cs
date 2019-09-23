using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoML.Experiment.MetricsAgents
{
    internal class AnomalyMetricsAgent : IMetricsAgent<AnomalyDetectionMetrics>
    {
        private readonly MLContext _mlContext;
        private readonly AnomalyDetectionMetric _optimizingMetric;

        public AnomalyMetricsAgent(MLContext mlContext,
            AnomalyDetectionMetric optimizingMetric)
        {
            _mlContext = mlContext;
            _optimizingMetric = optimizingMetric;
        }

        public double GetScore(AnomalyDetectionMetrics metrics)
        {
            if (metrics == null)
            {
                return double.NaN;
            }

            switch (_optimizingMetric)
            {
                case AnomalyDetectionMetric.FakeAccuracy:
                    return metrics.FakeAccuracy;
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
                case AnomalyDetectionMetric.FakeAccuracy:
                    return score == 1;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public AnomalyDetectionMetrics EvaluateMetrics(IDataView data, string labelColumn)
        {
            return new AnomalyDetectionMetrics();
        }
    }
}
