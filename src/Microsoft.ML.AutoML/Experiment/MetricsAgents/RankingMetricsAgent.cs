// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal class RankingMetricsAgent : IMetricsAgent<RankingMetrics>
    {
        private readonly MLContext _mlContext;
        private readonly RankingMetric _optimizingMetric;
        private readonly string _groupIdColumnName;

        public RankingMetricsAgent(MLContext mlContext, RankingMetric optimizingMetric, string groupIdColumnName)
        {
            _mlContext = mlContext;
            _optimizingMetric = optimizingMetric;
            _groupIdColumnName = groupIdColumnName;
        }

        // Optimizing metric used: NDCG@10 and DCG@10
        public double GetScore(RankingMetrics metrics)
        {
            if (metrics == null)
            {
                return double.NaN;
            }

            switch (_optimizingMetric)
            {
                case RankingMetric.Ndcg:
                    return (metrics.NormalizedDiscountedCumulativeGains.Count >= 10) ? metrics.NormalizedDiscountedCumulativeGains[9] :
                        metrics.NormalizedDiscountedCumulativeGains[metrics.NormalizedDiscountedCumulativeGains.Count - 1];
                case RankingMetric.Dcg:
                    return (metrics.DiscountedCumulativeGains.Count >= 10) ? metrics.DiscountedCumulativeGains[9] :
                        metrics.DiscountedCumulativeGains[metrics.DiscountedCumulativeGains.Count-1];
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        // REVIEW: model can't be perfect with DCG
        public bool IsModelPerfect(double score)
        {
            if (double.IsNaN(score))
            {
                return false;
            }

            switch (_optimizingMetric)
            {
                case RankingMetric.Ndcg:
                    return score == 1;
                case RankingMetric.Dcg:
                    return false;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public RankingMetrics EvaluateMetrics(IDataView data, string labelColumn)
        {
            return _mlContext.Ranking.Evaluate(data, labelColumn, _groupIdColumnName);
        }
    }
}
