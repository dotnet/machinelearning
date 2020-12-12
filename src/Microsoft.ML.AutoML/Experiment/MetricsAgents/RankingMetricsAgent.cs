// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    internal class RankingMetricsAgent : IMetricsAgent<RankingMetrics>
    {
        private readonly MLContext _mlContext;
        private readonly RankingMetric _optimizingMetric;
        private readonly uint _dcgTruncationLevel;

        public RankingMetricsAgent(MLContext mlContext, RankingMetric metric, uint optimizationMetricTruncationLevel)
        {
            _mlContext = mlContext;
            _optimizingMetric = metric;

            if (optimizationMetricTruncationLevel <= 0)
                throw _mlContext.ExceptUserArg(nameof(optimizationMetricTruncationLevel), "DCG Truncation Level must be greater than 0");

            // We want to make sure we always report metrics for at least 10 results (e.g. NDCG@10) to the user.
            // Producing extra results adds no measurable performance impact, so we report at least 2x of the
            // user's requested optimization truncation level.
            _dcgTruncationLevel = optimizationMetricTruncationLevel;
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
                    return metrics.NormalizedDiscountedCumulativeGains[Math.Min(metrics.NormalizedDiscountedCumulativeGains.Count, (int)_dcgTruncationLevel) - 1];
                case RankingMetric.Dcg:
                    return metrics.DiscountedCumulativeGains[Math.Min(metrics.DiscountedCumulativeGains.Count, (int)_dcgTruncationLevel) - 1];
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

        public RankingMetrics EvaluateMetrics(IDataView data, string labelColumn, string groupIdColumn)
        {
            var rankingEvalOptions = new RankingEvaluatorOptions
            {
                DcgTruncationLevel = Math.Max(10, 2 * (int)_dcgTruncationLevel)
            };

            return _mlContext.Ranking.Evaluate(data, rankingEvalOptions, labelColumn, groupIdColumn);
        }
    }
}
