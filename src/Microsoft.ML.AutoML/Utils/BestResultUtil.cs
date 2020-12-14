// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal class BestResultUtil
    {
        public static RunDetail<BinaryClassificationMetrics> GetBestRun(IEnumerable<RunDetail<BinaryClassificationMetrics>> results,
            BinaryClassificationMetric metric)
        {
            var metricsAgent = new BinaryMetricsAgent(null, metric);
            var metricInfo = new OptimizingMetricInfo(metric);
            return GetBestRun(results, metricsAgent, metricInfo.IsMaximizing);
        }

        public static RunDetail<RegressionMetrics> GetBestRun(IEnumerable<RunDetail<RegressionMetrics>> results,
            RegressionMetric metric)
        {
            var metricsAgent = new RegressionMetricsAgent(null, metric);
            var metricInfo = new OptimizingMetricInfo(metric);
            return GetBestRun(results, metricsAgent, metricInfo.IsMaximizing);
        }

        public static RunDetail<MulticlassClassificationMetrics> GetBestRun(IEnumerable<RunDetail<MulticlassClassificationMetrics>> results,
            MulticlassClassificationMetric metric)
        {
            var metricsAgent = new MultiMetricsAgent(null, metric);
            var metricInfo = new OptimizingMetricInfo(metric);
            return GetBestRun(results, metricsAgent, metricInfo.IsMaximizing);
        }

        public static RunDetail<RankingMetrics> GetBestRun(IEnumerable<RunDetail<RankingMetrics>> results,
            RankingMetric metric, uint dcgTruncationLevel)
        {
            var metricsAgent = new RankingMetricsAgent(null, metric, dcgTruncationLevel);

            var metricInfo = new OptimizingMetricInfo(metric);
            return GetBestRun(results, metricsAgent, metricInfo.IsMaximizing);
        }

        public static RunDetail<TMetrics> GetBestRun<TMetrics>(IEnumerable<RunDetail<TMetrics>> results,
            IMetricsAgent<TMetrics> metricsAgent, bool isMetricMaximizing)
        {
            results = results.Where(r => r.ValidationMetrics != null);
            if (!results.Any()) { return null; }
            var scores = results.Select(r => metricsAgent.GetScore(r.ValidationMetrics));
            var indexOfBestScore = GetIndexOfBestScore(scores, isMetricMaximizing);
            // indexOfBestScore will be -1 if the optimization metric for all models is NaN.
            // In this case, return the first model.
            indexOfBestScore = indexOfBestScore != -1 ? indexOfBestScore : 0;
            return results.ElementAt(indexOfBestScore);
        }

        public static CrossValidationRunDetail<TMetrics> GetBestRun<TMetrics>(IEnumerable<CrossValidationRunDetail<TMetrics>> results,
            IMetricsAgent<TMetrics> metricsAgent, bool isMetricMaximizing)
        {
            results = results.Where(r => r.Results != null && r.Results.Any(x => x.ValidationMetrics != null));
            if (!results.Any()) { return null; }
            var scores = results.Select(r => r.Results.Average(x => metricsAgent.GetScore(x.ValidationMetrics)));
            var indexOfBestScore = GetIndexOfBestScore(scores, isMetricMaximizing);
            // indexOfBestScore will be -1 if the optimization metric for all models is NaN.
            // In this case, return the first model.
            indexOfBestScore = indexOfBestScore != -1 ? indexOfBestScore : 0;
            return results.ElementAt(indexOfBestScore);
        }

        public static IEnumerable<(RunDetail<T>, int)> GetTopNRunResults<T>(IEnumerable<RunDetail<T>> results,
            IMetricsAgent<T> metricsAgent, int n, bool isMetricMaximizing)
        {
            results = results.Where(r => r.ValidationMetrics != null);
            if (!results.Any()) { return null; }

            var indexedValues = results.Select((k, v) => (k, v));

            IEnumerable<(RunDetail<T>, int)> orderedResults;
            if (isMetricMaximizing)
            {
                orderedResults = indexedValues.OrderByDescending(t => metricsAgent.GetScore(t.Item1.ValidationMetrics));

            }
            else
            {
                orderedResults = indexedValues.OrderBy(t => metricsAgent.GetScore(t.Item1.ValidationMetrics));
            }

            return orderedResults.Take(n);
        }

        public static int GetIndexOfBestScore(IEnumerable<double> scores, bool isMetricMaximizing)
        {
            return isMetricMaximizing ? GetIndexOfMaxScore(scores) : GetIndexOfMinScore(scores);
        }

        private static int GetIndexOfMinScore(IEnumerable<double> scores)
        {
            var minScore = double.PositiveInfinity;
            var minIndex = -1;
            for (var i = 0; i < scores.Count(); i++)
            {
                if (scores.ElementAt(i) < minScore)
                {
                    minScore = scores.ElementAt(i);
                    minIndex = i;
                }
            }
            return minIndex;
        }

        private static int GetIndexOfMaxScore(IEnumerable<double> scores)
        {
            var maxScore = double.NegativeInfinity;
            var maxIndex = -1;
            for (var i = 0; i < scores.Count(); i++)
            {
                if (scores.ElementAt(i) > maxScore)
                {
                    maxScore = scores.ElementAt(i);
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
    }
}
