// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.AutoML
{
    public sealed class RankingExperimentSettings : ExperimentSettings
    {
        /// <summary>
        /// Metric that AutoML will try to optimize over the course of the experiment.
        /// </summary>
        /// <value>The default value is <see cref="RankingMetric" />.</value>
        public RankingMetric OptimizingMetric { get; set; }

        /// <summary>
        /// Collection of trainers the AutoML experiment can leverage.
        /// </summary>
        /// <value>
        /// The default value is a collection auto-populated with all possible trainers (all values of <see cref="RankingTrainer" />).
        /// </value>
        public ICollection<RankingTrainer> Trainers { get; }

        /// <summary>
        /// Maximum truncation level for computing (N)DCG
        /// </summary>
        /// <value>
        /// The default value is 10.
        /// </value>
        public uint OptimizationMetricTruncationLevel { get; set; }

        /// <summary>
        /// Initializes a new instance of <see cref="RankingExperimentSettings"/>.
        /// </summary>
        public RankingExperimentSettings()
        {
            OptimizingMetric = RankingMetric.Ndcg;
            Trainers = Enum.GetValues(typeof(RankingTrainer)).OfType<RankingTrainer>().ToList();
            OptimizationMetricTruncationLevel = 10;
        }
    }
    public enum RankingMetric
    {
        /// <summary>
        /// See <see cref="RankingMetrics.NormalizedDiscountedCumulativeGains"/>.
        /// </summary>
        Ndcg,
        /// <summary>
        /// See <see cref="RankingMetrics.DiscountedCumulativeGains"/>.
        /// </summary>
        Dcg
    }
    /// <summary>
    /// Enumeration of ML.NET ranking trainers used by AutoML.
    /// </summary>
    public enum RankingTrainer
    {
        /// <summary>
        /// See <see cref="LightGbmRankingTrainer"/>.
        /// </summary>
        LightGbmRanking,
        /// <summary>
        /// See <see cref="FastTreeRankingTrainer"/>.
        /// </summary>
        FastTreeRanking
    }

    /// <summary>
    /// Extension methods that operate over ranking experiment run results.
    /// </summary>
    public static class RankingExperimentResultExtensions
    {
        /// <summary>
        /// Select the best run from an enumeration of experiment runs.
        /// </summary>
        /// <param name="results">Enumeration of AutoML experiment run results.</param>
        /// <param name="metric">Metric to consider when selecting the best run.</param>
        /// <param name="optimizationMetricTruncationLevel">Maximum truncation level for computing (N)DCG. Defaults to 10.</param>
        /// <returns>The best experiment run.</returns>
        public static RunDetail<RankingMetrics> Best(this IEnumerable<RunDetail<RankingMetrics>> results, RankingMetric metric = RankingMetric.Ndcg, uint optimizationMetricTruncationLevel = 10)
        {
            var metricsAgent = new RankingMetricsAgent(null, metric, optimizationMetricTruncationLevel);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }

        /// <summary>
        /// Select the best run from an enumeration of experiment cross validation runs.
        /// </summary>
        /// <param name="results">Enumeration of AutoML experiment cross validation run results.</param>
        /// <param name="metric">Metric to consider when selecting the best run.</param>
        /// <param name="optimizationMetricTruncationLevel">Maximum truncation level for computing (N)DCG. Defaults to 10.</param>
        /// <returns>The best experiment run.</returns>
        public static CrossValidationRunDetail<RankingMetrics> Best(this IEnumerable<CrossValidationRunDetail<RankingMetrics>> results, RankingMetric metric = RankingMetric.Ndcg, uint optimizationMetricTruncationLevel = 10)
        {
            var metricsAgent = new RankingMetricsAgent(null, metric, optimizationMetricTruncationLevel);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }
    }

    /// <summary>
    /// AutoML experiment on ranking datasets.
    /// </summary>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    ///  [!code-csharp[RankingExperiment](~/../docs/samples/docs/samples/Microsoft.ML.AutoML.Samples/RankingExperiment.cs)]
    /// ]]></format>
    /// </example>
    public sealed class RankingExperiment : ExperimentBase<RankingMetrics, RankingExperimentSettings>
    {
        internal RankingExperiment(MLContext context, RankingExperimentSettings settings)
            : base(context,
                  new RankingMetricsAgent(context, settings.OptimizingMetric, settings.OptimizationMetricTruncationLevel),
                  new OptimizingMetricInfo(settings.OptimizingMetric),
                  settings,
                  TaskKind.Ranking,
                  TrainerExtensionUtil.GetTrainerNames(settings.Trainers))
        {
        }

        private protected override CrossValidationRunDetail<RankingMetrics> GetBestCrossValRun(IEnumerable<CrossValidationRunDetail<RankingMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private protected override RunDetail<RankingMetrics> GetBestRun(IEnumerable<RunDetail<RankingMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }
    }
}
