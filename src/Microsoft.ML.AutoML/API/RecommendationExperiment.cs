// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Settings for AutoML experiments on recommendation datasets.
    /// </summary>
    public sealed class RecommendationExperimentSettings : ExperimentSettings
    {
        /// <summary>
        /// Metric that AutoML will try to optimize over the course of the experiment.
        /// </summary>
        /// <value>The default value is <see cref="RegressionMetric.RSquared"/>.</value>
        public RegressionMetric OptimizingMetric { get; set; }

        /// <summary>
        /// Collection of trainers the AutoML experiment can leverage.
        /// </summary>
        /// <value>The default value is a collection auto-populated with all possible trainers (all values of <see cref="RecommendationTrainer" />).</value>
        public ICollection<RecommendationTrainer> Trainers { get; }

        /// <summary>
        /// Initializes a new instance of <see cref="RecommendationExperimentSettings"/>.
        /// </summary>
        public RecommendationExperimentSettings()
        {
            OptimizingMetric = RegressionMetric.RSquared;
            Trainers = Enum.GetValues(typeof(RecommendationTrainer)).OfType<RecommendationTrainer>().ToList();
        }
    }

    /// <summary>
    /// Enumeration of ML.NET recommendation trainers used by AutoML.
    /// </summary>
    public enum RecommendationTrainer
    {
        MatrixFactorization
    }

    /// <summary>
    /// AutoML experiment on recommendation datasets.
    /// </summary>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    ///  [!code-csharp[RecommendationExperiment](~/../docs/samples/docs/samples/Microsoft.ML.AutoML.Samples/RecommendationExperiment.cs)]
    /// ]]></format>
    /// </example>
    public sealed class RecommendationExperiment : ExperimentBase<RegressionMetrics, RecommendationExperimentSettings>
    {
        internal RecommendationExperiment(MLContext context, RecommendationExperimentSettings settings)
            : base(context,
                  new RegressionMetricsAgent(context, settings.OptimizingMetric),
                  new OptimizingMetricInfo(settings.OptimizingMetric),
                  settings,
                  TaskKind.Recommendation,
                  TrainerExtensionUtil.GetTrainerNames(settings.Trainers))
        {
        }

        private protected override CrossValidationRunDetail<RegressionMetrics> GetBestCrossValRun(IEnumerable<CrossValidationRunDetail<RegressionMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private protected override RunDetail<RegressionMetrics> GetBestRun(IEnumerable<RunDetail<RegressionMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }
    }
}
