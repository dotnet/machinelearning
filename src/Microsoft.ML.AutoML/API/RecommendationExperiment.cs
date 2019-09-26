// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.AutoML
{
    public enum RecommendationExperimentScenario
    {
        MatrixFactorization = 0,
    }

    public sealed class RecommendationExperimentSettings : ExperimentSettings
    {
        public RecommendationExperimentScenario Scenerio { get; set; }

        public string MatrixColumnIndexColumnName { get; set; }

        public string MatrixRowIndexColumnName { get; set; }

        // We can use RegressionMetric as evaluation Metric
        public RegressionMetric OptimizingMetric { get; set; }

        public ICollection<RecommendationTrainer> Trainers { get; }

        public RecommendationExperimentSettings(RecommendationExperimentScenario scenario, string columnIndexName, string rowIndexName)
            : this()
        {
            if (scenario == RecommendationExperimentScenario.MatrixFactorization)
            {
                AutoCatalog.ValuePairs[nameof(MatrixFactorizationTrainer.Options.MatrixColumnIndexColumnName)] = columnIndexName;
                AutoCatalog.ValuePairs[nameof(MatrixFactorizationTrainer.Options.MatrixRowIndexColumnName)] = rowIndexName;
                return;
            }
            throw new NotImplementedException();
        }

        private RecommendationExperimentSettings()
        {
            OptimizingMetric = RegressionMetric.RSquared;
            Trainers = Enum.GetValues(typeof(RecommendationTrainer)).OfType<RecommendationTrainer>().ToList();
        }
    }

    public enum RecommendationTrainer
    {
        MatrixFactorization,
    }

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
