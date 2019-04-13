// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public sealed class RegressionExperimentSettings : ExperimentSettings
    {
        public RegressionMetric OptimizingMetric { get; set; } = RegressionMetric.RSquared;
        public ICollection<RegressionTrainer> Trainers { get; } =
                     Enum.GetValues(typeof(RegressionTrainer)).OfType<RegressionTrainer>().ToList();
    }

    public enum RegressionMetric
    {
        MeanAbsoluteError,
        MeanSquaredError,
        RootMeanSquaredError,
        RSquared
    }

    public enum RegressionTrainer
    {
        FastForest,
        FastTree,
        FastTreeTweedie,
        LightGbm,
        OnlineGradientDescent,
        Ols,
        LbfgsPoissonRegression,
        StochasticDualCoordinateAscent,
    }

    public sealed class RegressionExperiment : ExperimentBase<RegressionMetrics>
    {
        internal RegressionExperiment(MLContext context, RegressionExperimentSettings settings) 
            : base(context,
                  new RegressionMetricsAgent(context, settings.OptimizingMetric),
                  new OptimizingMetricInfo(settings.OptimizingMetric),
                  settings,
                  TaskKind.Regression,
                  TrainerExtensionUtil.GetTrainerNames(settings.Trainers))
        {
        }
    }

    public static class RegressionExperimentResultExtensions
    {
        public static RunDetail<RegressionMetrics> Best(this IEnumerable<RunDetail<RegressionMetrics>> results, RegressionMetric metric = RegressionMetric.RSquared)
        {
            var metricsAgent = new RegressionMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }

        public static CrossValidationRunDetail<RegressionMetrics> Best(this IEnumerable<CrossValidationRunDetail<RegressionMetrics>> results, RegressionMetric metric = RegressionMetric.RSquared)
        {
            var metricsAgent = new RegressionMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }
    }
}
