// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Settings for AutoML experiments on regression datasets.
    /// </summary>
    public sealed class RegressionExperimentSettings : ExperimentSettings
    {
        /// <summary>
        /// Metric that AutoML will try to optimize over the course of the experiment.
        /// </summary>
        /// <value>The default value is <see cref="RegressionMetric.RSquared" />.</value>
        public RegressionMetric OptimizingMetric { get; set; }

        /// <summary>
        /// Collection of trainers the AutoML experiment can leverage.
        /// </summary>
        /// <value>
        /// The default value is a collection auto-populated with all possible trainers (all values of <see cref="RegressionTrainer" />).
        /// </value>
        public ICollection<RegressionTrainer> Trainers { get; }

        /// <summary>
        /// Initializes a new instance of <see cref="RegressionExperimentSettings"/>.
        /// </summary>
        public RegressionExperimentSettings()
        {
            OptimizingMetric = RegressionMetric.RSquared;
            Trainers = Enum.GetValues(typeof(RegressionTrainer)).OfType<RegressionTrainer>().ToList();
        }
    }

    /// <summary>
    /// Regression metric that AutoML will aim to optimize in its sweeping process during an experiment.
    /// </summary>
    public enum RegressionMetric
    {
        /// <summary>
        /// See <see cref="RegressionMetrics.MeanAbsoluteError"/>.
        /// </summary>
        MeanAbsoluteError,

        /// <summary>
        /// See <see cref="RegressionMetrics.MeanSquaredError"/>.
        /// </summary>
        MeanSquaredError,

        /// <summary>
        /// See <see cref="RegressionMetrics.RootMeanSquaredError"/>.
        /// </summary>
        RootMeanSquaredError,

        /// <summary>
        /// See <see cref="RegressionMetrics.RSquared"/>.
        /// </summary>
        RSquared
    }

    /// <summary>
    /// Enumeration of ML.NET multiclass classification trainers used by AutoML.
    /// </summary>
    public enum RegressionTrainer
    {
        /// <summary>
        /// See <see cref="FastForestRegressionTrainer"/>.
        /// </summary>
        FastForest,

        /// <summary>
        /// See <see cref="FastTreeRegressionTrainer"/>.
        /// </summary>
        FastTree,

        /// <summary>
        /// See <see cref="FastTreeTweedieTrainer"/>.
        /// </summary>
        FastTreeTweedie,

        /// <summary>
        /// See <see cref="LightGbmRegressionTrainer"/>.
        /// </summary>
        LightGbm,

        /// <summary>
        /// See <see cref="OnlineGradientDescentTrainer"/>.
        /// </summary>
        OnlineGradientDescent,

        /// <summary>
        /// See <see cref="OlsTrainer"/>.
        /// </summary>
        Ols,

        /// <summary>
        /// See <see cref="LbfgsPoissonRegressionTrainer"/>.
        /// </summary>
        LbfgsPoissonRegression,

        /// <summary>
        /// See <see cref="SdcaRegressionTrainer"/>.
        /// </summary>
        StochasticDualCoordinateAscent,
    }

    /// <summary>
    /// AutoML experiment on regression classification datasets.
    /// </summary>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    ///  [!code-csharp[RegressionExperiment](~/../docs/samples/docs/samples/Microsoft.ML.AutoML.Samples/RegressionExperiment.cs)]
    /// ]]></format>
    /// </example>
    public sealed class RegressionExperiment : ExperimentBase<RegressionMetrics, RegressionExperimentSettings>
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

        private protected override CrossValidationRunDetail<RegressionMetrics> GetBestCrossValRun(IEnumerable<CrossValidationRunDetail<RegressionMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private protected override RunDetail<RegressionMetrics> GetBestRun(IEnumerable<RunDetail<RegressionMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }
    }

    /// <summary>
    /// Extension methods that operate over regression experiment run results.
    /// </summary>
    public static class RegressionExperimentResultExtensions
    {
        /// <summary>
        /// Select the best run from an enumeration of experiment runs.
        /// </summary>
        /// <param name="results">Enumeration of AutoML experiment run results.</param>
        /// <param name="metric">Metric to consider when selecting the best run.</param>
        /// <returns>The best experiment run.</returns>
        public static RunDetail<RegressionMetrics> Best(this IEnumerable<RunDetail<RegressionMetrics>> results, RegressionMetric metric = RegressionMetric.RSquared)
        {
            var metricsAgent = new RegressionMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }

        /// <summary>
        /// Select the best run from an enumeration of experiment cross validation runs.
        /// </summary>
        /// <param name="results">Enumeration of AutoML experiment cross validation run results.</param>
        /// <param name="metric">Metric to consider when selecting the best run.</param>
        /// <returns>The best experiment run.</returns>
        public static CrossValidationRunDetail<RegressionMetrics> Best(this IEnumerable<CrossValidationRunDetail<RegressionMetrics>> results, RegressionMetric metric = RegressionMetric.RSquared)
        {
            var metricsAgent = new RegressionMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }
    }
}
