// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.Auto
{
    /// <summary>
    /// Settings for AutoML experiments on regression datasets.
    /// </summary>
    public sealed class RegressionExperimentSettings : ExperimentSettings
    {
        /// <summary>
        /// Metric that AutoML will try to optimize over the course of the experiment.
        /// </summary>
        public RegressionMetric OptimizingMetric { get; set; } = RegressionMetric.RSquared;

        /// <summary>
        /// Collection of trainers the AutoML experiment can leverage.
        /// </summary>
        /// <remarks>
        /// The collection is auto-populated with all possible trainers (all values of <see cref="RegressionTrainer" />).
        /// </remarks>
        /// <example>
        /// <code>
        /// // Create an AutoML experiment that can only use LightGBM
        /// var experimentSettings = new RegressionExperimentSettings();
        /// experimentSettings.MaxExperimentTimeInSeconds = 60;
        /// experimentSettings.Trainers.Clear();
        /// experimentSettings.Trainers.Add(RegressionTrainer.LightGbm);
        /// var experiment = new MLContext().Auto().CreateRegressionExperiment(experimentSettings);
        /// </code>
        /// </example>
        public ICollection<RegressionTrainer> Trainers { get; } =
                     Enum.GetValues(typeof(RegressionTrainer)).OfType<RegressionTrainer>().ToList();
    }

    /// <summary>
    /// Regression metric that AutoML will try to optimize in its sweeping process during an experiment.
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
        /// <see cref="FastForestRegressionTrainer"/>
        /// </summary>
        FastForest,

        /// <summary>
        /// <see cref="FastTreeRegressionTrainer"/>
        /// </summary>
        FastTree,

        /// <summary>
        /// <see cref="FastTreeTweedieTrainer"/>
        /// </summary>
        FastTreeTweedie,

        /// <summary>
        /// <see cref="LightGbmRegressionTrainer"/>
        /// </summary>
        LightGbm,

        /// <summary>
        /// <see cref="OnlineGradientDescentTrainer"/>
        /// </summary>
        OnlineGradientDescent,

        /// <summary>
        /// <see cref="OlsTrainer"/>
        /// </summary>
        Ols,

        /// <summary>
        /// <see cref="LbfgsPoissonRegressionTrainer"/>
        /// </summary>
        LbfgsPoissonRegression,

        /// <summary>
        /// <see cref="SdcaRegressionTrainer"/>
        /// </summary>
        StochasticDualCoordinateAscent,
    }

    /// <summary>
    /// AutoML experiment on regression classification datasets.
    /// </summary>
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
        /// <returns></returns>
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
        public static CrossValidationRunDetail<RegressionMetrics> Best(this IEnumerable<CrossValidationRunDetail<RegressionMetrics>> results, RegressionMetric metric = RegressionMetric.RSquared)
        {
            var metricsAgent = new RegressionMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }
    }
}
