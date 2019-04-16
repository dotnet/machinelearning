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
    /// Settings for AutoML experiments on multiclass classification datasets.
    /// </summary>
    public sealed class MulticlassExperimentSettings : ExperimentSettings
    {
        /// <summary>
        /// Metric that AutoML will try to optimize over the course of the experiment.
        /// </summary>
        public MulticlassClassificationMetric OptimizingMetric { get; set; } = MulticlassClassificationMetric.MicroAccuracy;

        /// <summary>
        /// Collection of trainers the AutoML experiment can leverage.
        /// </summary>
        /// <remarks>
        /// The collection is auto-populated with all possible trainers (all values of <see cref="MulticlassClassificationTrainer" />).
        /// </remarks>
        public ICollection<MulticlassClassificationTrainer> Trainers { get; } =
            Enum.GetValues(typeof(MulticlassClassificationTrainer)).OfType<MulticlassClassificationTrainer>().ToList();
    }

    /// <summary>
    /// Multiclass classification metric that AutoML will aim to optimize in its sweeping process during an experiment.
    /// </summary>
    public enum MulticlassClassificationMetric
    {
        /// <summary>
        /// See <see cref="MulticlassClassificationMetrics.MicroAccuracy"/>.
        /// </summary>
        MicroAccuracy,

        /// <summary>
        /// See <see cref="MulticlassClassificationMetrics.MacroAccuracy"/>.
        /// </summary>
        MacroAccuracy,

        /// <summary>
        /// See <see cref="MulticlassClassificationMetrics.LogLoss"/>.
        /// </summary>
        LogLoss,

        /// <summary>
        /// See <see cref="MulticlassClassificationMetrics.LogLossReduction"/>.
        /// </summary>
        LogLossReduction,

        /// <summary>
        /// See <see cref="MulticlassClassificationMetrics.TopKAccuracy"/>.
        /// </summary>
        TopKAccuracy,
    }

    /// <summary>
    /// Enumeration of ML.NET multiclass classification trainers used by AutoML.
    /// </summary>
    public enum MulticlassClassificationTrainer
    {
        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="AveragedPerceptronTrainer"/>.
        /// </summary>
        AveragedPerceptronOVA,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="FastForestBinaryTrainer"/>.
        /// </summary>
        FastForestOVA,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="FastTreeBinaryTrainer"/>.
        /// </summary>
        FastTreeOVA,

        /// <summary>
        /// See <see cref="LightGbmMulticlassTrainer"/>.
        /// </summary>
        LightGbm,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="LinearSvmTrainer"/>.
        /// </summary>
        LinearSupportVectorMachinesOVA,

        /// <summary>
        /// See <see cref="LbfgsMaximumEntropyMulticlassTrainer"/>.
        /// </summary>
        LbfgsMaximumEntropy,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="LbfgsLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        LbfgsLogisticRegressionOVA,

        /// <summary>
        /// See <see cref="SdcaMaximumEntropyMulticlassTrainer"/>.
        /// </summary>
        SdcaMaximumEntropy,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="LbfgsLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        SgdCalibratedOVA,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="SymbolicSgdLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        SymbolicSgdLogisticRegressionOVA,
    }

    /// <summary>
    /// AutoML experiment on multiclass classification datasets.
    /// </summary>
    public sealed class MulticlassClassificationExperiment : ExperimentBase<MulticlassClassificationMetrics>
    {
        internal MulticlassClassificationExperiment(MLContext context, MulticlassExperimentSettings settings)
            : base(context,
                  new MultiMetricsAgent(context, settings.OptimizingMetric),
                  new OptimizingMetricInfo(settings.OptimizingMetric),
                  settings,
                  TaskKind.MulticlassClassification,
                  TrainerExtensionUtil.GetTrainerNames(settings.Trainers))
        {
        }
    }

    /// <summary>
    /// Extension methods that operate over multiclass experiment run results.
    /// </summary>
    public static class MulticlassExperimentResultExtensions
    {
        /// <summary>
        /// Select the best run from an enumeration of experiment runs.
        /// </summary>
        /// <param name="results">Enumeration of AutoML experiment run results.</param>
        /// <param name="metric">Metric to consider when selecting the best run.</param>
        /// <returns>The best experiment run.</returns>
        public static RunDetail<MulticlassClassificationMetrics> Best(this IEnumerable<RunDetail<MulticlassClassificationMetrics>> results, MulticlassClassificationMetric metric = MulticlassClassificationMetric.MicroAccuracy)
        {
            var metricsAgent = new MultiMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }


        /// <summary>
        /// Select the best run from an enumeration of experiment cross validation runs.
        /// </summary>
        /// <param name="results">Enumeration of AutoML experiment cross validation run results.</param>
        /// <param name="metric">Metric to consider when selecting the best run.</param>
        /// <returns>The best experiment run.</returns>
        public static CrossValidationRunDetail<MulticlassClassificationMetrics> Best(this IEnumerable<CrossValidationRunDetail<MulticlassClassificationMetrics>> results, MulticlassClassificationMetric metric = MulticlassClassificationMetric.MicroAccuracy)
        {
            var metricsAgent = new MultiMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }
    }
}