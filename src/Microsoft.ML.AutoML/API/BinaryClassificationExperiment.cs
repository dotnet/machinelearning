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

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Settings for AutoML experiments on binary classification datasets.
    /// </summary>
    public sealed class BinaryExperimentSettings : ExperimentSettings
    {
        /// <summary>
        /// Metric that AutoML will try to optimize over the course of the experiment.
        /// </summary>
        /// <value>The default value is <see cref="BinaryClassificationMetric.Accuracy"/>.</value>
        public BinaryClassificationMetric OptimizingMetric { get; set; }

        /// <summary>
        /// Collection of trainers the AutoML experiment can leverage.
        /// </summary>
        /// <value>The default value is a collection auto-populated with all possible trainers (all values of <see cref="BinaryClassificationTrainer" />).</value>
        public ICollection<BinaryClassificationTrainer> Trainers { get; }

        /// <summary>
        /// Initializes a new instance of <see cref="BinaryExperimentSettings"/>.
        /// </summary>
        public BinaryExperimentSettings()
        {
            OptimizingMetric = BinaryClassificationMetric.Accuracy;
            Trainers = Enum.GetValues(typeof(BinaryClassificationTrainer)).OfType<BinaryClassificationTrainer>().ToList();
        }
    }

    /// <summary>
    /// Binary classification metric that AutoML will aim to optimize in its sweeping process during an experiment.
    /// </summary>
    public enum BinaryClassificationMetric
    {
        /// <summary>
        /// See <see cref="BinaryClassificationMetrics.Accuracy"/>.
        /// </summary>
        Accuracy,

        /// <summary>
        /// See <see cref="BinaryClassificationMetrics.AreaUnderRocCurve"/>.
        /// </summary>
        AreaUnderRocCurve,

        /// <summary>
        /// See <see cref="BinaryClassificationMetrics.AreaUnderPrecisionRecallCurve"/>.
        /// </summary>
        AreaUnderPrecisionRecallCurve,

        /// <summary>
        /// See <see cref="BinaryClassificationMetrics.F1Score"/>.
        /// </summary>
        F1Score,

        /// <summary>
        /// See <see cref="BinaryClassificationMetrics.PositivePrecision"/>.
        /// </summary>
        PositivePrecision,

        /// <summary>
        /// See <see cref="BinaryClassificationMetrics.PositiveRecall"/>.
        /// </summary>
        PositiveRecall,

        /// <summary>
        /// See <see cref="BinaryClassificationMetrics.NegativePrecision"/>.
        /// </summary>
        NegativePrecision,

        /// <summary>
        /// See <see cref="BinaryClassificationMetrics.NegativeRecall"/>.
        /// </summary>
        NegativeRecall,
    }

    /// <summary>
    /// Enumeration of ML.NET binary classification trainers used by AutoML.
    /// </summary>
    public enum BinaryClassificationTrainer
    {
        /// <summary>
        /// See <see cref="AveragedPerceptronTrainer"/>.
        /// </summary>
        AveragedPerceptron,

        /// <summary>
        /// See <see cref="FastForestBinaryTrainer"/>.
        /// </summary>
        FastForest,

        /// <summary>
        /// See <see cref="FastTreeBinaryTrainer"/>.
        /// </summary>
        FastTree,

        /// <summary>
        /// See <see cref="LightGbmBinaryTrainer"/>.
        /// </summary>
        LightGbm,

        /// <summary>
        /// See <see cref="LinearSvmTrainer"/>.
        /// </summary>
        LinearSvm,

        /// <summary>
        /// See <see cref="LbfgsLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        LbfgsLogisticRegression,

        /// <summary>
        /// See <see cref="SdcaLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        SdcaLogisticRegression,

        /// <summary>
        /// See <see cref="SgdCalibratedTrainer"/>.
        /// </summary>
        SgdCalibrated,

        /// <summary>
        /// See <see cref="SymbolicSgdLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        SymbolicSgdLogisticRegression,
    }

    /// <summary>
    /// AutoML experiment on binary classification datasets.
    /// </summary>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    ///  [!code-csharp[BinaryClassificationExperiment](~/../docs/samples/docs/samples/Microsoft.ML.AutoML.Samples/BinaryClassificationExperiment.cs)]
    /// ]]></format>
    /// </example>
    public sealed class BinaryClassificationExperiment : ExperimentBase<BinaryClassificationMetrics, BinaryExperimentSettings>
    {
        internal BinaryClassificationExperiment(MLContext context, BinaryExperimentSettings settings)
            : base(context,
                  new BinaryMetricsAgent(context, settings.OptimizingMetric),
                  new OptimizingMetricInfo(settings.OptimizingMetric),
                  settings,
                  TaskKind.BinaryClassification,
                  TrainerExtensionUtil.GetTrainerNames(settings.Trainers))
        {
        }

        private protected override RunDetail<BinaryClassificationMetrics> GetBestRun(IEnumerable<RunDetail<BinaryClassificationMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private protected override CrossValidationRunDetail<BinaryClassificationMetrics> GetBestCrossValRun(IEnumerable<CrossValidationRunDetail<BinaryClassificationMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }
    }
}
