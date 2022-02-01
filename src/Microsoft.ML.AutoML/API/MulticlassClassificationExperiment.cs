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
    /// Settings for AutoML experiments on multiclass classification datasets.
    /// </summary>
    public sealed class MulticlassExperimentSettings : ExperimentSettings
    {
        /// <summary>
        /// Metric that AutoML will try to optimize over the course of the experiment.
        /// </summary>
        /// <value>The default value is <see cref="MulticlassClassificationMetric.MicroAccuracy"/>.</value>
        public MulticlassClassificationMetric OptimizingMetric { get; set; }

        /// <summary>
        /// Collection of trainers the AutoML experiment can leverage.
        /// </summary>
        /// <value>
        /// The default value is a collection auto-populated with all possible trainers (all values of <see cref="MulticlassClassificationTrainer" />).
        /// </value>
        public ICollection<MulticlassClassificationTrainer> Trainers { get; }

        /// <summary>
        /// Initializes a new instances of <see cref="MulticlassExperimentSettings"/>.
        /// </summary>
        public MulticlassExperimentSettings()
        {
            OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy;
            Trainers = Enum.GetValues(typeof(MulticlassClassificationTrainer)).OfType<MulticlassClassificationTrainer>().ToList();
        }
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
        AveragedPerceptronOva,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="FastForestBinaryTrainer"/>.
        /// </summary>
        FastForestOva,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="FastTreeBinaryTrainer"/>.
        /// </summary>
        FastTreeOva,

        /// <summary>
        /// See <see cref="LightGbmMulticlassTrainer"/>.
        /// </summary>
        LightGbm,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="LinearSvmTrainer"/>.
        /// </summary>
        LinearSupportVectorMachinesOva,

        /// <summary>
        /// See <see cref="LbfgsMaximumEntropyMulticlassTrainer"/>.
        /// </summary>
        LbfgsMaximumEntropy,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="LbfgsLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        LbfgsLogisticRegressionOva,

        /// <summary>
        /// See <see cref="SdcaMaximumEntropyMulticlassTrainer"/>.
        /// </summary>
        SdcaMaximumEntropy,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="LbfgsLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        SgdCalibratedOva,

        /// <summary>
        /// <see cref="OneVersusAllTrainer"/> using <see cref="SymbolicSgdLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        SymbolicSgdLogisticRegressionOva,
    }

    /// <summary>
    /// AutoML experiment on multiclass classification datasets.
    /// </summary>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    ///  [!code-csharp[MulticlassClassificationExperiment](~/../docs/samples/docs/samples/Microsoft.ML.AutoML.Samples/MulticlassClassificationExperiment.cs)]
    /// ]]></format>
    /// </example>
    public sealed class MulticlassClassificationExperiment : ExperimentBase<MulticlassClassificationMetrics, MulticlassExperimentSettings>
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

        private protected override CrossValidationRunDetail<MulticlassClassificationMetrics> GetBestCrossValRun(IEnumerable<CrossValidationRunDetail<MulticlassClassificationMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private protected override RunDetail<MulticlassClassificationMetrics> GetBestRun(IEnumerable<RunDetail<MulticlassClassificationMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }
    }
}
