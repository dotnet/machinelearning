// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public sealed class MulticlassExperimentSettings : ExperimentSettings
    {
        public MulticlassClassificationMetric OptimizingMetric { get; set; } = MulticlassClassificationMetric.MicroAccuracy;
        public ICollection<MulticlassClassificationTrainer> Trainers { get; } =
            Enum.GetValues(typeof(MulticlassClassificationTrainer)).OfType<MulticlassClassificationTrainer>().ToList();
    }

    public enum MulticlassClassificationMetric
    {
        MicroAccuracy,
        MacroAccuracy,
        LogLoss,
        LogLossReduction,
        TopKAccuracy,
    }

    public enum MulticlassClassificationTrainer
    {
        AveragedPerceptronOVA,
        FastForestOVA,
        FastTreeOVA,
        LightGbm,
        LinearSupportVectorMachinesOVA,
        LbfgsMaximumEntropy,
        LbfgsLogisticRegressionOVA,
        SdcaMaximumEntropy,
        SgdCalibratedOVA,
        SymbolicSgdLogisticRegressionOVA,
    }

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

    public static class MulticlassExperimentResultExtensions
    {
        public static RunDetail<MulticlassClassificationMetrics> Best(this IEnumerable<RunDetail<MulticlassClassificationMetrics>> results, MulticlassClassificationMetric metric = MulticlassClassificationMetric.MicroAccuracy)
        {
            var metricsAgent = new MultiMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }

        public static CrossValidationRunDetail<MulticlassClassificationMetrics> Best(this IEnumerable<CrossValidationRunDetail<MulticlassClassificationMetrics>> results, MulticlassClassificationMetric metric = MulticlassClassificationMetric.MicroAccuracy)
        {
            var metricsAgent = new MultiMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }
    }
}