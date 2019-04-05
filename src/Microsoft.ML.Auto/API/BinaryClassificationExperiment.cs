// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public sealed class BinaryExperimentSettings : ExperimentSettings
    {
        public BinaryClassificationMetric OptimizingMetric { get; set; } = BinaryClassificationMetric.Accuracy;
        public ICollection<BinaryClassificationTrainer> Trainers { get; } =
                    Enum.GetValues(typeof(BinaryClassificationTrainer)).OfType<BinaryClassificationTrainer>().ToList();
    }

    public enum BinaryClassificationMetric
    {
        Accuracy,
        AreaUnderRocCurve,
        AreaUnderPrecisionRecallCurve,
        F1Score,
        PositivePrecision,
        PositiveRecall,
        NegativePrecision,
        NegativeRecall,
    }

    public enum BinaryClassificationTrainer
    {
        AveragedPerceptron,
        FastForest,
        FastTree,
        LightGbm,
        LinearSupportVectorMachines,
        LbfgsLogisticRegression,
        SdcaLogisticRegression,
        SgdCalibrated,
        SymbolicSgdLogisticRegression,
    }

    public sealed class BinaryClassificationExperiment : ExperimentBase<BinaryClassificationMetrics>
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
    }

    public static class BinaryExperimentResultExtensions
    {
        public static RunDetail<BinaryClassificationMetrics> Best(this IEnumerable<RunDetail<BinaryClassificationMetrics>> results, BinaryClassificationMetric metric = BinaryClassificationMetric.Accuracy)
        {
            var metricsAgent = new BinaryMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }

        public static CrossValidationRunDetail<BinaryClassificationMetrics> Best(this IEnumerable<CrossValidationRunDetail<BinaryClassificationMetrics>> results, BinaryClassificationMetric metric = BinaryClassificationMetric.Accuracy)
        {
            var metricsAgent = new BinaryMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }
    }
}
