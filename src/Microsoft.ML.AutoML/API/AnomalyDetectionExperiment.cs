// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.AutoML.Experiment.MetricsAgents;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.AutoML
{
    public sealed class AnomalyExperimentSettings : ExperimentSettings
    {
        public AnomalyDetectionMetric OptimizingMetric { get; set; }

        public ICollection<AnomalyDetectionTrainer> Trainers { get; }

        public AnomalyExperimentSettings()
        {
            OptimizingMetric = AnomalyDetectionMetric.FakeAccuracy;
            Trainers = Enum.GetValues(typeof(AnomalyDetectionTrainer)).OfType<AnomalyDetectionTrainer>().ToList();
        }
    }

    public class AnomalyDetectionMetrics
    {
        public double FakeAccuracy { get; }

        public AnomalyDetectionMetrics()
        {
            FakeAccuracy = 1;
        }
    }

    /// <summary>
    /// Binary classification metric that AutoML will aim to optimize in its sweeping process during an experiment.
    /// </summary>
    public enum AnomalyDetectionMetric
    {
        /// <summary>
        /// See <see cref="AnomalyDetectionMetrics.FakeAccuracy"/>.
        /// </summary>
        FakeAccuracy
    }

    /// <summary>
    /// Enumeration of ML.NET binary classification trainers used by AutoML.
    /// </summary>
    public enum AnomalyDetectionTrainer
    {
        /// <summary>
        /// See <see cref="RandomizedPcaTrainer"/>.
        /// </summary>
        RandomizedPca
    }

    /// <summary>
    /// AutoML experiment on binary classification datasets.
    /// </summary>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    ///  [!code-csharp[AnomalyDetectionExperiment](~/../docs/samples/docs/samples/Microsoft.ML.AutoML.Samples/AnomalyDetectionExperiment.cs)]
    /// ]]></format>
    /// </example>
    public sealed class AnomalyDetectionExperiment : ExperimentBase<AnomalyDetectionMetrics, AnomalyExperimentSettings>
    {
        internal AnomalyDetectionExperiment(MLContext context, AnomalyExperimentSettings settings)
            : base(context,
                  new AnomalyMetricsAgent(context, settings.OptimizingMetric),
                  new OptimizingMetricInfo(settings.OptimizingMetric),
                  settings,
                  TaskKind.Anomaly,
                  TrainerExtensionUtil.GetTrainerNames(settings.Trainers))
        {
        }

        private protected override RunDetail<AnomalyDetectionMetrics> GetBestRun(IEnumerable<RunDetail<AnomalyDetectionMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private protected override CrossValidationRunDetail<AnomalyDetectionMetrics> GetBestCrossValRun(IEnumerable<CrossValidationRunDetail<AnomalyDetectionMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }
    }
}
