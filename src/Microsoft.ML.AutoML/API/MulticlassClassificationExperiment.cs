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
        private readonly AutoMLExperiment _experiment;
        private const string Features = "__Features__";

        internal MulticlassClassificationExperiment(MLContext context, MulticlassExperimentSettings settings)
            : base(context,
                  new MultiMetricsAgent(context, settings.OptimizingMetric),
                  new OptimizingMetricInfo(settings.OptimizingMetric),
                  settings,
                  TaskKind.MulticlassClassification,
                  TrainerExtensionUtil.GetTrainerNames(settings.Trainers))
        {
            _experiment = context.Auto().CreateExperiment();
        }

        public override ExperimentResult<MulticlassClassificationMetrics> Execute(IDataView trainData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<MulticlassClassificationMetrics>> progressHandler = null)
        {
            var label = columnInformation.LabelColumnName;
            _experiment.SetEvaluateMetric(Settings.OptimizingMetric, label);
            _experiment.SetTrainingTimeInSeconds(Settings.MaxExperimentTimeInSeconds);

            // Cross val threshold for # of dataset rows --
            // If dataset has < threshold # of rows, use cross val.
            // Else, run experiment using train-validate split.
            const int crossValRowCountThreshold = 15000;
            var rowCount = DatasetDimensionsUtil.CountRows(trainData, crossValRowCountThreshold);
            // TODO
            // split cross validation result according to sample key as well.
            if (rowCount < crossValRowCountThreshold)
            {
                const int numCrossValFolds = 10;
                _experiment.SetDataset(trainData, numCrossValFolds);
            }
            else
            {
                var splitData = Context.Data.TrainTestSplit(trainData);
                return this.Execute(splitData.TrainSet, splitData.TestSet, columnInformation, preFeaturizer, progressHandler);
            }

            var pipeline = this.CreateMulticlassClassificationPipeline(trainData, columnInformation, preFeaturizer);
            _experiment.SetPipeline(pipeline);

            var monitor = new TrialResultMonitor<MulticlassClassificationMetrics>(Context);
            monitor.OnTrialCompleted += (o, e) =>
            {
                var detail = BestResultUtil.ToRunDetail(Context, e);
                progressHandler?.Report(detail);
            };

            _experiment.SetMonitor(monitor);
            _experiment.Run();

            var runDetails = monitor.RunDetails.Select(e => BestResultUtil.ToRunDetail(Context, e));
            var bestRun = BestResultUtil.ToRunDetail(Context, monitor.BestRun);
            var result = new ExperimentResult<MulticlassClassificationMetrics>(runDetails, bestRun);

            return result;
        }

        public override ExperimentResult<MulticlassClassificationMetrics> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<MulticlassClassificationMetrics>> progressHandler = null)
        {
            var label = columnInformation.LabelColumnName;
            _experiment.SetEvaluateMetric(Settings.OptimizingMetric, label);
            _experiment.SetTrainingTimeInSeconds(Settings.MaxExperimentTimeInSeconds);
            _experiment.SetDataset(trainData, validationData);

            var pipeline = this.CreateMulticlassClassificationPipeline(trainData, columnInformation, preFeaturizer);
            _experiment.SetPipeline(pipeline);
            var monitor = new TrialResultMonitor<MulticlassClassificationMetrics>(Context);
            monitor.OnTrialCompleted += (o, e) =>
            {
                var detail = BestResultUtil.ToRunDetail(Context, e);
                progressHandler?.Report(detail);
            };

            _experiment.SetMonitor(monitor);
            _experiment.Run();

            var runDetails = monitor.RunDetails.Select(e => BestResultUtil.ToRunDetail(Context, e));
            var bestRun = BestResultUtil.ToRunDetail(Context, monitor.BestRun);
            var result = new ExperimentResult<MulticlassClassificationMetrics>(runDetails, bestRun);

            return result;
        }

        public override ExperimentResult<MulticlassClassificationMetrics> Execute(IDataView trainData, IDataView validationData, string labelColumnName = "Label", IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<MulticlassClassificationMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
            };

            return this.Execute(trainData, validationData, columnInformation, preFeaturizer, progressHandler);
        }

        public override ExperimentResult<MulticlassClassificationMetrics> Execute(IDataView trainData, string labelColumnName = "Label", string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<MulticlassClassificationMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
                SamplingKeyColumnName = samplingKeyColumn,
            };

            return this.Execute(trainData, columnInformation, preFeaturizer, progressHandler);
        }

        public override CrossValidationExperimentResult<MulticlassClassificationMetrics> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<CrossValidationRunDetail<MulticlassClassificationMetrics>> progressHandler = null)
        {
            var label = columnInformation.LabelColumnName;
            _experiment.SetEvaluateMetric(Settings.OptimizingMetric, label);
            _experiment.SetTrainingTimeInSeconds(Settings.MaxExperimentTimeInSeconds);
            _experiment.SetDataset(trainData, (int)numberOfCVFolds);

            var pipeline = this.CreateMulticlassClassificationPipeline(trainData, columnInformation, preFeaturizer);
            _experiment.SetPipeline(pipeline);

            var monitor = new TrialResultMonitor<MulticlassClassificationMetrics>(Context);
            monitor.OnTrialCompleted += (o, e) =>
            {
                var runDetails = BestResultUtil.ToCrossValidationRunDetail(Context, e);

                progressHandler?.Report(runDetails);
            };

            _experiment.SetMonitor(monitor);
            _experiment.Run();

            var runDetails = monitor.RunDetails.Select(e => BestResultUtil.ToCrossValidationRunDetail(Context, e));
            var bestResult = BestResultUtil.ToCrossValidationRunDetail(Context, monitor.BestRun);

            var result = new CrossValidationExperimentResult<MulticlassClassificationMetrics>(runDetails, bestResult);

            return result;
        }

        public override CrossValidationExperimentResult<MulticlassClassificationMetrics> Execute(IDataView trainData, uint numberOfCVFolds, string labelColumnName = "Label", string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<CrossValidationRunDetail<MulticlassClassificationMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
                SamplingKeyColumnName = samplingKeyColumn,
            };

            return this.Execute(trainData, numberOfCVFolds, columnInformation, preFeaturizer, progressHandler);
        }

        private protected override CrossValidationRunDetail<MulticlassClassificationMetrics> GetBestCrossValRun(IEnumerable<CrossValidationRunDetail<MulticlassClassificationMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private protected override RunDetail<MulticlassClassificationMetrics> GetBestRun(IEnumerable<RunDetail<MulticlassClassificationMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private MultiModelPipeline CreateMulticlassClassificationPipeline(IDataView trainData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null)
        {
            var useSdca = this.Settings.Trainers.Contains(MulticlassClassificationTrainer.SdcaMaximumEntropy);
            var uselbfgs = this.Settings.Trainers.Contains(MulticlassClassificationTrainer.LbfgsLogisticRegressionOva);
            var useLgbm = this.Settings.Trainers.Contains(MulticlassClassificationTrainer.LightGbm);
            var useFastForest = this.Settings.Trainers.Contains(MulticlassClassificationTrainer.FastForestOva);
            var useFastTree = this.Settings.Trainers.Contains(MulticlassClassificationTrainer.FastTreeOva);

            MultiModelPipeline pipeline = new MultiModelPipeline();
            if (preFeaturizer != null)
            {
                pipeline = pipeline.Append(preFeaturizer);
            }
            var label = columnInformation.LabelColumnName;


            pipeline = pipeline.Append(Context.Auto().Featurizer(trainData, columnInformation, Features));
            pipeline = pipeline.Append(Context.Transforms.Conversion.MapValueToKey(label, label));
            pipeline = pipeline.Append(Context.Auto().MultiClassification(label, useSdca: useSdca, useFastTree: useFastTree, useLgbm: useLgbm, useLbfgs: uselbfgs, useFastForest: useFastForest, featureColumnName: Features));
            pipeline = pipeline.Append(Context.Transforms.Conversion.MapKeyToValue(label, label));

            return pipeline;
        }
    }
}
