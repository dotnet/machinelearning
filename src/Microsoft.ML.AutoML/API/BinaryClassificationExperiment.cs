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
        private readonly AutoMLExperiment _experiment;
        private const string Features = "__Features__";

        internal BinaryClassificationExperiment(MLContext context, BinaryExperimentSettings settings)
            : base(context,
                  new BinaryMetricsAgent(context, settings.OptimizingMetric),
                  new OptimizingMetricInfo(settings.OptimizingMetric),
                  settings,
                  TaskKind.BinaryClassification,
                  TrainerExtensionUtil.GetTrainerNames(settings.Trainers))
        {
            _experiment = context.Auto().CreateExperiment();
        }

        public override ExperimentResult<BinaryClassificationMetrics> Execute(IDataView trainData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<BinaryClassificationMetrics>> progressHandler = null)
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
                _experiment.SetDataset(splitData.TrainSet, splitData.TestSet);
            }

            MultiModelPipeline pipeline = new MultiModelPipeline();
            if (preFeaturizer != null)
            {
                pipeline = pipeline.Append(preFeaturizer);
            }

            pipeline = pipeline.Append(Context.Auto().Featurizer(trainData, columnInformation, Features))
                               .Append(Context.Auto().BinaryClassification(label, Features));
            _experiment.SetPipeline(pipeline);

            var monitor = new BinaryClassificationTrialResultMonitor();
            monitor.OnTrialCompleted += (o, e) =>
            {
                var detail = ToRunDetail(e);
                progressHandler?.Report(detail);
            };

            _experiment.SetMonitor(monitor);
            _experiment.Run();

            var runDetails = monitor.RunDetails.Select(e => ToRunDetail(e));
            var bestRun = ToRunDetail(monitor.BestRun);
            var result = new ExperimentResult<BinaryClassificationMetrics>(runDetails, bestRun);

            return result;
        }

        public override ExperimentResult<BinaryClassificationMetrics> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<BinaryClassificationMetrics>> progressHandler = null)
        {
            var label = columnInformation.LabelColumnName;
            _experiment.SetEvaluateMetric(Settings.OptimizingMetric, label);
            _experiment.SetTrainingTimeInSeconds(Settings.MaxExperimentTimeInSeconds);
            _experiment.SetDataset(trainData, validationData);

            MultiModelPipeline pipeline = new MultiModelPipeline();
            if (preFeaturizer != null)
            {
                pipeline = pipeline.Append(preFeaturizer);
            }

            pipeline = pipeline.Append(Context.Auto().Featurizer(trainData, columnInformation, "__Features__"))
                               .Append(Context.Auto().BinaryClassification(label, featureColumnName: Features));

            _experiment.SetPipeline(pipeline);
            var monitor = new BinaryClassificationTrialResultMonitor();
            monitor.OnTrialCompleted += (o, e) =>
            {
                var detail = ToRunDetail(e);
                progressHandler?.Report(detail);
            };

            _experiment.SetMonitor(monitor);
            _experiment.Run();

            var runDetails = monitor.RunDetails.Select(e => ToRunDetail(e));
            var bestRun = ToRunDetail(monitor.BestRun);
            var result = new ExperimentResult<BinaryClassificationMetrics>(runDetails, bestRun);

            return result;
        }

        public override ExperimentResult<BinaryClassificationMetrics> Execute(IDataView trainData, IDataView validationData, string labelColumnName = "Label", IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<BinaryClassificationMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
            };

            return this.Execute(trainData, validationData, columnInformation, preFeaturizer, progressHandler);
        }

        public override ExperimentResult<BinaryClassificationMetrics> Execute(IDataView trainData, string labelColumnName = "Label", string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<BinaryClassificationMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
                SamplingKeyColumnName = samplingKeyColumn,
            };

            return this.Execute(trainData, columnInformation, preFeaturizer, progressHandler);
        }

        public override CrossValidationExperimentResult<BinaryClassificationMetrics> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<CrossValidationRunDetail<BinaryClassificationMetrics>> progressHandler = null)
        {
            var label = columnInformation.LabelColumnName;
            _experiment.SetEvaluateMetric(Settings.OptimizingMetric, label);
            _experiment.SetTrainingTimeInSeconds(Settings.MaxExperimentTimeInSeconds);
            _experiment.SetDataset(trainData, (int)numberOfCVFolds);

            MultiModelPipeline pipeline = new MultiModelPipeline();
            if (preFeaturizer != null)
            {
                pipeline = pipeline.Append(preFeaturizer);
            }

            pipeline = pipeline.Append(Context.Auto().Featurizer(trainData, columnInformation, "__Features__"))
                               .Append(Context.Auto().BinaryClassification(label, featureColumnName: Features));

            _experiment.SetPipeline(pipeline);

            var monitor = new BinaryClassificationTrialResultMonitor();
            monitor.OnTrialCompleted += (o, e) =>
            {
                var runDetails = ToCrossValidationRunDetail(e);

                progressHandler?.Report(runDetails);
            };

            _experiment.SetMonitor(monitor);
            _experiment.Run();

            var runDetails = monitor.RunDetails.Select(e => ToCrossValidationRunDetail(e));
            var bestResult = ToCrossValidationRunDetail(monitor.BestRun);

            var result = new CrossValidationExperimentResult<BinaryClassificationMetrics>(runDetails, bestResult);

            return result;
        }

        public override CrossValidationExperimentResult<BinaryClassificationMetrics> Execute(IDataView trainData, uint numberOfCVFolds, string labelColumnName = "Label", string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<CrossValidationRunDetail<BinaryClassificationMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
                SamplingKeyColumnName = samplingKeyColumn,
            };

            return this.Execute(trainData, numberOfCVFolds, columnInformation, preFeaturizer, progressHandler);
        }

        private protected override RunDetail<BinaryClassificationMetrics> GetBestRun(IEnumerable<RunDetail<BinaryClassificationMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private protected override CrossValidationRunDetail<BinaryClassificationMetrics> GetBestCrossValRun(IEnumerable<CrossValidationRunDetail<BinaryClassificationMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private RunDetail<BinaryClassificationMetrics> ToRunDetail(BinaryClassificationTrialResult result)
        {
            var pipeline = result.TrialSettings.Pipeline;
            var trainerName = pipeline.ToString();
            var parameter = result.TrialSettings.Parameter;
            var estimator = pipeline.BuildTrainingPipeline(Context, parameter);
            var modelContainer = new ModelContainer(Context, result.Model);
            return new RunDetail<BinaryClassificationMetrics>(trainerName, estimator, null, modelContainer, result.BinaryClassificationMetrics, result.Exception);
        }

        private CrossValidationRunDetail<BinaryClassificationMetrics> ToCrossValidationRunDetail(BinaryClassificationTrialResult result)
        {
            var pipeline = result.TrialSettings.Pipeline;
            var trainerName = pipeline.ToString();
            var parameter = result.TrialSettings.Parameter;
            var estimator = pipeline.BuildTrainingPipeline(Context, parameter);
            var crossValidationResult = result.CrossValidationMetrics.Select(m => new TrainResult<BinaryClassificationMetrics>(new ModelContainer(Context, m.Model), m.Metrics, result.Exception));
            return new CrossValidationRunDetail<BinaryClassificationMetrics>(trainerName, estimator, null, crossValidationResult);
        }
    }

    internal class BinaryClassificationTrialResultMonitor : IMonitor
    {
        public BinaryClassificationTrialResultMonitor()
        {
            this.RunDetails = new List<BinaryClassificationTrialResult>();
        }

        public event EventHandler<BinaryClassificationTrialResult> OnTrialCompleted;

        public List<BinaryClassificationTrialResult> RunDetails { get; }

        public BinaryClassificationTrialResult BestRun { get; private set; }

        public void ReportBestTrial(TrialResult result)
        {
            if (result is BinaryClassificationTrialResult binaryClassificationResult)
            {
                BestRun = binaryClassificationResult;
            }
            else
            {
                throw new ArgumentException($"result must be of type {typeof(BinaryClassificationTrialResult)}");
            }
        }

        public void ReportCompletedTrial(TrialResult result)
        {
            if (result is BinaryClassificationTrialResult binaryClassificationResult)
            {
                RunDetails.Add(binaryClassificationResult);
                OnTrialCompleted?.Invoke(this, binaryClassificationResult);
            }
            else
            {
                throw new ArgumentException($"result must be of type {typeof(BinaryClassificationTrialResult)}");
            }
        }

        public void ReportFailTrial(TrialSettings settings, Exception exp)
        {
            var result = new BinaryClassificationTrialResult
            {
                TrialSettings = settings,
                Exception = exp,
            };

            RunDetails.Add(result);
        }

        public void ReportRunningTrial(TrialSettings setting)
        {
        }
    }
}
