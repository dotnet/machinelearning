// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using static Microsoft.ML.TrainCatalogBase;

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
        private SweepablePipeline _pipeline;

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

            if (preFeaturizer != null)
            {
                _pipeline = preFeaturizer.Append(Context.Auto().Featurizer(trainData, columnInformation, Features))
                                        .Append(Context.Auto().BinaryClassification(label, Features));
            }
            else
            {
                _pipeline = Context.Auto().Featurizer(trainData, columnInformation, Features)
                   .Append(Context.Auto().BinaryClassification(label, Features));
            }

            _experiment.SetPipeline(_pipeline);

            var monitor = new BinaryClassificationTrialResultMonitor();
            monitor.OnTrialCompleted += (o, e) =>
            {
                var detail = ToRunDetail(e);
                progressHandler?.Report(detail);
            };

            _experiment.SetTrialRunnerFactory<BinaryExperimentTrialRunnerFactory>();
            _experiment.SetHyperParameterProposer<EciHyperParameterProposer>();
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

            if (preFeaturizer != null)
            {
                _pipeline = preFeaturizer.Append(Context.Auto().Featurizer(trainData, columnInformation, Features))
                                        .Append(Context.Auto().BinaryClassification(label, Features));
            }
            else
            {
                _pipeline = Context.Auto().Featurizer(trainData, columnInformation, Features)
                   .Append(Context.Auto().BinaryClassification(label, Features));
            }

            _experiment.SetPipeline(_pipeline);
            var monitor = new BinaryClassificationTrialResultMonitor();
            monitor.OnTrialCompleted += (o, e) =>
            {
                var detail = ToRunDetail(e);
                progressHandler?.Report(detail);
            };

            _experiment.SetMonitor(monitor);
            _experiment.SetTrialRunnerFactory<BinaryExperimentTrialRunnerFactory>();
            _experiment.SetHyperParameterProposer<EciHyperParameterProposer>();
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

            if (preFeaturizer != null)
            {
                _pipeline = preFeaturizer.Append(Context.Auto().Featurizer(trainData, columnInformation, Features))
                                        .Append(Context.Auto().BinaryClassification(label, Features));
            }
            else
            {
                _pipeline = Context.Auto().Featurizer(trainData, columnInformation, Features)
                   .Append(Context.Auto().BinaryClassification(label, Features));
            }

            _experiment.SetPipeline(_pipeline);

            var monitor = new BinaryClassificationTrialResultMonitor();
            monitor.OnTrialCompleted += (o, e) =>
            {
                var runDetails = ToCrossValidationRunDetail(e);

                progressHandler?.Report(runDetails);
            };

            _experiment.SetMonitor(monitor);
            _experiment.SetTrialRunnerFactory<BinaryExperimentTrialRunnerFactory>();
            _experiment.SetHyperParameterProposer<EciHyperParameterProposer>();
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
            var trainerName = result.TrialSettings.Pipeline.ToString();
            var modelContainer = new ModelContainer(Context, result.Model);
            return new RunDetail<BinaryClassificationMetrics>(trainerName, result.Pipeline, null, modelContainer, result.BinaryClassificationMetrics, result.Exception);
        }

        private CrossValidationRunDetail<BinaryClassificationMetrics> ToCrossValidationRunDetail(BinaryClassificationTrialResult result)
        {
            var trainerName = result.TrialSettings.Pipeline.ToString();
            var crossValidationResult = result.CrossValidationMetrics.Select(m => new TrainResult<BinaryClassificationMetrics>(new ModelContainer(Context, m.Model), m.Metrics, result.Exception));
            return new CrossValidationRunDetail<BinaryClassificationMetrics>(trainerName, result.Pipeline, null, crossValidationResult);
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

    internal class BinaryClassificationCVRunner : ITrialRunner
    {
        private readonly MLContext _context;
        private readonly IDatasetManager _datasetManager;
        private readonly IMetricManager _metricManager;

        private readonly EciHyperParameterProposer _proposer;

        public BinaryClassificationCVRunner(MLContext context, IDatasetManager datasetManager, IMetricManager metricManager, EciHyperParameterProposer proposer)
        {
            _context = context;
            _datasetManager = datasetManager;
            _metricManager = metricManager;
            _proposer = proposer;
        }

        public TrialResult Run(TrialSettings settings, IServiceProvider provider)
        {
            var rnd = new Random(settings.ExperimentSettings.Seed ?? 0);
            if (_datasetManager is CrossValidateDatasetManager datasetSettings
                && _metricManager is BinaryMetricManager metricSettings)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();
                var fold = datasetSettings.Fold ?? 5;
                var pipeline = settings.Pipeline.BuildTrainingPipeline(_context, settings.Parameter);
                var metrics = _context.BinaryClassification.CrossValidateNonCalibrated(datasetSettings.Dataset, pipeline, fold, metricSettings.LabelColumn);

                // now we just randomly pick a model, but a better way is to provide option to pick a model which score is the cloest to average or the best.
                var res = metrics[rnd.Next(fold)];
                var model = res.Model;
                var metric = metricSettings.Metric switch
                {
                    BinaryClassificationMetric.PositivePrecision => res.Metrics.PositivePrecision,
                    BinaryClassificationMetric.Accuracy => res.Metrics.Accuracy,
                    BinaryClassificationMetric.AreaUnderRocCurve => res.Metrics.AreaUnderRocCurve,
                    BinaryClassificationMetric.AreaUnderPrecisionRecallCurve => res.Metrics.AreaUnderPrecisionRecallCurve,
                    _ => throw new NotImplementedException($"{metricSettings.Metric} is not supported!"),
                };

                stopWatch.Stop();


                return new BinaryClassificationTrialResult()
                {
                    Metric = metric,
                    Model = model,
                    TrialSettings = settings,
                    DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                    BinaryClassificationMetrics = res.Metrics,
                    CrossValidationMetrics = metrics,
                    Pipeline = pipeline,
                };
            }

            throw new ArgumentException();
        }
    }


    internal class BinaryClassificationTrainTestRunner : ITrialRunner
    {
        private readonly MLContext _context;
        private readonly IDatasetManager _datasetManager;
        private readonly IMetricManager _metricManager;
        private readonly EciHyperParameterProposer _proposer;

        public BinaryClassificationTrainTestRunner(MLContext context, IDatasetManager datasetManager, IMetricManager metricManager, EciHyperParameterProposer proposer)
        {
            _context = context;
            _metricManager = metricManager;
            _datasetManager = datasetManager;
            _proposer = proposer;
        }

        public TrialResult Run(TrialSettings settings, IServiceProvider provider)
        {
            if (_datasetManager is TrainTestDatasetManager datasetSettings
                && _metricManager is BinaryMetricManager metricSettings)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();
                var pipeline = settings.Pipeline.BuildTrainingPipeline(_context, settings.Parameter);
                var model = pipeline.Fit(datasetSettings.TrainDataset);
                var eval = model.Transform(datasetSettings.TestDataset);
                var metrics = _context.BinaryClassification.EvaluateNonCalibrated(eval, metricSettings.LabelColumn, predictedLabelColumnName: metricSettings.PredictedColumn);

                // now we just randomly pick a model, but a better way is to provide option to pick a model which score is the cloest to average or the best.
                var metric = metricSettings.Metric switch
                {
                    BinaryClassificationMetric.PositivePrecision => metrics.PositivePrecision,
                    BinaryClassificationMetric.Accuracy => metrics.Accuracy,
                    BinaryClassificationMetric.AreaUnderRocCurve => metrics.AreaUnderRocCurve,
                    BinaryClassificationMetric.AreaUnderPrecisionRecallCurve => metrics.AreaUnderPrecisionRecallCurve,
                    _ => throw new NotImplementedException($"{metricSettings.Metric} is not supported!"),
                };

                stopWatch.Stop();


                return new BinaryClassificationTrialResult()
                {
                    Metric = metric,
                    Model = model,
                    TrialSettings = settings,
                    DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                    BinaryClassificationMetrics = metrics,
                    Pipeline = pipeline,
                };
            }

            throw new ArgumentException();
        }
    }


    internal class BinaryExperimentTrialRunnerFactory : ITrialRunnerFactory
    {
        private readonly IServiceProvider _provider;

        public BinaryExperimentTrialRunnerFactory(IServiceProvider provider)
        {
            _provider = provider;
        }

        public ITrialRunner CreateTrialRunner()
        {
            var datasetManager = _provider.GetService<IDatasetManager>();
            var metricManager = _provider.GetService<IMetricManager>();

            ITrialRunner runner = (datasetManager, metricManager) switch
            {
                (CrossValidateDatasetManager, BinaryMetricManager) => _provider.GetService<BinaryClassificationCVRunner>(),
                (TrainTestDatasetManager, BinaryMetricManager) => _provider.GetService<BinaryClassificationTrainTestRunner>(),
                _ => throw new NotImplementedException(),
            };

            return runner;
        }
    }

    /// <summary>
    /// TrialResult with Binary Classification Metrics
    /// </summary>
    internal class BinaryClassificationTrialResult : TrialResult
    {
        public BinaryClassificationMetrics BinaryClassificationMetrics { get; set; }

        public IEnumerable<CrossValidationResult<BinaryClassificationMetrics>> CrossValidationMetrics { get; set; }

        public Exception Exception { get; set; }

        public bool IsSucceed { get => Exception == null; }

        public bool IsCrossValidation { get => CrossValidationMetrics == null; }

        public EstimatorChain<ITransformer> Pipeline { get; set; }
    }
}
