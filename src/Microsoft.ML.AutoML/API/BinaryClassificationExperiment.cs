// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
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
        /// See <see cref="LbfgsLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        LbfgsLogisticRegression,

        /// <summary>
        /// See <see cref="SdcaLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        SdcaLogisticRegression,
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
            if (settings.MaximumMemoryUsageInMegaByte is double d)
            {
                _experiment.SetMaximumMemoryUsageInMegaByte(d);
            }
            _experiment.SetMaxModelToExplore(settings.MaxModels);
        }

        public override ExperimentResult<BinaryClassificationMetrics> Execute(IDataView trainData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<BinaryClassificationMetrics>> progressHandler = null)
        {
            var label = columnInformation.LabelColumnName;
            _experiment.SetBinaryClassificationMetric(Settings.OptimizingMetric, label);
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
            _pipeline = CreateBinaryClassificationPipeline(trainData, columnInformation, preFeaturizer);
            _experiment.SetPipeline(_pipeline);

            // set monitor
            TrialResultMonitor<BinaryClassificationMetrics> monitor = null;
            _experiment.SetMonitor((provider) =>
            {
                var channel = provider.GetService<IChannel>();
                var pipeline = provider.GetService<SweepablePipeline>();
                monitor = new TrialResultMonitor<BinaryClassificationMetrics>(channel, pipeline);
                monitor.OnTrialCompleted += (o, e) =>
                {
                    var detail = BestResultUtil.ToRunDetail(Context, e, _pipeline);
                    progressHandler?.Report(detail);
                };

                return monitor;
            });
            _experiment.SetTrialRunner<BinaryClassificationRunner>();
            _experiment.Run();

            var runDetails = monitor.RunDetails.Select(e => BestResultUtil.ToRunDetail(Context, e, _pipeline));
            var bestRun = BestResultUtil.ToRunDetail(Context, monitor.BestRun, _pipeline);
            var result = new ExperimentResult<BinaryClassificationMetrics>(runDetails, bestRun);

            return result;
        }

        public override ExperimentResult<BinaryClassificationMetrics> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<BinaryClassificationMetrics>> progressHandler = null)
        {
            var label = columnInformation.LabelColumnName;
            _experiment.SetBinaryClassificationMetric(Settings.OptimizingMetric, label);
            _experiment.SetTrainingTimeInSeconds(Settings.MaxExperimentTimeInSeconds);
            _experiment.SetDataset(trainData, validationData);
            _pipeline = CreateBinaryClassificationPipeline(trainData, columnInformation, preFeaturizer);
            _experiment.SetPipeline(_pipeline);

            // set monitor
            TrialResultMonitor<BinaryClassificationMetrics> monitor = null;
            _experiment.SetMonitor((provider) =>
            {
                var channel = provider.GetService<IChannel>();
                var pipeline = provider.GetService<SweepablePipeline>();
                monitor = new TrialResultMonitor<BinaryClassificationMetrics>(channel, pipeline);
                monitor.OnTrialCompleted += (o, e) =>
                {
                    var detail = BestResultUtil.ToRunDetail(Context, e, _pipeline);
                    progressHandler?.Report(detail);
                };

                return monitor;
            });
            _experiment.SetTrialRunner<BinaryClassificationRunner>();
            _experiment.Run();

            var runDetails = monitor.RunDetails.Select(e => BestResultUtil.ToRunDetail(Context, e, _pipeline));
            var bestRun = BestResultUtil.ToRunDetail(Context, monitor.BestRun, _pipeline);
            var result = new ExperimentResult<BinaryClassificationMetrics>(runDetails, bestRun);

            return result;
        }

        public override ExperimentResult<BinaryClassificationMetrics> Execute(IDataView trainData, IDataView validationData, string labelColumnName = "Label", IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<BinaryClassificationMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
            };

            return Execute(trainData, validationData, columnInformation, preFeaturizer, progressHandler);
        }

        public override ExperimentResult<BinaryClassificationMetrics> Execute(IDataView trainData, string labelColumnName = "Label", string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<BinaryClassificationMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
                SamplingKeyColumnName = samplingKeyColumn,
            };

            return Execute(trainData, columnInformation, preFeaturizer, progressHandler);
        }

        public override CrossValidationExperimentResult<BinaryClassificationMetrics> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<CrossValidationRunDetail<BinaryClassificationMetrics>> progressHandler = null)
        {
            var label = columnInformation.LabelColumnName;
            _experiment.SetBinaryClassificationMetric(Settings.OptimizingMetric, label);
            _experiment.SetTrainingTimeInSeconds(Settings.MaxExperimentTimeInSeconds);
            _experiment.SetDataset(trainData, (int)numberOfCVFolds);
            _pipeline = CreateBinaryClassificationPipeline(trainData, columnInformation, preFeaturizer);
            _experiment.SetPipeline(_pipeline);

            // set monitor
            TrialResultMonitor<BinaryClassificationMetrics> monitor = null;
            _experiment.SetMonitor((provider) =>
            {
                var channel = provider.GetService<IChannel>();
                var pipeline = provider.GetService<SweepablePipeline>();
                monitor = new TrialResultMonitor<BinaryClassificationMetrics>(channel, pipeline);
                monitor.OnTrialCompleted += (o, e) =>
                {
                    var detail = BestResultUtil.ToCrossValidationRunDetail(Context, e, _pipeline);
                    progressHandler?.Report(detail);
                };

                return monitor;
            });

            _experiment.SetTrialRunner<BinaryClassificationRunner>();
            _experiment.Run();

            var runDetails = monitor.RunDetails.Select(e => BestResultUtil.ToCrossValidationRunDetail(Context, e, _pipeline));
            var bestResult = BestResultUtil.ToCrossValidationRunDetail(Context, monitor.BestRun, _pipeline);

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

            return Execute(trainData, numberOfCVFolds, columnInformation, preFeaturizer, progressHandler);
        }

        private protected override RunDetail<BinaryClassificationMetrics> GetBestRun(IEnumerable<RunDetail<BinaryClassificationMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private protected override CrossValidationRunDetail<BinaryClassificationMetrics> GetBestCrossValRun(IEnumerable<CrossValidationRunDetail<BinaryClassificationMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private SweepablePipeline CreateBinaryClassificationPipeline(IDataView trainData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null)
        {
            var useSdca = Settings.Trainers.Contains(BinaryClassificationTrainer.SdcaLogisticRegression);
            var uselbfgs = Settings.Trainers.Contains(BinaryClassificationTrainer.LbfgsLogisticRegression);
            var useLgbm = Settings.Trainers.Contains(BinaryClassificationTrainer.LightGbm);
            var useFastForest = Settings.Trainers.Contains(BinaryClassificationTrainer.FastForest);
            var useFastTree = Settings.Trainers.Contains(BinaryClassificationTrainer.FastTree);

            if (preFeaturizer != null)
            {
                return preFeaturizer.Append(Context.Auto().Featurizer(trainData, columnInformation, Features))
                                        .Append(Context.Auto().BinaryClassification(labelColumnName: columnInformation.LabelColumnName, useSdca: useSdca, useFastTree: useFastTree, useLgbm: useLgbm, useLbfgs: uselbfgs, useFastForest: useFastForest, featureColumnName: Features));
            }
            else
            {
                return Context.Auto().Featurizer(trainData, columnInformation, Features)
                           .Append(Context.Auto().BinaryClassification(labelColumnName: columnInformation.LabelColumnName, useSdca: useSdca, useFastTree: useFastTree, useLgbm: useLgbm, useLbfgs: uselbfgs, useFastForest: useFastForest, featureColumnName: Features));
            }
        }
    }

    internal class BinaryClassificationRunner : ITrialRunner
    {
        private MLContext _context;
        private readonly IDatasetManager _datasetManager;
        private readonly IMetricManager _metricManager;
        private readonly SweepablePipeline _pipeline;
        private readonly Random _rnd;
        public BinaryClassificationRunner(MLContext context, IDatasetManager datasetManager, IMetricManager metricManager, SweepablePipeline pipeline, AutoMLExperiment.AutoMLExperimentSettings settings)
        {
            _context = context;
            _datasetManager = datasetManager;
            _metricManager = metricManager;
            _pipeline = pipeline;
            _rnd = settings.Seed.HasValue ? new Random(settings.Seed.Value) : new Random();
        }

        public void Dispose()
        {
            _context.CancelExecution();
            _context = null;
        }

        public TrialResult Run(TrialSettings settings)
        {
            if (_metricManager is BinaryMetricManager metricManager)
            {
                var parameter = settings.Parameter[AutoMLExperiment.PipelineSearchspaceName];
                var pipeline = _pipeline.BuildFromOption(_context, parameter);
                if (_datasetManager is ICrossValidateDatasetManager datasetManager)
                {
                    var stopWatch = new Stopwatch();
                    stopWatch.Start();
                    var fold = datasetManager.Fold ?? 5;
                    var metrics = _context.BinaryClassification.CrossValidateNonCalibrated(datasetManager.Dataset, pipeline, fold, metricManager.LabelColumn);

                    // now we just randomly pick a model, but a better way is to provide option to pick a model which score is the cloest to average or the best.
                    var res = metrics[_rnd.Next(fold)];
                    var model = res.Model;
                    var metric = metricManager.Metric switch
                    {
                        BinaryClassificationMetric.PositivePrecision => res.Metrics.PositivePrecision,
                        BinaryClassificationMetric.Accuracy => res.Metrics.Accuracy,
                        BinaryClassificationMetric.AreaUnderRocCurve => res.Metrics.AreaUnderRocCurve,
                        BinaryClassificationMetric.AreaUnderPrecisionRecallCurve => res.Metrics.AreaUnderPrecisionRecallCurve,
                        _ => throw new NotImplementedException($"{metricManager.MetricName} is not supported!"),
                    };
                    var loss = metricManager.IsMaximize ? -metric : metric;
                    stopWatch.Stop();


                    return new TrialResult<BinaryClassificationMetrics>()
                    {
                        Loss = loss,
                        Metric = metric,
                        Model = model,
                        TrialSettings = settings,
                        DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                        Metrics = res.Metrics,
                        CrossValidationMetrics = metrics,
                        Pipeline = pipeline,
                    };
                }

                if (_datasetManager is ITrainTestDatasetManager trainTestDatasetManager)
                {
                    var stopWatch = new Stopwatch();
                    stopWatch.Start();
                    var model = pipeline.Fit(trainTestDatasetManager.TrainDataset);
                    var eval = model.Transform(trainTestDatasetManager.TestDataset);
                    var metrics = _context.BinaryClassification.EvaluateNonCalibrated(eval, metricManager.LabelColumn, predictedLabelColumnName: metricManager.PredictedColumn);

                    // now we just randomly pick a model, but a better way is to provide option to pick a model which score is the cloest to average or the best.
                    var metric = Enum.Parse(typeof(BinaryClassificationMetric), metricManager.MetricName) switch
                    {
                        BinaryClassificationMetric.PositivePrecision => metrics.PositivePrecision,
                        BinaryClassificationMetric.Accuracy => metrics.Accuracy,
                        BinaryClassificationMetric.AreaUnderRocCurve => metrics.AreaUnderRocCurve,
                        BinaryClassificationMetric.AreaUnderPrecisionRecallCurve => metrics.AreaUnderPrecisionRecallCurve,
                        _ => throw new NotImplementedException($"{metricManager.Metric} is not supported!"),
                    };
                    var loss = metricManager.IsMaximize ? -metric : metric;

                    stopWatch.Stop();


                    return new TrialResult<BinaryClassificationMetrics>()
                    {
                        Loss = loss,
                        Metric = metric,
                        Model = model,
                        TrialSettings = settings,
                        DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                        Metrics = metrics,
                        Pipeline = pipeline,
                    };
                }
            }

            throw new ArgumentException($"The runner metric manager is of type {_metricManager.GetType()} which expected to be of type {typeof(ITrainTestDatasetManager)} or {typeof(ICrossValidateDatasetManager)}");
        }

        public Task<TrialResult> RunAsync(TrialSettings settings, CancellationToken ct)
        {
            try
            {
                using (var ctRegistration = ct.Register(() =>
                {
                    _context?.CancelExecution();
                }))
                {
                    return Task.Run(() => Run(settings));
                }
            }
            catch (Exception ex) when (ct.IsCancellationRequested)
            {
                throw new OperationCanceledException(ex.Message, ex.InnerException);
            }
            catch (Exception)
            {
                throw;
            }
        }
    }
}
