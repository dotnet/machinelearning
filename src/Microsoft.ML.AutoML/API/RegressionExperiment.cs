// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.Threading;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Settings for AutoML experiments on regression datasets.
    /// </summary>
    public sealed class RegressionExperimentSettings : ExperimentSettings
    {
        /// <summary>
        /// Metric that AutoML will try to optimize over the course of the experiment.
        /// </summary>
        /// <value>The default value is <see cref="RegressionMetric.RSquared" />.</value>
        public RegressionMetric OptimizingMetric { get; set; }

        /// <summary>
        /// Collection of trainers the AutoML experiment can leverage.
        /// </summary>
        /// <value>
        /// The default value is a collection auto-populated with all possible trainers (all values of <see cref="RegressionTrainer" />).
        /// </value>
        public ICollection<RegressionTrainer> Trainers { get; }

        /// <summary>
        /// Initializes a new instance of <see cref="RegressionExperimentSettings"/>.
        /// </summary>
        public RegressionExperimentSettings()
        {
            OptimizingMetric = RegressionMetric.RSquared;
            Trainers = Enum.GetValues(typeof(RegressionTrainer)).OfType<RegressionTrainer>().ToList();
        }
    }

    /// <summary>
    /// Regression metric that AutoML will aim to optimize in its sweeping process during an experiment.
    /// </summary>
    public enum RegressionMetric
    {
        /// <summary>
        /// See <see cref="RegressionMetrics.MeanAbsoluteError"/>.
        /// </summary>
        MeanAbsoluteError,

        /// <summary>
        /// See <see cref="RegressionMetrics.MeanSquaredError"/>.
        /// </summary>
        MeanSquaredError,

        /// <summary>
        /// See <see cref="RegressionMetrics.RootMeanSquaredError"/>.
        /// </summary>
        RootMeanSquaredError,

        /// <summary>
        /// See <see cref="RegressionMetrics.RSquared"/>.
        /// </summary>
        RSquared
    }

    /// <summary>
    /// Enumeration of ML.NET multiclass classification trainers used by AutoML.
    /// </summary>
    public enum RegressionTrainer
    {
        /// <summary>
        /// See <see cref="FastForestRegressionTrainer"/>.
        /// </summary>
        FastForest,

        /// <summary>
        /// See <see cref="FastTreeRegressionTrainer"/>.
        /// </summary>
        FastTree,

        /// <summary>
        /// See <see cref="FastTreeTweedieTrainer"/>.
        /// </summary>
        FastTreeTweedie,

        /// <summary>
        /// See <see cref="LightGbmRegressionTrainer"/>.
        /// </summary>
        LightGbm,

        /// <summary>
        /// See <see cref="LbfgsPoissonRegressionTrainer"/>.
        /// </summary>
        LbfgsPoissonRegression,

        /// <summary>
        /// See <see cref="SdcaRegressionTrainer"/>.
        /// </summary>
        StochasticDualCoordinateAscent,
    }

    /// <summary>
    /// AutoML experiment on regression classification datasets.
    /// </summary>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    ///  [!code-csharp[RegressionExperiment](~/../docs/samples/docs/samples/Microsoft.ML.AutoML.Samples/RegressionExperiment.cs)]
    /// ]]></format>
    /// </example>
    public sealed class RegressionExperiment : ExperimentBase<RegressionMetrics, RegressionExperimentSettings>
    {
        private readonly AutoMLExperiment _experiment;
        private const string Features = "__Features__";
        private SweepablePipeline _pipeline;

        internal RegressionExperiment(MLContext context, RegressionExperimentSettings settings)
            : base(context,
                  new RegressionMetricsAgent(context, settings.OptimizingMetric),
                  new OptimizingMetricInfo(settings.OptimizingMetric),
                  settings,
                  TaskKind.Regression,
                  TrainerExtensionUtil.GetTrainerNames(settings.Trainers))
        {
            _experiment = context.Auto().CreateExperiment();

            if (settings.MaximumMemoryUsageInMegaByte is double d)
            {
                _experiment.SetMaximumMemoryUsageInMegaByte(d);
            }

            _experiment.SetTrainingTimeInSeconds(Settings.MaxExperimentTimeInSeconds);
            _experiment.SetMaxModelToExplore(Settings.MaxModels);
        }

        public override ExperimentResult<RegressionMetrics> Execute(IDataView trainData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<RegressionMetrics>> progressHandler = null)
        {
            var label = columnInformation.LabelColumnName;
            _experiment.SetRegressionMetric(Settings.OptimizingMetric, label);

            // Cross val threshold for # of dataset rows --
            // If dataset has < threshold # of rows, use cross val.
            // Else, run experiment using train-validate split.
            const int crossValRowCountThreshold = 15000;
            var rowCount = DatasetDimensionsUtil.CountRows(trainData, crossValRowCountThreshold);
            // TODO
            // split cross validation result according to sample key as well.
            if (rowCount < crossValRowCountThreshold)
            {
                int numCrossValFolds = 10;
                _experiment.SetDataset(trainData, numCrossValFolds);
                _pipeline = CreateRegressionPipeline(trainData, columnInformation, preFeaturizer);

                TrialResultMonitor<RegressionMetrics> monitor = null;
                _experiment.SetMonitor((provider) =>
                {
                    var channel = provider.GetService<IChannel>();
                    var pipeline = provider.GetService<SweepablePipeline>();
                    monitor = new TrialResultMonitor<RegressionMetrics>(channel, pipeline);
                    monitor.OnTrialCompleted += (o, e) =>
                    {
                        var detail = BestResultUtil.ToRunDetail(Context, e, _pipeline);
                        progressHandler?.Report(detail);
                    };

                    return monitor;
                });

                _experiment.SetTrialRunner<RegressionTrialRunner>();
                _experiment.Run();

                var runDetails = monitor.RunDetails.Select(e => BestResultUtil.ToRunDetail(Context, e, _pipeline));
                var bestRun = BestResultUtil.ToRunDetail(Context, monitor.BestRun, _pipeline);
                var result = new ExperimentResult<RegressionMetrics>(runDetails, bestRun);

                return result;
            }
            else
            {
                var splitData = Context.Data.TrainTestSplit(trainData);
                return Execute(splitData.TrainSet, splitData.TestSet, columnInformation, preFeaturizer, progressHandler);
            }
        }

        public override ExperimentResult<RegressionMetrics> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<RegressionMetrics>> progressHandler = null)
        {
            var label = columnInformation.LabelColumnName;
            _experiment.SetRegressionMetric(Settings.OptimizingMetric, label);
            _experiment.SetDataset(trainData, validationData);

            _pipeline = CreateRegressionPipeline(trainData, columnInformation, preFeaturizer);
            _experiment.SetPipeline(_pipeline);

            // set monitor
            TrialResultMonitor<RegressionMetrics> monitor = null;
            _experiment.SetMonitor((provider) =>
            {
                var channel = provider.GetService<IChannel>();
                var pipeline = provider.GetService<SweepablePipeline>();
                monitor = new TrialResultMonitor<RegressionMetrics>(channel, pipeline);
                monitor.OnTrialCompleted += (o, e) =>
                {
                    var detail = BestResultUtil.ToRunDetail(Context, e, _pipeline);
                    progressHandler?.Report(detail);
                };

                return monitor;
            });

            _experiment.SetTrialRunner<RegressionTrialRunner>();
            _experiment.Run();

            var runDetails = monitor.RunDetails.Select(e => BestResultUtil.ToRunDetail(Context, e, _pipeline));
            var bestRun = BestResultUtil.ToRunDetail(Context, monitor.BestRun, _pipeline);
            var result = new ExperimentResult<RegressionMetrics>(runDetails, bestRun);

            return result;
        }

        public override ExperimentResult<RegressionMetrics> Execute(IDataView trainData, IDataView validationData, string labelColumnName = "Label", IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<RegressionMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
            };

            return Execute(trainData, validationData, columnInformation, preFeaturizer, progressHandler);
        }

        public override ExperimentResult<RegressionMetrics> Execute(IDataView trainData, string labelColumnName = "Label", string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<RegressionMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
                SamplingKeyColumnName = samplingKeyColumn,
            };

            return Execute(trainData, columnInformation, preFeaturizer, progressHandler);
        }

        public override CrossValidationExperimentResult<RegressionMetrics> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<CrossValidationRunDetail<RegressionMetrics>> progressHandler = null)
        {
            var label = columnInformation.LabelColumnName;
            _experiment.SetRegressionMetric(Settings.OptimizingMetric, label);
            _experiment.SetDataset(trainData, (int)numberOfCVFolds);

            _pipeline = CreateRegressionPipeline(trainData, columnInformation, preFeaturizer);
            _experiment.SetPipeline(_pipeline);

            // set monitor
            TrialResultMonitor<RegressionMetrics> monitor = null;
            _experiment.SetMonitor((provider) =>
            {
                var channel = provider.GetService<IChannel>();
                var pipeline = provider.GetService<SweepablePipeline>();
                monitor = new TrialResultMonitor<RegressionMetrics>(channel, pipeline);
                monitor.OnTrialCompleted += (o, e) =>
                {
                    var detail = BestResultUtil.ToCrossValidationRunDetail(Context, e, _pipeline);
                    progressHandler?.Report(detail);
                };

                return monitor;
            });

            _experiment.SetTrialRunner<RegressionTrialRunner>();
            _experiment.Run();

            var runDetails = monitor.RunDetails.Select(e => BestResultUtil.ToCrossValidationRunDetail(Context, e, _pipeline));
            var bestResult = BestResultUtil.ToCrossValidationRunDetail(Context, monitor.BestRun, _pipeline);

            var result = new CrossValidationExperimentResult<RegressionMetrics>(runDetails, bestResult);

            return result;
        }

        public override CrossValidationExperimentResult<RegressionMetrics> Execute(IDataView trainData, uint numberOfCVFolds, string labelColumnName = "Label", string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<CrossValidationRunDetail<RegressionMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
                SamplingKeyColumnName = samplingKeyColumn,
            };

            return Execute(trainData, numberOfCVFolds, columnInformation, preFeaturizer, progressHandler);
        }

        private SweepablePipeline CreateRegressionPipeline(IDataView trainData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null)
        {
            var useSdca = Settings.Trainers.Contains(RegressionTrainer.StochasticDualCoordinateAscent);
            var uselbfgs = Settings.Trainers.Contains(RegressionTrainer.LbfgsPoissonRegression);
            var useLgbm = Settings.Trainers.Contains(RegressionTrainer.LightGbm);
            var useFastForest = Settings.Trainers.Contains(RegressionTrainer.FastForest);
            var useFastTree = Settings.Trainers.Contains(RegressionTrainer.FastTree) || Settings.Trainers.Contains(RegressionTrainer.FastTreeTweedie);

            SweepablePipeline pipeline = new SweepablePipeline();
            if (preFeaturizer != null)
            {
                pipeline = pipeline.Append(preFeaturizer);
            }

            var label = columnInformation.LabelColumnName;
            pipeline = pipeline.Append(Context.Auto().Featurizer(trainData, columnInformation, Features));
            pipeline = pipeline.Append(Context.Auto().Regression(label, useSdca: useSdca, useFastTree: useFastTree, useLgbm: useLgbm, useLbfgs: uselbfgs, useFastForest: useFastForest, featureColumnName: Features));

            return pipeline;
        }

        private protected override CrossValidationRunDetail<RegressionMetrics> GetBestCrossValRun(IEnumerable<CrossValidationRunDetail<RegressionMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }

        private protected override RunDetail<RegressionMetrics> GetBestRun(IEnumerable<RunDetail<RegressionMetrics>> results)
        {
            return BestResultUtil.GetBestRun(results, MetricsAgent, OptimizingMetricInfo.IsMaximizing);
        }
    }

    /// <summary>
    /// Extension methods that operate over regression experiment run results.
    /// </summary>
    public static class RegressionExperimentResultExtensions
    {
        /// <summary>
        /// Select the best run from an enumeration of experiment runs.
        /// </summary>
        /// <param name="results">Enumeration of AutoML experiment run results.</param>
        /// <param name="metric">Metric to consider when selecting the best run.</param>
        /// <returns>The best experiment run.</returns>
        public static RunDetail<RegressionMetrics> Best(this IEnumerable<RunDetail<RegressionMetrics>> results, RegressionMetric metric = RegressionMetric.RSquared)
        {
            var metricsAgent = new RegressionMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }

        /// <summary>
        /// Select the best run from an enumeration of experiment cross validation runs.
        /// </summary>
        /// <param name="results">Enumeration of AutoML experiment cross validation run results.</param>
        /// <param name="metric">Metric to consider when selecting the best run.</param>
        /// <returns>The best experiment run.</returns>
        public static CrossValidationRunDetail<RegressionMetrics> Best(this IEnumerable<CrossValidationRunDetail<RegressionMetrics>> results, RegressionMetric metric = RegressionMetric.RSquared)
        {
            var metricsAgent = new RegressionMetricsAgent(null, metric);
            var isMetricMaximizing = new OptimizingMetricInfo(metric).IsMaximizing;
            return BestResultUtil.GetBestRun(results, metricsAgent, isMetricMaximizing);
        }
    }

    internal class RegressionTrialRunner : ITrialRunner
    {
        private MLContext _context;
        private readonly IDatasetManager _datasetManager;
        private readonly IMetricManager _metricManager;
        private readonly SweepablePipeline _pipeline;
        private readonly Random _rnd;

        public RegressionTrialRunner(MLContext context, IDatasetManager datasetManager, IMetricManager metricManager, SweepablePipeline pipeline, AutoMLExperiment.AutoMLExperimentSettings settings)
        {
            _context = context;
            _datasetManager = datasetManager;
            _metricManager = metricManager;
            _pipeline = pipeline;
            _rnd = settings.Seed.HasValue ? new Random(settings.Seed.Value) : new Random();
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
                    if (_metricManager is RegressionMetricManager metricManager)
                    {
                        var parameter = settings.Parameter[AutoMLExperiment.PipelineSearchspaceName];
                        var pipeline = _pipeline.BuildFromOption(_context, parameter);
                        if (_datasetManager is ICrossValidateDatasetManager datasetManager)
                        {
                            var stopWatch = new Stopwatch();
                            stopWatch.Start();
                            var fold = datasetManager.Fold ?? 5;
                            var metrics = _context.Regression.CrossValidate(datasetManager.Dataset, pipeline, fold, metricManager.LabelColumn);

                            // now we just randomly pick a model, but a better way is to provide option to pick a model which score is the cloest to average or the best.
                            var res = metrics[_rnd.Next(fold)];
                            var model = res.Model;
                            var metric = metricManager.Metric switch
                            {
                                RegressionMetric.RootMeanSquaredError => res.Metrics.RootMeanSquaredError,
                                RegressionMetric.RSquared => res.Metrics.RSquared,
                                RegressionMetric.MeanSquaredError => res.Metrics.MeanSquaredError,
                                RegressionMetric.MeanAbsoluteError => res.Metrics.MeanAbsoluteError,
                                _ => throw new NotImplementedException($"{metricManager.MetricName} is not supported!"),
                            };
                            var loss = metricManager.IsMaximize ? -metric : metric;

                            stopWatch.Stop();


                            return Task.FromResult(new TrialResult<RegressionMetrics>()
                            {
                                Loss = loss,
                                Metric = metric,
                                Model = model,
                                TrialSettings = settings,
                                DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                                Metrics = res.Metrics,
                                CrossValidationMetrics = metrics,
                                Pipeline = pipeline,
                            } as TrialResult);
                        }

                        if (_datasetManager is ITrainTestDatasetManager trainTestDatasetManager)
                        {
                            var stopWatch = new Stopwatch();
                            stopWatch.Start();
                            var model = pipeline.Fit(trainTestDatasetManager.TrainDataset);
                            var eval = model.Transform(trainTestDatasetManager.TestDataset);
                            var res = _context.Regression.Evaluate(eval, metricManager.LabelColumn, scoreColumnName: metricManager.ScoreColumn);

                            var metric = metricManager.Metric switch
                            {
                                RegressionMetric.RootMeanSquaredError => res.RootMeanSquaredError,
                                RegressionMetric.RSquared => res.RSquared,
                                RegressionMetric.MeanSquaredError => res.MeanSquaredError,
                                RegressionMetric.MeanAbsoluteError => res.MeanAbsoluteError,
                                _ => throw new NotImplementedException($"{metricManager.Metric} is not supported!"),
                            };
                            var loss = metricManager.IsMaximize ? -metric : metric;

                            stopWatch.Stop();


                            return Task.FromResult(new TrialResult<RegressionMetrics>()
                            {
                                Loss = loss,
                                Metric = metric,
                                Model = model,
                                TrialSettings = settings,
                                DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                                Metrics = res,
                                Pipeline = pipeline,
                            } as TrialResult);
                        }
                    }

                    throw new ArgumentException($"The runner metric manager is of type {_metricManager.GetType()} which expected to be of type {typeof(ITrainTestDatasetManager)} or {typeof(ICrossValidateDatasetManager)}");
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

        public void Dispose()
        {
            _context.CancelExecution();
            _context = null;
        }
    }
}
