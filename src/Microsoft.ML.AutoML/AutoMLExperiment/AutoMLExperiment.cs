// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.ML.Runtime;
using static Microsoft.ML.DataOperationsCatalog;

namespace Microsoft.ML.AutoML
{
    public class AutoMLExperiment
    {
        private readonly AutoMLExperimentSettings _settings;
        private readonly MLContext _context;
        private double _bestError = double.MaxValue;
        private TrialResult _bestTrialResult = null;
        private readonly IServiceCollection _serviceCollection;

        public AutoMLExperiment(MLContext context, AutoMLExperimentSettings settings)
        {
            _context = context;
            _settings = settings;
            _serviceCollection = new ServiceCollection();
        }

        private void InitializeServiceCollection()
        {
            _serviceCollection.TryAddSingleton(_context);
            _serviceCollection.TryAddSingleton(_settings);
            _serviceCollection.TryAddSingleton<IMonitor, MLContextMonitor>();
            _serviceCollection.TryAddSingleton<ITrialRunnerFactory, TrialRunnerFactory>();
            _serviceCollection.TryAddSingleton<ITunerFactory, CostFrugalTunerFactory>();
            _serviceCollection.TryAddTransient<BinaryClassificationCVRunner>();
            _serviceCollection.TryAddTransient<BinaryClassificationTrainTestRunner>();
            _serviceCollection.TryAddTransient<RegressionTrainTestRunner>();
            _serviceCollection.TryAddTransient<RegressionCVRunner>();
            _serviceCollection.TryAddTransient<MultiClassificationCVRunner>();
            _serviceCollection.TryAddTransient<MultiClassificationTrainTestRunner>();
            _serviceCollection.TryAddScoped<HyperParameterProposer>();
            _serviceCollection.TryAddScoped<PipelineProposer>();
        }

        public AutoMLExperiment SetTrainingTimeInSeconds(uint trainingTimeInSeconds)
        {
            _settings.MaxExperimentTimeInSeconds = trainingTimeInSeconds;
            return this;
        }

        public AutoMLExperiment SetDataset(IDataView train, IDataView test)
        {
            var datasetManager = new TrainTestDatasetManager()
            {
                TrainDataset = train,
                TestDataset = test
            };

            _serviceCollection.AddSingleton<IDatasetManager>(datasetManager);

            return this;
        }

        public AutoMLExperiment SetDataset(TrainTestData trainTestSplit)
        {
            SetDataset(trainTestSplit.TrainSet, trainTestSplit.TestSet);

            return this;
        }

        public AutoMLExperiment SetDataset(IDataView dataset, int fold = 10)
        {
            var datasetManager = new CrossValidateDatasetManager()
            {
                Dataset = dataset,
                Fold = fold,
            };

            _serviceCollection.AddSingleton<IDatasetManager>(datasetManager);

            return this;
        }

        public AutoMLExperiment SetTunerFactory<TTunerFactory>()
            where TTunerFactory : ITunerFactory
        {
            var descriptor = new ServiceDescriptor(typeof(ITunerFactory), typeof(TTunerFactory), ServiceLifetime.Singleton);
            if (_serviceCollection.Contains(descriptor))
            {
                _serviceCollection.Replace(descriptor);
            }
            else
            {
                _serviceCollection.Add(descriptor);
            }

            return this;
        }

        public AutoMLExperiment SetMonitor(IMonitor monitor)
        {
            var descriptor = new ServiceDescriptor(typeof(IMonitor), monitor);
            if (_serviceCollection.Contains(descriptor))
            {
                _serviceCollection.Replace(descriptor);
            }
            else
            {
                _serviceCollection.Add(descriptor);
            }

            return this;
        }

        public AutoMLExperiment SetPipeline(MultiModelPipeline pipeline)
        {
            _settings.Pipeline = pipeline;
            return this;
        }

        public AutoMLExperiment SetIsMaximizeMetric(bool isMaximize)
        {
            _settings.IsMaximizeMetric = isMaximize;
            return this;
        }

        public AutoMLExperiment SetTrialRunner(ITrialRunner runner)
        {
            var factory = new CustomRunnerFactory(runner);
            var descriptor = new ServiceDescriptor(typeof(ITrialRunnerFactory), factory);
            if (_serviceCollection.Contains(descriptor))
            {
                _serviceCollection.Replace(descriptor);
            }
            else
            {
                _serviceCollection.Add(descriptor);
            }

            return this;
        }

        public AutoMLExperiment SetPipeline(SweepableEstimatorPipeline pipeline)
        {
            var res = new MultiModelPipeline();
            foreach (var e in pipeline.Estimators)
            {
                res = res.Append(e);
            }

            SetPipeline(res);

            return this;
        }

        public AutoMLExperiment SetEvaluateMetric(BinaryClassificationMetric metric, string labelColumn = "label", string predictedColumn = "PredictedLabel")
        {
            var metricManager = new BinaryMetricManager()
            {
                Metric = metric,
                PredictedColumn = predictedColumn,
                LabelColumn = labelColumn,
            };
            _serviceCollection.AddSingleton<IMetricManager>(metricManager);
            SetIsMaximizeMetric(metricManager.IsMaximize);

            return this;
        }

        public AutoMLExperiment SetEvaluateMetric(MulticlassClassificationMetric metric, string labelColumn = "label", string predictedColumn = "PredictedLabel")
        {
            var metricManager = new MultiClassMetricManager()
            {
                Metric = metric,
                PredictedColumn = predictedColumn,
                LabelColumn = labelColumn,
            };
            _serviceCollection.AddSingleton<IMetricManager>(metricManager);
            SetIsMaximizeMetric(metricManager.IsMaximize);

            return this;
        }

        public AutoMLExperiment SetEvaluateMetric(RegressionMetric metric, string labelColumn = "Label", string scoreColumn = "Score")
        {
            var metricManager = new RegressionMetricManager()
            {
                Metric = metric,
                ScoreColumn = scoreColumn,
                LabelColumn = labelColumn,
            };
            _serviceCollection.AddSingleton<IMetricManager>(metricManager);
            SetIsMaximizeMetric(metricManager.IsMaximize);

            return this;
        }

        /// <summary>
        /// Run experiment and return the best trial result synchronizely.
        /// </summary>
        public TrialResult Run()
        {
            return this.RunAsync().Result;
        }

        /// <summary>
        /// Run experiment and return the best trial result asynchronizely. The experiment returns the current best trial result if there's any trial completed when <paramref name="ct"/> get cancelled,
        /// and throws <see cref="TimeoutException"/> with message "Training time finished without completing a trial run" when no trial has completed.
        /// Another thing needs to notice is that this function won't immediately return after <paramref name="ct"/> get cancelled. Instead, it will call <see cref="MLContext.CancelExecution"/> to cancel all training process
        /// and wait all running trials get cancelled or completed.
        /// </summary>
        /// <returns></returns>
        public async Task<TrialResult> RunAsync(CancellationToken ct = default)
        {
            ValidateSettings();
            var cts = new CancellationTokenSource();
            _settings.CancellationToken = ct;
            cts.CancelAfter((int)_settings.MaxExperimentTimeInSeconds * 1000);
            _settings.CancellationToken.Register(() => cts.Cancel());
            cts.Token.Register(() =>
            {
                // only force-canceling running trials when there's completed trials.
                // otherwise, wait for the current running trial to be completed.
                if (_bestTrialResult != null)
                    _context.CancelExecution();
            });

            InitializeServiceCollection();
            var serviceProvider = _serviceCollection.BuildServiceProvider();
            var monitor = serviceProvider.GetService<IMonitor>();
            var trialNum = 0;
            var pipelineProposer = serviceProvider.GetService<PipelineProposer>();
            var hyperParameterProposer = serviceProvider.GetService<HyperParameterProposer>();
            var runnerFactory = serviceProvider.GetService<ITrialRunnerFactory>();

            while (true)
            {
                if (cts.Token.IsCancellationRequested)
                {
                    break;
                }
                var setting = new TrialSettings()
                {
                    ExperimentSettings = _settings,
                    TrialId = trialNum++,
                };

                setting = pipelineProposer.Propose(setting);
                setting = hyperParameterProposer.Propose(setting);
                monitor.ReportRunningTrial(setting);
                var runner = runnerFactory.CreateTrialRunner();

                try
                {
                    var trialResult = runner.Run(setting, serviceProvider);
                    monitor.ReportCompletedTrial(trialResult);
                    hyperParameterProposer.Update(setting, trialResult);
                    pipelineProposer.Update(setting, trialResult);

                    var error = _settings.IsMaximizeMetric ? 1 - trialResult.Metric : trialResult.Metric;
                    if (error < _bestError)
                    {
                        _bestTrialResult = trialResult;
                        _bestError = error;
                        monitor.ReportBestTrial(trialResult);
                    }
                }
                catch (Exception ex)
                {
                    if (cts.Token.IsCancellationRequested)
                    {
                        break;
                    }
                    else
                    {
                        // TODO
                        // it's questionable on whether to abort the entire training process
                        // for a single fail trial. We should make it an option and only exit
                        // when error is fatal (like schema mismatch).
                        monitor.ReportFailTrial(setting, ex);
                        throw;
                    }
                }
            }

            if (_bestTrialResult == null)
            {
                throw new TimeoutException("Training time finished without completing a trial run");
            }
            else
            {
                return await Task.FromResult(_bestTrialResult);
            }
        }

        private void ValidateSettings()
        {
            Contracts.Assert(_settings.MaxExperimentTimeInSeconds > 0, $"{nameof(ExperimentSettings.MaxExperimentTimeInSeconds)} must be larger than 0");
        }


        public class AutoMLExperimentSettings : ExperimentSettings
        {
            public MultiModelPipeline Pipeline { get; set; }

            public int? Seed { get; set; }

            public bool IsMaximizeMetric { get; set; }
        }
    }
}
