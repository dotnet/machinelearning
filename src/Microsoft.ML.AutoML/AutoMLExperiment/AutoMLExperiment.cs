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
    internal class AutoMLExperiment
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
            _settings.DatasetSettings = new TrainTestDatasetSettings()
            {
                TrainDataset = train,
                TestDataset = test
            };

            return this;
        }

        public AutoMLExperiment SetDataset(TrainTestData trainTestSplit)
        {
            SetDataset(trainTestSplit.TrainSet, trainTestSplit.TestSet);

            return this;
        }

        public AutoMLExperiment SetDataset(IDataView dataset, int fold = 10)
        {
            _settings.DatasetSettings = new CrossValidateDatasetSettings()
            {
                Dataset = dataset,
                Fold = fold,
            };

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

        public AutoMLExperiment SetTrialRunnerFactory(ITrialRunnerFactory factory)
        {
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

        public AutoMLExperiment SetEvaluateMetric(BinaryClassificationMetric metric, string labelColumn = "label", string predictedColumn = "Predicted")
        {
            _settings.EvaluateMetric = new BinaryMetricSettings()
            {
                Metric = metric,
                PredictedColumn = predictedColumn,
                LabelColumn = labelColumn,
            };

            return this;
        }

        public AutoMLExperiment SetEvaluateMetric(MulticlassClassificationMetric metric, string labelColumn = "label", string predictedColumn = "Predicted")
        {
            _settings.EvaluateMetric = new MultiClassMetricSettings()
            {
                Metric = metric,
                PredictedColumn = predictedColumn,
                LabelColumn = labelColumn,
            };

            return this;
        }

        public AutoMLExperiment SetEvaluateMetric(RegressionMetric metric, string labelColumn = "label", string scoreColumn = "Score")
        {
            _settings.EvaluateMetric = new RegressionMetricSettings()
            {
                Metric = metric,
                ScoreColumn = scoreColumn,
                LabelColumn = labelColumn,
            };

            return this;
        }

        /// <summary>
        /// Run experiment and return the best trial result.
        /// </summary>
        /// <returns></returns>
        public Task<TrialResult> Run()
        {
            ValidateSettings();
            var cts = new CancellationTokenSource();
            cts.CancelAfter((int)_settings.MaxExperimentTimeInSeconds * 1000);
            _settings.CancellationToken.Register(() => cts.Cancel());
            var ct = cts.Token;
            ct.Register(() => _context.CancelExecution());

            return RunAsync(ct);
        }

        private async Task<TrialResult> RunAsync(CancellationToken ct)
        {
            InitializeServiceCollection();
            var serviceProvider = _serviceCollection.BuildServiceProvider();
            var monitor = serviceProvider.GetService<IMonitor>();
            var trialNum = 0;
            var pipelineProposer = serviceProvider.GetService<PipelineProposer>();
            var hyperParameterProposer = serviceProvider.GetService<HyperParameterProposer>();
            var runnerFactory = serviceProvider.GetService<ITrialRunnerFactory>();

            while (true)
            {
                try
                {
                    if (ct.IsCancellationRequested)
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
                    var runner = runnerFactory.CreateTrialRunner(setting);
                    var trialResult = runner.Run(setting);
                    monitor.ReportCompletedTrial(trialResult);
                    hyperParameterProposer.Update(setting, trialResult);
                    pipelineProposer.Update(setting, trialResult);

                    var error = _settings.EvaluateMetric.IsMaximize ? 1 - trialResult.Metric : trialResult.Metric;
                    if (error < _bestError)
                    {
                        _bestTrialResult = trialResult;
                        _bestError = error;
                        monitor.ReportBestTrial(trialResult);
                    }
                }
                catch (Exception)
                {
                    if (ct.IsCancellationRequested)
                    {
                        break;
                    }
                    else
                    {
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
            Contracts.Assert(_settings.DatasetSettings != null, $"{nameof(_settings.DatasetSettings)} must be not null");
            Contracts.Assert(_settings.EvaluateMetric != null, $"{nameof(_settings.EvaluateMetric)} must be not null");
        }


        public class AutoMLExperimentSettings : ExperimentSettings
        {
            public IDatasetSettings DatasetSettings { get; set; }

            public IMetricSettings EvaluateMetric { get; set; }

            public MultiModelPipeline Pipeline { get; set; }

            public int? Seed { get; set; }
        }
    }
}
