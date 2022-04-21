// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
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
            this._context = context;
            this._settings = settings;
            this._serviceCollection = new ServiceCollection();
        }

        private void InitializeServiceCollection()
        {
            this._serviceCollection.TryAddSingleton(this._context);
            this._serviceCollection.TryAddSingleton(this._settings);
            this._serviceCollection.TryAddSingleton<IMonitor, MLContextMonitor>();
            this._serviceCollection.TryAddSingleton<ITrialRunnerFactory, TrialRunnerFactory>();
            this._serviceCollection.TryAddSingleton<ITunerFactory, CostFrugalTunerFactory>();
            this._serviceCollection.TryAddTransient<BinaryClassificationCVRunner>();
            this._serviceCollection.TryAddTransient<BinaryClassificationTrainTestRunner>();
            this._serviceCollection.TryAddTransient<RegressionTrainTestRunner>();
            this._serviceCollection.TryAddTransient<RegressionCVRunner>();
            this._serviceCollection.TryAddTransient<MultiClassificationCVRunner>();
            this._serviceCollection.TryAddTransient<MultiClassificationTrainTestRunner>();
            this._serviceCollection.TryAddScoped<HyperParameterProposer>();
            this._serviceCollection.TryAddScoped<PipelineProposer>();
        }

        public AutoMLExperiment SetTrainingTimeInSeconds(uint trainingTimeInSeconds)
        {
            this._settings.MaxExperimentTimeInSeconds = trainingTimeInSeconds;
            return this;
        }

        public AutoMLExperiment SetDataset(IDataView train, IDataView test)
        {
            this._settings.DatasetSettings = new TrainTestDatasetSettings()
            {
                TrainDataset = train,
                TestDataset = test
            };

            return this;
        }

        public AutoMLExperiment SetDataset(TrainTestData trainTestSplit)
        {
            this.SetDataset(trainTestSplit.TrainSet, trainTestSplit.TestSet);

            return this;
        }

        public AutoMLExperiment SetDataset(IDataView dataset, int fold = 10)
        {
            this._settings.DatasetSettings = new CrossValidateDatasetSettings()
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
            if (this._serviceCollection.Contains(descriptor))
            {
                this._serviceCollection.Replace(descriptor);
            }
            else
            {
                this._serviceCollection.Add(descriptor);
            }

            return this;
        }

        public AutoMLExperiment SetMonitor(IMonitor monitor)
        {
            var descriptor = new ServiceDescriptor(typeof(IMonitor), monitor);
            if (this._serviceCollection.Contains(descriptor))
            {
                this._serviceCollection.Replace(descriptor);
            }
            else
            {
                this._serviceCollection.Add(descriptor);
            }

            return this;
        }

        public AutoMLExperiment SetPipeline(MultiModelPipeline pipeline)
        {
            this._settings.Pipeline = pipeline;
            return this;
        }

        public AutoMLExperiment SetTrialRunnerFactory(ITrialRunnerFactory factory)
        {
            var descriptor = new ServiceDescriptor(typeof(ITrialRunnerFactory), factory);
            if (this._serviceCollection.Contains(descriptor))
            {
                this._serviceCollection.Replace(descriptor);
            }
            else
            {
                this._serviceCollection.Add(descriptor);
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

            this.SetPipeline(res);

            return this;
        }

        public AutoMLExperiment SetEvaluateMetric(BinaryClassificationMetric metric, string labelColumn = "label", string predictedColumn = "Predicted")
        {
            this._settings.EvaluateMetric = new BinaryMetricSettings()
            {
                Metric = metric,
                PredictedColumn = predictedColumn,
                LabelColumn = labelColumn,
            };

            return this;
        }

        public AutoMLExperiment SetEvaluateMetric(MulticlassClassificationMetric metric, string labelColumn = "label", string predictedColumn = "Predicted")
        {
            this._settings.EvaluateMetric = new MultiClassMetricSettings()
            {
                Metric = metric,
                PredictedColumn = predictedColumn,
                LabelColumn = labelColumn,
            };

            return this;
        }

        public AutoMLExperiment SetEvaluateMetric(RegressionMetric metric, string labelColumn = "label", string scoreColumn = "Score")
        {
            this._settings.EvaluateMetric = new RegressionMetricSettings()
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
            this.ValidateSettings();
            var cts = new CancellationTokenSource();
            cts.CancelAfter((int)this._settings.MaxExperimentTimeInSeconds * 1000);
            this._settings.CancellationToken.Register(() => cts.Cancel());
            var ct = cts.Token;
            ct.Register(() => this._context.CancelExecution());

            return this.RunAsync(ct);
        }

        private async Task<TrialResult> RunAsync(CancellationToken ct)
        {
            this.InitializeServiceCollection();
            var serviceProvider = this._serviceCollection.BuildServiceProvider();
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
                        ExperimentSettings = this._settings,
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

                    var error = this._settings.EvaluateMetric.IsMaximize ? 1 - trialResult.Metric : trialResult.Metric;
                    if (error < this._bestError)
                    {
                        this._bestTrialResult = trialResult;
                        this._bestError = error;
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

            if (this._bestTrialResult == null)
            {
                throw new TimeoutException("Training time finished without completing a trial run");
            }
            else
            {
                return await Task.FromResult(this._bestTrialResult);
            }
        }

        private void ValidateSettings()
        {
            Contracts.Assert(this._settings.MaxExperimentTimeInSeconds > 0, $"{nameof(ExperimentSettings.MaxExperimentTimeInSeconds)} must be larger than 0");
            Contracts.Assert(this._settings.DatasetSettings != null, $"{nameof(this._settings.DatasetSettings)} must be not null");
            Contracts.Assert(this._settings.EvaluateMetric != null, $"{nameof(this._settings.EvaluateMetric)} must be not null");
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
