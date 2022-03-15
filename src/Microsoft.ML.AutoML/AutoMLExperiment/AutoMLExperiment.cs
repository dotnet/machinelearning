// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    internal class AutoMLExperiment
    {
        private readonly AutoMLExperimentSettings _settings;
        private readonly MLContext _context;
        private double _bestError = double.MaxValue;
        private TrialResult _bestTrialResult = null;

        public AutoMLExperiment(MLContext context, AutoMLExperimentSettings settings)
        {
            this._context = context;
            this._settings = settings;
        }

        public AutoMLExperiment SetTrainingTimeInSeconds(int trainingTimeInSeconds)
        {
            this._settings.MaxExperimentTimeInSeconds = (uint)trainingTimeInSeconds;
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

        public AutoMLExperiment SetDataset(IDataView dataset, int fold = 10)
        {
            this._settings.DatasetSettings = new CrossValidateDatasetSettings()
            {
                Dataset = dataset,
                Fold = fold,
            };

            return this;
        }

        public AutoMLExperiment SetTunerFactory(Func<ITuner> tunerFactory)
        {
            this._settings.TunerFactory = tunerFactory;

            return this;
        }

        public AutoMLExperiment SetMonitor(IMonitor monitor)
        {
            this._settings.Monitor = monitor;
            return this;
        }

        public AutoMLExperiment SetPipeline(MultiModelPipeline pipeline)
        {
            this._settings.Pipeline = pipeline;
            return this;
        }

        public AutoMLExperiment SetPipeline(SweepableEstimatorPipeline pipeline)
        {
            this._settings.Pipeline = new MultiModelPipeline().Append(pipeline.Estimators.ToArray());
            return this;
        }

        public AutoMLExperiment SetEvaluateMetric(BinaryClassificationMetric metric, string predictedColumn = "Predicted", string truthColumn = "label")
        {
            this._settings.EvaluateMetric = new BinaryMetricSettings()
            {
                Metric = metric,
                PredictedColumn = predictedColumn,
                TruthColumn = truthColumn,
            };

            return this;
        }

        public AutoMLExperiment SetEvaluateMetric(MulticlassClassificationMetric metric, string predictedColumn = "Predicted", string truthColumn = "label")
        {
            this._settings.EvaluateMetric = new MultiClassMetricSettings()
            {
                Metric = metric,
                PredictedColumn = predictedColumn,
                TruthColumn = truthColumn,
            };

            return this;
        }

        public AutoMLExperiment SetEvaluateMetric(RegressionMetric metric, string predictedColumn = "Predicted", string truthColumn = "label")
        {
            this._settings.EvaluateMetric = new RegressionMetricSettings()
            {
                Metric = metric,
                PredictedColumn = predictedColumn,
                TruthColumn = truthColumn,
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
            var trialNum = 0;
            var pipelineProposer = new PipelineProposer(this._settings.Seed ?? 0);
            var hyperParameterProposer = new HyperParameterProposer();

            while (true)
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

                ITrialRunner runner = (this._settings.DatasetSettings, this._settings.EvaluateMetric) switch
                {
                    (CrossValidateDatasetSettings, BinaryMetricSettings) => new BinaryClassificationCVRunner(),
                    (TrainTestDatasetSettings, BinaryMetricSettings) => new BinaryClassificationTrainTestRunner(),
                    _ => throw new NotImplementedException(),
                };

                this._settings.Monitor.ReportRunningTrial(setting);
                var trialResult = runner.Run(this._context, setting);
                this._settings.Monitor.ReportCompletedTrial(trialResult);
                hyperParameterProposer.Update(setting, trialResult);
                pipelineProposer.Update(setting, trialResult);

                var error = this._settings.EvaluateMetric.IsMaximize ? 1 - trialResult.Metric : trialResult.Metric;
                if (error < this._bestError)
                {
                    this._bestTrialResult = trialResult;
                    this._bestError = error;
                    this._settings.Monitor.ReportBestTrial(trialResult);
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
            Contracts.Assert(this._settings.TunerFactory != null, $"{nameof(this._settings.TunerFactory)} must be not null");
            Contracts.Assert(this._settings.EvaluateMetric != null, $"{nameof(this._settings.EvaluateMetric)} must be not null");
        }


        public class AutoMLExperimentSettings : ExperimentSettings
        {
            public IDatasetSettings DatasetSettings { get; set; }

            public IMetricSettings EvaluateMetric { get; set; }

            public MultiModelPipeline Pipeline { get; set; }

            public Func<ITuner> TunerFactory { get; set; }

            public IMonitor Monitor { get; set; }

            public int? Seed { get; set; }
        }
    }
}
