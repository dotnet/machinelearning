// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#nullable enable
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    internal class SweepablePipelineRunner : ITrialRunner
    {
        private MLContext? _mLContext;
        private readonly IEvaluateMetricManager _metricManager;
        private readonly IDatasetManager _datasetManager;
        private readonly SweepablePipeline _pipeline;
        private readonly IChannel? _logger;
        private bool _disposedValue;
        private Task<TrialResult>? _disposableTrainingTask;
        private Task<TrialResult>? _disposableCancellationTask;

        public SweepablePipelineRunner(MLContext context, SweepablePipeline pipeline, IEvaluateMetricManager metricManager, IDatasetManager datasetManager, IChannel? logger = null)
        {
            _mLContext = context;
            _metricManager = metricManager;
            _pipeline = pipeline;
            _datasetManager = datasetManager;
            _logger = logger;
        }

        public TrialResult Run(TrialSettings settings)
        {
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            var parameter = settings.Parameter[AutoMLExperiment.PipelineSearchspaceName];
            var mlnetPipeline = _pipeline.BuildFromOption(_mLContext, parameter);
            if (_datasetManager is ICrossValidateDatasetManager crossValidateDatasetManager)
            {
                var datasetSplit = _mLContext!.Data.CrossValidationSplit(crossValidateDatasetManager.Dataset, crossValidateDatasetManager.Fold ?? 5);
                var metrics = new List<double>();
                var models = new List<ITransformer>();
                foreach (var split in datasetSplit)
                {
                    var model = mlnetPipeline.Fit(split.TrainSet);
                    var eval = model.Transform(split.TestSet);
                    metrics.Add(_metricManager.Evaluate(_mLContext, eval));
                    models.Add(model);
                }

                stopWatch.Stop();

                return new TrialResult
                {
                    Metric = metrics.Average(),
                    Model = models.First(),
                    DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                    TrialSettings = settings,
                };
            }

            if (_datasetManager is ITrainTestDatasetManager trainTestDatasetManager)
            {
                var model = mlnetPipeline.Fit(trainTestDatasetManager.TrainDataset);
                var eval = model.Transform(trainTestDatasetManager.TestDataset);
                var metric = _metricManager.Evaluate(_mLContext, eval);
                stopWatch.Stop();

                return new TrialResult
                {
                    Metric = metric,
                    Model = model,
                    DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                    TrialSettings = settings,
                };
            }

            throw new ArgumentException("IDatasetManager must be either ITrainTestDatasetManager or ICrossValidationDatasetManager");
        }

        public async Task<TrialResult> RunAsync(TrialSettings settings, CancellationToken ct)
        {
            var cts = new CancellationTokenSource();
            ct.Register(async () =>
            {
                cts.Cancel();
                await Task.Delay(100);
                _mLContext?.CancelExecution();
            });

            _disposableTrainingTask = Task.Run(() => Run(settings));
            _disposableCancellationTask = Task.Run(async () =>
            {
                while (!ct.IsCancellationRequested)
                {
                    await Task.Delay(100);
                }

                return new TrialResult();
            });
            var task = await Task.WhenAny(_disposableTrainingTask, _disposableCancellationTask);
            var result = await task;
            if (!cts.Token.IsCancellationRequested)
            {
                cts.Cancel();
                await _disposableCancellationTask;

                return result;
            }
            else
            {
                try
                {
                    await _disposableTrainingTask;
                }
                catch (Exception ex)
                {
                    throw new OperationCanceledException(ex.Message, ex);
                }
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    _mLContext?.CancelExecution();
                    _mLContext = null;
                    _disposableTrainingTask?.Dispose();
                    _disposableTrainingTask = null;
                    _disposableCancellationTask?.Dispose();
                    _disposableCancellationTask = null;
                }

                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                _disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~SweepablePipelineRunner()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
