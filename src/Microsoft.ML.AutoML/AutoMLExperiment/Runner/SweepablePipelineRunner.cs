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

                var metric = metrics.Average();
                var loss = _metricManager.IsMaximize ? -metric : metric;

                return new TrialResult
                {
                    Metric = metric,
                    Model = models.First(),
                    DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                    TrialSettings = settings,
                    Loss = loss,
                };
            }

            if (_datasetManager is ITrainTestDatasetManager trainTestDatasetManager)
            {
                var model = mlnetPipeline.Fit(trainTestDatasetManager.TrainDataset);
                var eval = model.Transform(trainTestDatasetManager.TestDataset);
                var metric = _metricManager.Evaluate(_mLContext, eval);
                stopWatch.Stop();
                var loss = _metricManager.IsMaximize ? -metric : metric;

                return new TrialResult
                {
                    Loss = loss,
                    Metric = metric,
                    Model = model,
                    DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                    TrialSettings = settings,
                };
            }

            throw new ArgumentException("IDatasetManager must be either ITrainTestDatasetManager or ICrossValidationDatasetManager");
        }

        public Task<TrialResult> RunAsync(TrialSettings settings, CancellationToken ct)
        {
            try
            {
                using (var ctRegistration = ct.Register(() =>
                {
                    _mLContext?.CancelExecution();
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

        public void Dispose()
        {
            _mLContext!.CancelExecution();
            _mLContext = null;
        }
    }
}
