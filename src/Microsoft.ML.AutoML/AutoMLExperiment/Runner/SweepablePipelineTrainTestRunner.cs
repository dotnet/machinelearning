// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal class SweepablePipelineTrainTestRunner : ITrialRunner
    {
        private readonly MLContext _mLContext;
        private readonly IEvaluateMetricManager _metricManager;
        private readonly ITrainTestDatasetManager _trainTestDatasetManager;
        private readonly SweepablePipeline _pipeline;

        public SweepablePipelineTrainTestRunner(MLContext context, SweepablePipeline pipeline, IEvaluateMetricManager metricManager, ITrainTestDatasetManager trainTestDatasetManager)
        {
            _mLContext = context;
            _metricManager = metricManager;
            _pipeline = pipeline;
            _trainTestDatasetManager = trainTestDatasetManager;
        }

        public TrialResult Run(TrialSettings settings, IServiceProvider provider)
        {
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            var parameter = settings.Parameter[AutoMLExperiment.PipelineSearchspaceName];
            var mlnetPipeline = _pipeline.BuildFromOption(_mLContext, parameter);
            var model = mlnetPipeline.Fit(_trainTestDatasetManager.TrainDataset);
            var eval = model.Transform(_trainTestDatasetManager.TestDataset);
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
    }
}
