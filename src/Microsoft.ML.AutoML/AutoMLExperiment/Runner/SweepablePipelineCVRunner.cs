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
    internal class SweepablePipelineCVRunner : ITrialRunner
    {
        private readonly MLContext _mLContext;
        private readonly IEvaluateMetricManager _metricManager;
        private readonly ICrossValidateDatasetManager _crossValidateDatasetManager;
        private readonly SweepablePipeline _pipeline;

        public SweepablePipelineCVRunner(MLContext context, SweepablePipeline pipeline, IEvaluateMetricManager metricManager, ICrossValidateDatasetManager crossValidateDatasetManager)
        {
            _mLContext = context;
            _metricManager = metricManager;
            _pipeline = pipeline;
            _crossValidateDatasetManager = crossValidateDatasetManager;
        }

        public TrialResult Run(TrialSettings settings, IServiceProvider provider)
        {
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            var parameter = settings.Parameter[AutoMLExperiment.PipelineSearchspaceName];
            var mlnetPipeline = _pipeline.BuildFromOption(_mLContext, parameter);
            var datasetSplit = _mLContext.Data.CrossValidationSplit(_crossValidateDatasetManager.Dataset, _crossValidateDatasetManager.Fold ?? 5);
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
    }
}
