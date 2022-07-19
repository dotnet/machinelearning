// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Microsoft.ML.AutoML;

namespace Microsoft.ML.Fairlearn.reductions
{
    /// <summary>
    /// 
    /// 1, generate cost column from lamda parameter
    /// 2. insert cost column into dataset
    /// 3. restore trainable pipeline
    /// 4. train
    /// 5. calculate metric = observe loss + fairness loss
    /// </summary>
    internal class GridSearchTrailRunner : ITrialRunner
    {
        private readonly MLContext _context;
        private readonly IDatasetManager _datasetManager;
        private readonly IMetricManager _metricManager;
        public GridSearchTrailRunner(MLContext context, IDatasetManager datasetManager, IMetricManager metricManager)
        {
            _context = context;
            _metricManager = metricManager;
            _datasetManager = datasetManager;
        }

        public TrialResult Run(TrialSettings settings, IServiceProvider provider)
        {
            if (_datasetManager is TrainTestDatasetManager datasetSettings
                && _metricManager is BinaryMetricManager metricSettings)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();

                var pipeline = settings.Pipeline.BuildTrainingPipeline(_context, settings.Parameter);
                var model = pipeline.Fit(datasetSettings.TrainDataset);
                var eval = model.Transform(datasetSettings.TestDataset);
                //TODO: calcualte fairnessLost
                double fairnessLost = 0.0f;
                var metrics = _context.BinaryClassification.EvaluateNonCalibrated(eval, metricSettings.LabelColumn, predictedLabelColumnName: metricSettings.PredictedColumn);
                var observedLoss = metricSettings.Metric switch
                {
                    BinaryClassificationMetric.PositivePrecision => metrics.PositivePrecision,
                    BinaryClassificationMetric.Accuracy => metrics.Accuracy,
                    BinaryClassificationMetric.AreaUnderRocCurve => metrics.AreaUnderRocCurve,
                    BinaryClassificationMetric.AreaUnderPrecisionRecallCurve => metrics.AreaUnderPrecisionRecallCurve,
                    _ => throw new NotImplementedException($"{metricSettings.Metric} is not supported!"),
                };
                // the metric should be the combination of the observed loss from the model and the fairness loss
                double metric = 0.0f;
                if (metricSettings.IsMaximize == true)
                {
                    metric = observedLoss - fairnessLost;
                }
                else
                {
                    metric = observedLoss + fairnessLost;
                }

                stopWatch.Stop();

                return new TrialResult()
                {
                    Metric = metric,
                    Model = model,
                    TrialSettings = settings,
                    DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                };
            }
            throw new ArgumentException();
        }
    }
}
