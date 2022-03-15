// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;

namespace Microsoft.ML.AutoML
{
    internal interface ITrialRunner
    {
        TrialResult Run(MLContext context, TrialSettings settings);
    }

    internal class BinaryClassificationCVRunner : ITrialRunner
    {
        public TrialResult Run(MLContext context, TrialSettings settings)
        {
            var rnd = new Random(settings.ExperimentSettings.Seed ?? 0);
            if (settings.ExperimentSettings.DatasetSettings is CrossValidateDatasetSettings datasetSettings
                && settings.ExperimentSettings.EvaluateMetric is BinaryMetricSettings metricSettings)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();
                var fold = datasetSettings.Fold ?? 5;

                var pipeline = settings.Pipeline.BuildTrainingPipeline(context, settings.Parameter);
                var metrics = context.BinaryClassification.CrossValidateNonCalibrated(datasetSettings.Dataset, pipeline, fold, metricSettings.PredictedColumn);

                // now we just randomly pick a model, but a better way is to provide option to pick a model which score is the cloest to average or the best.
                var res = metrics[rnd.Next(fold)];
                var model = res.Model;
                var metric = metricSettings.Metric switch
                {
                    BinaryClassificationMetric.PositivePrecision => res.Metrics.PositivePrecision,
                    BinaryClassificationMetric.Accuracy => res.Metrics.Accuracy,
                    BinaryClassificationMetric.AreaUnderRocCurve => res.Metrics.AreaUnderRocCurve,
                    BinaryClassificationMetric.AreaUnderPrecisionRecallCurve => res.Metrics.AreaUnderPrecisionRecallCurve,
                    _ => throw new NotImplementedException($"{metricSettings.Metric} is not supported!"),
                };

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

    internal class BinaryClassificationTrainTestRunner : ITrialRunner
    {
        public TrialResult Run(MLContext context, TrialSettings settings)
        {
            var rnd = new Random(settings.ExperimentSettings.Seed ?? 0);
            if (settings.ExperimentSettings.DatasetSettings is TrainTestDatasetSettings datasetSettings
                && settings.ExperimentSettings.EvaluateMetric is BinaryMetricSettings metricSettings)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();

                var pipeline = settings.Pipeline.BuildTrainingPipeline(context, settings.Parameter);
                var model = pipeline.Fit(datasetSettings.TrainDataset);
                var eval = model.Transform(datasetSettings.TestDataset);
                var metrics = context.BinaryClassification.EvaluateNonCalibrated(eval, metricSettings.PredictedColumn, predictedLabelColumnName: metricSettings.TruthColumn);

                // now we just randomly pick a model, but a better way is to provide option to pick a model which score is the cloest to average or the best.
                var metric = metricSettings.Metric switch
                {
                    BinaryClassificationMetric.PositivePrecision => metrics.PositivePrecision,
                    BinaryClassificationMetric.Accuracy => metrics.Accuracy,
                    BinaryClassificationMetric.AreaUnderRocCurve => metrics.AreaUnderRocCurve,
                    BinaryClassificationMetric.AreaUnderPrecisionRecallCurve => metrics.AreaUnderPrecisionRecallCurve,
                    _ => throw new NotImplementedException($"{metricSettings.Metric} is not supported!"),
                };

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
