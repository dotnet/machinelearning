// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;

namespace Microsoft.ML.AutoML
{
    internal interface ITrialRunner
    {
        TrialResult Run(TrialSettings settings);
    }

    internal class BinaryClassificationCVRunner : ITrialRunner
    {
        private readonly MLContext _context;
        public BinaryClassificationCVRunner(MLContext context)
        {
            this._context = context;
        }

        public TrialResult Run(TrialSettings settings)
        {
            var rnd = new Random(settings.ExperimentSettings.Seed ?? 0);
            if (settings.ExperimentSettings.DatasetSettings is CrossValidateDatasetSettings datasetSettings
                && settings.ExperimentSettings.EvaluateMetric is BinaryMetricSettings metricSettings)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();
                var fold = datasetSettings.Fold ?? 5;

                var pipeline = settings.Pipeline.BuildTrainingPipeline(this._context, settings.Parameter);
                var metrics = this._context.BinaryClassification.CrossValidateNonCalibrated(datasetSettings.Dataset, pipeline, fold, metricSettings.LabelColumn);

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
        private readonly MLContext _context;
        public BinaryClassificationTrainTestRunner(MLContext context)
        {
            this._context = context;
        }

        public TrialResult Run(TrialSettings settings)
        {
            var rnd = new Random(settings.ExperimentSettings.Seed ?? 0);
            if (settings.ExperimentSettings.DatasetSettings is TrainTestDatasetSettings datasetSettings
                && settings.ExperimentSettings.EvaluateMetric is BinaryMetricSettings metricSettings)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();

                var pipeline = settings.Pipeline.BuildTrainingPipeline(this._context, settings.Parameter);
                var model = pipeline.Fit(datasetSettings.TrainDataset);
                var eval = model.Transform(datasetSettings.TestDataset);
                var metrics = this._context.BinaryClassification.EvaluateNonCalibrated(eval, metricSettings.LabelColumn, predictedLabelColumnName: metricSettings.PredictedColumn);

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

    internal class MultiClassificationTrainTestRunner : ITrialRunner
    {
        private readonly MLContext _context;
        public MultiClassificationTrainTestRunner(MLContext context)
        {
            this._context = context;
        }

        public TrialResult Run(TrialSettings settings)
        {
            if (settings.ExperimentSettings.DatasetSettings is TrainTestDatasetSettings datasetSettings
                && settings.ExperimentSettings.EvaluateMetric is MultiClassMetricSettings metricSettings)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();

                var pipeline = settings.Pipeline.BuildTrainingPipeline(this._context, settings.Parameter);
                var model = pipeline.Fit(datasetSettings.TrainDataset);
                var eval = model.Transform(datasetSettings.TestDataset);
                var metrics = this._context.MulticlassClassification.Evaluate(eval, metricSettings.LabelColumn, predictedLabelColumnName: metricSettings.PredictedColumn);

                var metric = metricSettings.Metric switch
                {
                    MulticlassClassificationMetric.MicroAccuracy => metrics.MicroAccuracy,
                    MulticlassClassificationMetric.MacroAccuracy => metrics.MacroAccuracy,
                    MulticlassClassificationMetric.TopKAccuracy => metrics.TopKAccuracy,
                    MulticlassClassificationMetric.LogLoss => metrics.LogLoss,
                    MulticlassClassificationMetric.LogLossReduction => metrics.LogLossReduction,
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

    internal class MultiClassificationCVRunner : ITrialRunner
    {
        private readonly MLContext _context;
        public MultiClassificationCVRunner(MLContext context)
        {
            this._context = context;
        }

        public TrialResult Run(TrialSettings settings)
        {
            var rnd = new Random(settings.ExperimentSettings.Seed ?? 0);
            if (settings.ExperimentSettings.DatasetSettings is CrossValidateDatasetSettings datasetSettings
                && settings.ExperimentSettings.EvaluateMetric is MultiClassMetricSettings metricSettings)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();
                var fold = datasetSettings.Fold ?? 5;

                var pipeline = settings.Pipeline.BuildTrainingPipeline(this._context, settings.Parameter);
                var metrics = this._context.MulticlassClassification.CrossValidate(datasetSettings.Dataset, pipeline, fold, metricSettings.LabelColumn, seed: settings.ExperimentSettings?.Seed);
                // now we just randomly pick a model, but a better way is to provide option to pick a model which score is the cloest to average or the best.
                var res = metrics[rnd.Next(fold)];
                var model = res.Model;
                var metric = metricSettings.Metric switch
                {
                    MulticlassClassificationMetric.MicroAccuracy => res.Metrics.MicroAccuracy,
                    MulticlassClassificationMetric.MacroAccuracy => res.Metrics.MacroAccuracy,
                    MulticlassClassificationMetric.TopKAccuracy => res.Metrics.TopKAccuracy,
                    MulticlassClassificationMetric.LogLoss => res.Metrics.LogLoss,
                    MulticlassClassificationMetric.LogLossReduction => res.Metrics.LogLossReduction,
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

    internal class RegressionTrainTestRunner : ITrialRunner
    {
        private readonly MLContext _context;
        public RegressionTrainTestRunner(MLContext context)
        {
            this._context = context;
        }

        public TrialResult Run(TrialSettings settings)
        {
            if (settings.ExperimentSettings.DatasetSettings is TrainTestDatasetSettings datasetSettings
                && settings.ExperimentSettings.EvaluateMetric is RegressionMetricSettings metricSettings)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();

                var pipeline = settings.Pipeline.BuildTrainingPipeline(this._context, settings.Parameter);
                var model = pipeline.Fit(datasetSettings.TrainDataset);
                var eval = model.Transform(datasetSettings.TestDataset);
                var metrics = this._context.Regression.Evaluate(eval, metricSettings.LabelColumn, scoreColumnName: metricSettings.ScoreColumn);

                var metric = metricSettings.Metric switch
                {
                    RegressionMetric.RootMeanSquaredError => metrics.RootMeanSquaredError,
                    RegressionMetric.RSquared => metrics.RSquared,
                    RegressionMetric.MeanSquaredError => metrics.MeanSquaredError,
                    RegressionMetric.MeanAbsoluteError => metrics.MeanAbsoluteError,
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

    internal class RegressionCVRunner : ITrialRunner
    {
        private readonly MLContext _context;
        public RegressionCVRunner(MLContext context)
        {
            this._context = context;
        }

        public TrialResult Run(TrialSettings settings)
        {
            var rnd = new Random(settings.ExperimentSettings.Seed ?? 0);
            if (settings.ExperimentSettings.DatasetSettings is CrossValidateDatasetSettings datasetSettings
                && settings.ExperimentSettings.EvaluateMetric is RegressionMetricSettings metricSettings)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();
                var fold = datasetSettings.Fold ?? 5;

                var pipeline = settings.Pipeline.BuildTrainingPipeline(this._context, settings.Parameter);
                var metrics = this._context.Regression.CrossValidate(datasetSettings.Dataset, pipeline, fold, metricSettings.LabelColumn, seed: settings.ExperimentSettings?.Seed);
                // now we just randomly pick a model, but a better way is to provide option to pick a model which score is the cloest to average or the best.
                var res = metrics[rnd.Next(fold)];
                var model = res.Model;
                var metric = metricSettings.Metric switch
                {
                    RegressionMetric.RootMeanSquaredError => res.Metrics.RootMeanSquaredError,
                    RegressionMetric.RSquared => res.Metrics.RSquared,
                    RegressionMetric.MeanSquaredError => res.Metrics.MeanSquaredError,
                    RegressionMetric.MeanAbsoluteError => res.Metrics.MeanAbsoluteError,
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
