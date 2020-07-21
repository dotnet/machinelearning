// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.AutoML.Test
{
    public enum TaskType
    {
        Classification = 1,
        Regression = 2,
        Recommendation = 3,
        Ranking = 4
    }

    /// <summary>
    /// make AutoFit and Score calls uniform across task types
    /// </summary>
    internal class TaskAgnosticAutoFit
    {
        private TaskType _taskType;
        private MLContext _context;

        internal interface IUniversalProgressHandler : IProgress<RunDetail<RegressionMetrics>>, IProgress<RunDetail<MulticlassClassificationMetrics>>
        {
        }

        internal TaskAgnosticAutoFit(TaskType taskType, MLContext context)
        {
            this._taskType = taskType;
            this._context = context;
        }

        internal IEnumerable<TaskAgnosticIterationResult> AutoFit(
            IDataView trainData,
            string label,
            int maxModels,
            uint maxExperimentTimeInSeconds,
            IDataView validationData = null,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null,
            IUniversalProgressHandler progressHandler = null)
        {
            var columnInformation = new ColumnInformation() { LabelColumnName = label };

            switch (this._taskType)
            {
                case TaskType.Classification:

                    var mcs = new MulticlassExperimentSettings
                    {
                        OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,

                        MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds,
                        MaxModels = maxModels
                    };

                    var classificationResult = this._context.Auto()
                        .CreateMulticlassClassificationExperiment(mcs)
                        .Execute(
                            trainData,
                            validationData,
                            columnInformation,
                            progressHandler: progressHandler);

                    var iterationResults = classificationResult.RunDetails.Select(i => new TaskAgnosticIterationResult(i)).ToList();

                    return iterationResults;

                case TaskType.Regression:

                    var rs = new RegressionExperimentSettings
                    {
                        OptimizingMetric = RegressionMetric.RSquared,

                        MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds,
                        MaxModels = maxModels
                    };

                    var regressionResult = this._context.Auto()
                        .CreateRegressionExperiment(rs)
                        .Execute(
                            trainData,
                            validationData,
                            columnInformation,
                            progressHandler: progressHandler);

                    iterationResults = regressionResult.RunDetails.Select(i => new TaskAgnosticIterationResult(i)).ToList();

                    return iterationResults;

                case TaskType.Recommendation:

                    var recommendationSettings = new RecommendationExperimentSettings
                    {
                        OptimizingMetric = RegressionMetric.RSquared,

                        MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds,
                        MaxModels = maxModels
                    };

                    var recommendationResult = this._context.Auto()
                        .CreateRecommendationExperiment(recommendationSettings)
                        .Execute(
                            trainData,
                            validationData,
                            columnInformation,
                            progressHandler: progressHandler);

                    iterationResults = recommendationResult.RunDetails.Select(i => new TaskAgnosticIterationResult(i)).ToList();

                    return iterationResults;

                default:
                    throw new ArgumentException($"Unknown task type {this._taskType}.", "TaskType");
            }
        }

        internal struct ScoreResult
        {
            public IDataView ScoredTestData;
            public double PrimaryMetricResult;
            public Dictionary<string, double> Metrics;
        }

        internal ScoreResult Score(
            IDataView testData,
            ITransformer model,
            string label)
        {
            var result = new ScoreResult();

            result.ScoredTestData = model.Transform(testData);

            switch (this._taskType)
            {
                case TaskType.Classification:

                    var classificationMetrics = _context.MulticlassClassification.Evaluate(result.ScoredTestData, labelColumnName: label);

                    //var classificationMetrics = context.MulticlassClassification.(scoredTestData, labelColumnName: label);
                    result.PrimaryMetricResult = classificationMetrics.MicroAccuracy; // TODO: don't hardcode metric
                    result.Metrics = TaskAgnosticIterationResult.MetricValuesToDictionary(classificationMetrics);

                    break;

                case TaskType.Regression:

                    var regressionMetrics = _context.Regression.Evaluate(result.ScoredTestData, labelColumnName: label);

                    result.PrimaryMetricResult = regressionMetrics.RSquared; // TODO: don't hardcode metric
                    result.Metrics = TaskAgnosticIterationResult.MetricValuesToDictionary(regressionMetrics);

                    break;

                default:
                    throw new ArgumentException($"Unknown task type {this._taskType}.", "TaskType");
            }

            return result;
        }
    }
}

