// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public static class RegressionExtensions
    {
        public static RegressionResult AutoFit(this RegressionContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            uint timeoutInMinutes = AutoFitDefaults.TimeOutInMinutes,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null,
            CancellationToken cancellationToken = default,
            IProgress<RegressionIterationResult> iterationCallback = null)
        {
            var settings = new AutoFitSettings();
            settings.StoppingCriteria.TimeOutInMinutes = timeoutInMinutes;

            return AutoFit(context, trainData, label, validationData, settings,
                preFeaturizers, columnPurposes, cancellationToken, iterationCallback, null);
        }
        
        internal static RegressionResult AutoFit(this RegressionContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null,
            CancellationToken cancellationToken = default,
            IProgress<RegressionIterationResult> iterationCallback = null,
            IDebugLogger debugLogger = null)
        {
            UserInputValidationUtil.ValidateAutoFitArgs(trainData, label, validationData, settings, columnPurposes);

            if (validationData == null)
            {
                (trainData, validationData) = context.TestValidateSplit(trainData);
            }

            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.Fit(trainData, validationData, label,
                settings, preFeaturizers, TaskKind.Regression, OptimizingMetric.RSquared, columnPurposes, debugLogger);

            var results = new RegressionIterationResult[allPipelines.Length];
            for (var i = 0; i < results.Length; i++)
            {
                var iterationResult = allPipelines[i];
                var result = new RegressionIterationResult(iterationResult.Model, (RegressionMetrics)iterationResult.EvaluatedMetrics, iterationResult.ScoredValidationData, iterationResult.Pipeline.ToPipeline());
                results[i] = result;
            }
            var bestResult = new RegressionIterationResult(bestPipeline.Model, (RegressionMetrics)bestPipeline.EvaluatedMetrics, bestPipeline.ScoredValidationData, bestPipeline.Pipeline.ToPipeline());
            return new RegressionResult(bestResult, results);
        }
    }

    public static class BinaryClassificationExtensions
    {
        public static BinaryClassificationResult AutoFit(this BinaryClassificationContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            uint timeoutInMinutes = AutoFitDefaults.TimeOutInMinutes,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null,
            CancellationToken cancellationToken = default,
            IProgress<BinaryClassificationItertionResult> iterationCallback = null)
        {
            var settings = new AutoFitSettings();
            settings.StoppingCriteria.TimeOutInMinutes = timeoutInMinutes;

            return AutoFit(context, trainData, label, validationData, settings,
                preFeaturizers, columnPurposes, cancellationToken, iterationCallback, null);
        }

        internal static BinaryClassificationResult AutoFit(this BinaryClassificationContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null,
            CancellationToken cancellationToken = default,
            IProgress<BinaryClassificationItertionResult> iterationCallback = null,
            IDebugLogger debugLogger = null)
        {
            UserInputValidationUtil.ValidateAutoFitArgs(trainData, label, validationData, settings, columnPurposes);

            if (validationData == null)
            {
                (trainData, validationData) = context.TestValidateSplit(trainData);
            }

            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.Fit(trainData, validationData, label,
                settings, preFeaturizers, TaskKind.BinaryClassification, OptimizingMetric.Accuracy,
                columnPurposes, debugLogger);

            var results = new BinaryClassificationItertionResult[allPipelines.Length];
            for (var i = 0; i < results.Length; i++)
            {
                var iterationResult = allPipelines[i];
                var result = new BinaryClassificationItertionResult(iterationResult.Model, (BinaryClassificationMetrics)iterationResult.EvaluatedMetrics, iterationResult.ScoredValidationData, iterationResult.Pipeline.ToPipeline());
                results[i] = result;
            }
            var bestResult = new BinaryClassificationItertionResult(bestPipeline.Model, (BinaryClassificationMetrics)bestPipeline.EvaluatedMetrics, bestPipeline.ScoredValidationData, bestPipeline.Pipeline.ToPipeline());
            return new BinaryClassificationResult(bestResult, results);
        }
    }

    public static class MulticlassExtensions
    {
        public static MulticlassClassificationResult AutoFit(this MulticlassClassificationContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            uint timeoutInMinutes = AutoFitDefaults.TimeOutInMinutes,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null,
            CancellationToken cancellationToken = default,
            IProgress<MulticlassClassificationIterationResult> iterationCallback = null)
        {
            var settings = new AutoFitSettings();
            settings.StoppingCriteria.TimeOutInMinutes = timeoutInMinutes;

            return AutoFit(context, trainData, label, validationData, settings,
                preFeaturizers, columnPurposes, cancellationToken, iterationCallback, null);
        }

        internal static MulticlassClassificationResult AutoFit(this MulticlassClassificationContext context,
            IDataView trainData,
            string label = DefaultColumnNames.Label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEstimator<ITransformer> preFeaturizers = null,
            IEnumerable<(string, ColumnPurpose)> columnPurposes = null,
            CancellationToken cancellationToken = default,
            IProgress<MulticlassClassificationIterationResult> iterationCallback = null, IDebugLogger debugLogger = null)
        {
            UserInputValidationUtil.ValidateAutoFitArgs(trainData, label, validationData, settings, columnPurposes);

            if (validationData == null)
            {
                (trainData, validationData) = context.TestValidateSplit(trainData);
            }

            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.Fit(trainData, validationData, label,
                settings, preFeaturizers, TaskKind.MulticlassClassification, OptimizingMetric.Accuracy,
                columnPurposes, debugLogger);

            var results = new MulticlassClassificationIterationResult[allPipelines.Length];
            for (var i = 0; i < results.Length; i++)
            {
                var iterationResult = allPipelines[i];
                var result = new MulticlassClassificationIterationResult(iterationResult.Model, (MultiClassClassifierMetrics)iterationResult.EvaluatedMetrics, iterationResult.ScoredValidationData, iterationResult.Pipeline.ToPipeline());
                results[i] = result;
            }
            var bestResult = new MulticlassClassificationIterationResult(bestPipeline.Model, (MultiClassClassifierMetrics)bestPipeline.EvaluatedMetrics, bestPipeline.ScoredValidationData, bestPipeline.Pipeline.ToPipeline());
            return new MulticlassClassificationResult(bestResult, results);
        }
    }

    public class BinaryClassificationResult
    {
        public readonly BinaryClassificationItertionResult BestIteration;
        public readonly BinaryClassificationItertionResult[] IterationResults;

        public BinaryClassificationResult(BinaryClassificationItertionResult bestPipeline,
            BinaryClassificationItertionResult[] iterationResults)
        {
            BestIteration = bestPipeline;
            IterationResults = iterationResults;
        }
    }

    public class MulticlassClassificationResult
    {
        public readonly MulticlassClassificationIterationResult BestIteration;
        public readonly MulticlassClassificationIterationResult[] IterationResults;

        public MulticlassClassificationResult(MulticlassClassificationIterationResult bestPipeline,
            MulticlassClassificationIterationResult[] iterationResults)
        {
            BestIteration = bestPipeline;
            IterationResults = iterationResults;
        }
    }

    public class RegressionResult
    {
        public readonly RegressionIterationResult BestIteration;
        public readonly RegressionIterationResult[] IterationResults;

        public RegressionResult(RegressionIterationResult bestPipeline,
            RegressionIterationResult[] iterationResults)
        {
            BestIteration = bestPipeline;
            IterationResults = iterationResults;
        }
    }

    public class BinaryClassificationItertionResult
    {
        public readonly BinaryClassificationMetrics Metrics;
        public readonly ITransformer Model;
        public readonly IDataView ScoredValidationData;
        internal readonly Pipeline Pipeline;

        internal BinaryClassificationItertionResult(ITransformer model, BinaryClassificationMetrics metrics, IDataView scoredValidationData, Pipeline pipeline)
        {
            Model = model;
            ScoredValidationData = scoredValidationData;
            Metrics = metrics;
            Pipeline = pipeline;
        }
    }

    public class MulticlassClassificationIterationResult
    {
        public readonly MultiClassClassifierMetrics Metrics;
        public readonly ITransformer Model;
        public readonly IDataView ScoredValidationData;
        internal readonly Pipeline Pipeline;

        internal MulticlassClassificationIterationResult(ITransformer model, MultiClassClassifierMetrics metrics, IDataView scoredValidationData, Pipeline pipeline)
        {
            Model = model;
            Metrics = metrics;
            ScoredValidationData = scoredValidationData;
            Pipeline = pipeline;
        }
    }

    public class RegressionIterationResult
    {
        public readonly RegressionMetrics Metrics;
        public readonly ITransformer Model;
        public readonly IDataView ScoredValidationData;
        internal readonly Pipeline Pipeline;

        internal RegressionIterationResult(ITransformer model, RegressionMetrics metrics, IDataView scoredValidationData, Pipeline pipeline)
        {
            Model = model;
            Metrics = metrics;
            ScoredValidationData = scoredValidationData;
            Pipeline = pipeline;
        }
    }
}
