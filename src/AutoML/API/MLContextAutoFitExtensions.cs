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
            string label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEnumerable<(string, ColumnPurpose)> purposeOverrides = null,
            CancellationToken cancellationToken = default,
            IProgress<RegressionIterationResult> iterationCallback = null)
        {
            return AutoFit(context, trainData, label, validationData, settings,
                purposeOverrides, cancellationToken, iterationCallback, null);
        }

        // todo: instead of internal methods, use static debug class w/ singleton logger?
        internal static RegressionResult AutoFit(this RegressionContext context,
            IDataView trainData,
            string label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEnumerable<(string, ColumnPurpose)> purposeOverrides = null,
            CancellationToken cancellationToken = default,
            IProgress<RegressionIterationResult> iterationCallback = null,
            IDebugLogger debugLogger = null)
        {
            UserInputValidationUtil.ValidateAutoFitArgs(trainData, label, validationData, settings, purposeOverrides);

            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.Fit(trainData, validationData, label,
                settings, TaskKind.Regression, OptimizingMetric.RSquared, purposeOverrides, debugLogger);

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

        public static Pipeline GetPipeline(this RegressionContext context, IDataView dataView, string label)
        {
            return PipelineSuggesterApi.GetPipeline(TaskKind.Regression, dataView, label);
        }
    }

    public static class BinaryClassificationExtensions
    {
        public static BinaryClassificationResult AutoFit(this BinaryClassificationContext context,
            IDataView trainData,
            string label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEnumerable<(string, ColumnPurpose)> purposeOverrides = null,
            CancellationToken cancellationToken = default,
            IProgress<BinaryClassificationItertionResult> iterationCallback = null)
        {
            return AutoFit(context, trainData, label, validationData, settings,
                purposeOverrides, cancellationToken, iterationCallback, null);
        }

        internal static BinaryClassificationResult AutoFit(this BinaryClassificationContext context,
            IDataView trainData,
            string label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEnumerable<(string, ColumnPurpose)> purposeOverrides = null,
            CancellationToken cancellationToken = default,
            IProgress<BinaryClassificationItertionResult> iterationCallback = null,
            IDebugLogger debugLogger = null)
        {
            UserInputValidationUtil.ValidateAutoFitArgs(trainData, label, validationData, settings, purposeOverrides);

            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.Fit(trainData, validationData, label,
                settings, TaskKind.BinaryClassification, OptimizingMetric.Accuracy,
                purposeOverrides, debugLogger);

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

        public static Pipeline GetPipeline(this BinaryClassificationContext context, IDataView dataView, string label)
        {
            return PipelineSuggesterApi.GetPipeline(TaskKind.BinaryClassification, dataView, label);
        }
    }

    public static class MulticlassExtensions
    {
        public static MulticlassClassificationResult AutoFit(this MulticlassClassificationContext context,
            IDataView trainData,
            string label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEnumerable<(string, ColumnPurpose)> purposeOverrides = null,
            CancellationToken cancellationToken = default,
            IProgress<MulticlassClassificationIterationResult> iterationCallback = null)
        {
            return AutoFit(context, trainData, label, validationData, settings,
                purposeOverrides, cancellationToken, iterationCallback, null);
        }

        internal static MulticlassClassificationResult AutoFit(this MulticlassClassificationContext context,
            IDataView trainData,
            string label,
            IDataView validationData = null,
            AutoFitSettings settings = null,
            IEnumerable<(string, ColumnPurpose)> purposeOverrides = null,
            CancellationToken cancellationToken = default,
            IProgress<MulticlassClassificationIterationResult> iterationCallback = null, IDebugLogger debugLogger = null)
        {
            UserInputValidationUtil.ValidateAutoFitArgs(trainData, label, validationData, settings, purposeOverrides);

            // run autofit & get all pipelines run in that process
            var (allPipelines, bestPipeline) = AutoFitApi.Fit(trainData, validationData, label,
                settings, TaskKind.MulticlassClassification, OptimizingMetric.Accuracy,
                purposeOverrides, debugLogger);

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

        public static Pipeline GetPipeline(this MulticlassClassificationContext context, IDataView dataView, string label)
        {
            return PipelineSuggesterApi.GetPipeline(TaskKind.MulticlassClassification, dataView, label);
        }
    }

    public class BinaryClassificationResult
    {
        public readonly BinaryClassificationItertionResult BestPipeline;
        public readonly BinaryClassificationItertionResult[] IterationResults;

        public BinaryClassificationResult(BinaryClassificationItertionResult bestPipeline,
            BinaryClassificationItertionResult[] iterationResults)
        {
            BestPipeline = bestPipeline;
            IterationResults = iterationResults;
        }
    }

    public class MulticlassClassificationResult
    {
        public readonly MulticlassClassificationIterationResult BestPipeline;
        public readonly MulticlassClassificationIterationResult[] IterationResults;

        public MulticlassClassificationResult(MulticlassClassificationIterationResult bestPipeline,
            MulticlassClassificationIterationResult[] iterationResults)
        {
            BestPipeline = bestPipeline;
            IterationResults = iterationResults;
        }
    }

    public class RegressionResult
    {
        public readonly RegressionIterationResult BestPipeline;
        public readonly RegressionIterationResult[] IterationResults;

        public RegressionResult(RegressionIterationResult bestPipeline,
            RegressionIterationResult[] iterationResults)
        {
            BestPipeline = bestPipeline;
            IterationResults = iterationResults;
        }
    }

    public class BinaryClassificationItertionResult
    {
        public readonly BinaryClassificationMetrics Metrics;
        public readonly ITransformer Model;
        public readonly IDataView ScoredValidationData;
        public readonly Pipeline Pipeline;

        public BinaryClassificationItertionResult(ITransformer model, BinaryClassificationMetrics metrics, IDataView scoredValidationData, Pipeline pipeline)
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
        public readonly Pipeline Pipeline;

        public MulticlassClassificationIterationResult(ITransformer model, MultiClassClassifierMetrics metrics, IDataView scoredValidationData, Pipeline pipeline)
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
        public readonly Pipeline Pipeline;

        public RegressionIterationResult(ITransformer model, RegressionMetrics metrics, IDataView scoredValidationData, Pipeline pipeline)
        {
            Model = model;
            Metrics = metrics;
            ScoredValidationData = scoredValidationData;
            Pipeline = pipeline;
        }
    }
}
