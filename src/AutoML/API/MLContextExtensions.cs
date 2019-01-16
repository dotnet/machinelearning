using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
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
            for(var i = 0; i < results.Length; i++)
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

    public static class TransformExtensions
    {
        public static IEstimator<ITransformer> InferTransforms(this TransformsCatalog catalog, IDataView data, string label)
        {
            UserInputValidationUtil.ValidateInferTransformArgs(data, label);
            var mlContext = new MLContext();
            var suggestedTransforms = TransformInferenceApi.InferTransforms(mlContext, data, label);
            var estimators = suggestedTransforms.Select(s => s.Estimator);
            var pipeline = new EstimatorChain<ITransformer>();
            foreach(var estimator in estimators)
            {
                pipeline = pipeline.Append(estimator);
            }
            return pipeline;
        }
    }

    public static class DataExtensions
    {
        // Delimiter, header, column datatype inference
        public static ColumnInferenceResult InferColumns(this DataOperations catalog, string path, string label,
            bool hasHeader = false, char? separatorChar = null, bool? allowQuotedStrings = null, bool? supportSparse = null, bool trimWhitespace = false)
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(path, label);
            var mlContext = new MLContext();
            return ColumnInferenceApi.InferColumns(mlContext, path, label, hasHeader, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace);
        }

        public static IDataView AutoRead(this DataOperations catalog, string path, string label, 
            bool hasHeader = false, char? separatorChar = null, bool? allowQuotedStrings = null, bool? supportSparse = null, bool trimWhitespace = false)
        {
            UserInputValidationUtil.ValidateAutoReadArgs(path, label);
            var mlContext = new MLContext();
            var columnInferenceResult = ColumnInferenceApi.InferColumns(mlContext, path, label, hasHeader, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace);
            var textLoader = columnInferenceResult.BuildTextLoader();
            return textLoader.Read(path);
        }

        public static TextLoader CreateTextReader(this DataOperations catalog, ColumnInferenceResult columnInferenceResult)
        {
            UserInputValidationUtil.ValidateCreateTextReaderArgs(columnInferenceResult);
            return columnInferenceResult.BuildTextLoader();
        }

        // Task inference
        public static MachineLearningTaskType InferTask(this DataOperations catalog, IDataView dataView)
        {
            throw new NotImplementedException();
        }

        public enum MachineLearningTaskType
        {
            Regression,
            BinaryClassification,
            MultiClassClassification
        }
    }
    
    public class ColumnInferenceResult
    {
        public readonly IEnumerable<(TextLoader.Column, ColumnPurpose)> Columns;
        public readonly bool AllowQuotedStrings;
        public readonly bool SupportSparse;
        public readonly string Separator;
        public readonly bool HasHeader;
        public readonly bool TrimWhitespace;

        public ColumnInferenceResult(IEnumerable<(TextLoader.Column, ColumnPurpose)> columns,
            bool allowQuotedStrings, bool supportSparse, string separator, bool hasHeader, bool trimWhitespace)
        {
            Columns = columns;
            AllowQuotedStrings = allowQuotedStrings;
            SupportSparse = supportSparse;
            Separator = separator;
            HasHeader = hasHeader;
            TrimWhitespace = trimWhitespace;
        }

        internal TextLoader BuildTextLoader()
        {
            var context = new MLContext();
            return new TextLoader(context, new TextLoader.Arguments() {
                AllowQuoting = AllowQuotedStrings,
                AllowSparse = SupportSparse,
                Column = Columns.Select(c => c.Item1).ToArray(),
                Separator = Separator,
                HasHeader = HasHeader,
                TrimWhitespace = TrimWhitespace
            });
        }
    }

    public class AutoFitSettings
    {
        public ExperimentStoppingCriteria StoppingCriteria = new ExperimentStoppingCriteria();
        internal IterationStoppingCriteria IterationStoppingCriteria;
        internal Concurrency Concurrency;
        internal Filters Filters;
        internal CrossValidationSettings CrossValidationSettings;
        internal OptimizingMetric OptimizingMetric;
        internal bool EnableEnsembling;
        internal bool EnableModelExplainability;
        internal bool EnableAutoTransformation;

        // spec question: Are following automatic or a user setting?
        internal bool EnableSubSampling;
        internal bool EnableCaching;
        internal bool ExternalizeTraining;
        internal TraceLevel TraceLevel; // Should this be controlled through code or appconfig?
    }

    public class ExperimentStoppingCriteria
    {
        public int MaxIterations = 100;
        public int TimeOutInMinutes = 300;
        internal bool StopAfterConverging;
        internal double ExperimentExitScore;
    }

    internal class Filters
    {
        internal IEnumerable<Trainers> WhitelistTrainers;
        internal IEnumerable<Trainers> BlackListTrainers;
        internal IEnumerable<Transformers> WhitelistTransformers;
        internal IEnumerable<Transformers> BlacklistTransformers;
        internal bool PreferExplainability;
        internal bool PreferInferenceSpeed;
        internal bool PreferSmallDeploymentSize;
        internal bool PreferSmallMemoryFootprint;
    }

    public class IterationStoppingCriteria
    {
        internal int TimeOutInSeconds;
        internal bool TerminateOnLowAccuracy;
    }

    public class Concurrency
    {
        internal int MaxConcurrentIterations;
        internal int MaxCoresPerIteration;
    }

    internal enum Trainers
    {
    }

    internal enum Transformers
    {
    }

    internal class CrossValidationSettings
    {
        internal int NumberOfFolds;
        internal int ValidationSizePercentage;
        internal IEnumerable<string> StratificationColumnNames;
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

    public enum InferenceType
    {
        Seperator,
        Header,
        Label,
        Task,
        ColumnDataKind,
        ColumnPurpose,
        Tranform,
        Trainer,
        Hyperparams,
        ColumnSplit
    }

    public class InferenceException : Exception
    {
        public InferenceType InferenceType;

        public InferenceException(InferenceType inferenceType, string message)
        : base(message)
        {
        }

        public InferenceException(InferenceType inferenceType, string message, Exception inner)
            : base(message, inner)
        {
        }
    }

    public class Pipeline
    {
        public readonly PipelineNode[] Elements;

        public Pipeline(PipelineNode[] elements)
        {
            Elements = elements;
        }
    }

    public class PipelineNode
    {
        public readonly string Name;
        public readonly PipelineNodeType ElementType;
        public readonly string[] InColumns;
        public readonly string[] OutColumns;
        public readonly IDictionary<string, object> Properties;

        public PipelineNode(string name, PipelineNodeType elementType,
            string[] inColumns, string[] outColumns,
            IDictionary<string, object> properties)
        {
            Name = name;
            ElementType = elementType;
            InColumns = inColumns;
            OutColumns = outColumns;
            Properties = properties;
        }
    }

    public enum PipelineNodeType
    {
        Transform,
        Trainer
    }
}
