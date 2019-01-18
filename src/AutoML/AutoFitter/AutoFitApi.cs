using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class AutoFitApi
    {
        public static (InferredPipelineRunResult[] allPipelines, InferredPipelineRunResult bestPipeline) Fit(IDataView trainData, 
            IDataView validationData, string label, AutoFitSettings settings, TaskKind task, OptimizingMetric metric, 
            IEnumerable<(string, ColumnPurpose)> purposeOverrides, IDebugLogger debugLogger)
        {
            // hack: init new MLContext
            var mlContext = new MLContext();

            var purposeOverridesDict = purposeOverrides?.ToDictionary(p => p.Item1, p => p.Item2);
            var optimizingMetricfInfo = new OptimizingMetricInfo(metric);

            // infer pipelines
            var autoFitter = new AutoFitter(mlContext, optimizingMetricfInfo, settings ?? new AutoFitSettings(), task,
                   label, trainData, validationData, purposeOverridesDict, debugLogger);
            var allPipelines = autoFitter.Fit();

            var bestScore = allPipelines.Max(p => p.Score);
            var bestPipeline = allPipelines.First(p => p.Score == bestScore);

            return (allPipelines, bestPipeline);
        }
    }
}
