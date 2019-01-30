// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class AutoFitApi
    {
        public static (InferredPipelineRunResult[] allPipelines, InferredPipelineRunResult bestPipeline) Fit(IDataView trainData, 
            IDataView validationData, string label, AutoFitSettings settings, IEstimator<ITransformer> preFeaturizers, TaskKind task, OptimizingMetric metric, 
            IEnumerable<(string, ColumnPurpose)> purposeOverrides, IDebugLogger debugLogger)
        {
            // hack: init new MLContext
            var mlContext = new MLContext();

            ITransformer preprocessorTransform = null;
            if (preFeaturizers != null)
            {
                // preprocess train and validation data
                preprocessorTransform = preFeaturizers.Fit(trainData);
                trainData = preprocessorTransform.Transform(trainData);
                validationData = preprocessorTransform.Transform(validationData);
            }

            var purposeOverridesDict = purposeOverrides?.ToDictionary(p => p.Item1, p => p.Item2);
            var optimizingMetricfInfo = new OptimizingMetricInfo(metric);

            // infer pipelines
            var autoFitter = new AutoFitter(mlContext, optimizingMetricfInfo, settings ?? new AutoFitSettings(), task,
                   label, trainData, validationData, purposeOverridesDict, debugLogger);
            var allPipelines = autoFitter.Fit();

            // apply preprocessor to returned models
            if (preprocessorTransform != null)
            {
                for (var i = 0; i < allPipelines.Length; i++)
                {
                    allPipelines[i].Model = preprocessorTransform.Append(allPipelines[i].Model);
                }
            }

            var bestScore = allPipelines.Max(p => p.Score);
            var bestPipeline = allPipelines.First(p => p.Score == bestScore);

            return (allPipelines, bestPipeline);
        }
    }
}
