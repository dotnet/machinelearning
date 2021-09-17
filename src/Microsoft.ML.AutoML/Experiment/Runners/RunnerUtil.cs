// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    internal static class RunnerUtil
    {
        public static (ModelContainer model, TMetrics metrics, Exception exception, double score)
            TrainAndScorePipeline<TMetrics>(MLContext context,
            SuggestedPipeline pipeline,
            IDataView trainData,
            IDataView validData,
            string groupId,
            string labelColumn,
            IMetricsAgent<TMetrics> metricsAgent,
            ITransformer preprocessorTransform,
            FileInfo modelFileInfo,
            DataViewSchema modelInputSchema,
            IChannel logger) where TMetrics : class
        {
            try
            {
                var estimator = pipeline.ToEstimator(trainData, validData);
                var model = estimator.Fit(trainData);

                var scoredData = model.Transform(validData);
                var metrics = metricsAgent.EvaluateMetrics(scoredData, labelColumn, groupId);
                var score = metricsAgent.GetScore(metrics);

                if (preprocessorTransform != null)
                {
                    model = preprocessorTransform.Append(model);
                }

                // Build container for model
                var modelContainer = modelFileInfo == null ?
                    new ModelContainer(context, model) :
                    new ModelContainer(context, modelFileInfo, model, modelInputSchema);

                return (modelContainer, metrics, null, score);
            }
            catch (Exception ex)
            {
                logger.Error($"Pipeline crashed: {pipeline.ToString()} . Exception: {ex}");
                return (null, null, ex, double.NaN);
            }
        }

        public static FileInfo GetModelFileInfo(DirectoryInfo modelDirectory, int iterationNum, int foldNum)
        {
            return modelDirectory == null ?
                null :
                new FileInfo(Path.Combine(modelDirectory.FullName, $"Model{iterationNum}_{foldNum}.zip"));
        }
    }
}
