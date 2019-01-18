// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class InferredPipelineRunResult
    {
        public readonly object EvaluatedMetrics;
        public readonly InferredPipeline Pipeline;
        public readonly double Score;
        public readonly IDataView ScoredValidationData;

        /// <summary>
        /// This setting is true if the pipeline run succeeded & ran to completion.
        /// Else, it is false if some exception was thrown before the run could complete.
        /// </summary>
        public readonly bool RunSucceded;

        public ITransformer Model { get; set; }

        public InferredPipelineRunResult(object evaluatedMetrics, ITransformer model, InferredPipeline pipeline, double score, IDataView scoredValidationData,
            bool runSucceeded = true)
        {
            EvaluatedMetrics = evaluatedMetrics;
            Model = model;
            Pipeline = pipeline;
            Score = score;
            ScoredValidationData = scoredValidationData;
            RunSucceded = runSucceeded;
        }

        public InferredPipelineRunResult(InferredPipeline pipeline, bool runSucceeded)
        {
            Pipeline = pipeline;
            RunSucceded = runSucceeded;
        }

        public static InferredPipelineRunResult FromPipelineRunResult(PipelineRunResult pipelineRunResult)
        {
            return new InferredPipelineRunResult(null, null, 
                InferredPipeline.FromPipeline(pipelineRunResult.Pipeline),
                pipelineRunResult.Score, null, pipelineRunResult.RunSucceded);
        }

        public IRunResult ToRunResult(bool isMetricMaximizing)
        {
           return new RunResult(Pipeline.Trainer.HyperParamSet, Score, isMetricMaximizing);
        }
    }
}
