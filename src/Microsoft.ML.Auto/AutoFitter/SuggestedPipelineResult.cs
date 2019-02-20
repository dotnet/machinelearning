// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Auto
{
    internal class SuggestedPipelineResult
    {
        public readonly SuggestedPipeline Pipeline;
        public readonly bool RunSucceded;
        public readonly double Score;

        public SuggestedPipelineResult(SuggestedPipeline pipeline, double score, bool runSucceeded)
        {
            Pipeline = pipeline;
            Score = score;
            RunSucceded = runSucceeded;
        }

        public static SuggestedPipelineResult FromPipelineRunResult(PipelineScore pipelineRunResult)
        {
            return new SuggestedPipelineResult(SuggestedPipeline.FromPipeline(pipelineRunResult.Pipeline), pipelineRunResult.Score, pipelineRunResult.RunSucceded);
        }

        public IRunResult ToRunResult(bool isMetricMaximizing)
        {
            return new RunResult(Pipeline.Trainer.HyperParamSet, Score, isMetricMaximizing);
        }
    }

    internal class SuggestedPipelineResult<T> : SuggestedPipelineResult
    {
        public readonly T EvaluatedMetrics;
        public ITransformer Model { get; set; }
        public Exception Exception { get; set; }

        public int RuntimeInSeconds { get; set; }
        public int PipelineInferenceTimeInSeconds { get; set; }

        public SuggestedPipelineResult(T evaluatedMetrics, ITransformer model, SuggestedPipeline pipeline, double score, Exception exception)
            : base(pipeline, score, exception == null)
        {
            EvaluatedMetrics = evaluatedMetrics;
            Model = model;
            Exception = exception;
        }

        public RunResult<T> ToIterationResult()
        {
            return new RunResult<T>(Model, EvaluatedMetrics, Pipeline.ToPipeline(), Exception, RuntimeInSeconds, PipelineInferenceTimeInSeconds);
        }
    }
}
