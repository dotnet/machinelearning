// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;

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

        public static SuggestedPipelineResult FromPipelineRunResult(MLContext context, PipelineScore pipelineRunResult)
        {
            return new SuggestedPipelineResult(SuggestedPipeline.FromPipeline(context, pipelineRunResult.Pipeline), pipelineRunResult.Score, pipelineRunResult.RunSucceded);
        }

        public IRunResult ToRunResult(bool isMetricMaximizing)
        {
            return new RunResult(Pipeline.Trainer.HyperParamSet, Score, isMetricMaximizing);
        }
    }

    internal class SuggestedPipelineResult<T> : SuggestedPipelineResult
    {
        public readonly T EvaluatedMetrics;
        public IEstimator<ITransformer> Estimator { get; set; }
        public ModelContainer ModelContainer { get; set; }
        public Exception Exception { get; set; }

        public double RuntimeInSeconds { get; set; }
        public double PipelineInferenceTimeInSeconds { get; set; }

        public SuggestedPipelineResult(T evaluatedMetrics, IEstimator<ITransformer> estimator,
            ModelContainer modelContainer, SuggestedPipeline pipeline, double score, Exception exception)
            : base(pipeline, score, exception == null)
        {
            EvaluatedMetrics = evaluatedMetrics;
            Estimator = estimator;
            ModelContainer = modelContainer;
            Exception = exception;
        }

        public RunResult<T> ToIterationResult()
        {
            return new RunResult<T>(ModelContainer, EvaluatedMetrics, Estimator, Pipeline.ToPipeline(), Exception, RuntimeInSeconds, PipelineInferenceTimeInSeconds);
        }
    }
}
