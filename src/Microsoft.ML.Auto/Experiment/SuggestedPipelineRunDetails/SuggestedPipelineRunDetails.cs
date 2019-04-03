// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Auto
{
    internal class SuggestedPipelineRunDetails
    {
        public readonly SuggestedPipeline Pipeline;
        public readonly bool RunSucceded;
        public readonly double Score;

        public SuggestedPipelineRunDetails(SuggestedPipeline pipeline, double score, bool runSucceeded)
        {
            Pipeline = pipeline;
            Score = score;
            RunSucceded = runSucceeded;
        }

        public static SuggestedPipelineRunDetails FromPipelineRunResult(MLContext context, PipelineScore pipelineRunResult)
        {
            return new SuggestedPipelineRunDetails(SuggestedPipeline.FromPipeline(context, pipelineRunResult.Pipeline), pipelineRunResult.Score, pipelineRunResult.RunSucceded);
        }

        public IRunResult ToRunResult(bool isMetricMaximizing)
        {
            return new RunResult(Pipeline.Trainer.HyperParamSet, Score, isMetricMaximizing);
        }
    }

    internal class SuggestedPipelineRunDetails<TMetrics> : SuggestedPipelineRunDetails
    {
        public readonly TMetrics ValidationMetrics;
        public readonly ModelContainer ModelContainer;
        public readonly Exception Exception;

        internal SuggestedPipelineRunDetails(SuggestedPipeline pipeline,
            double score,
            bool runSucceeded,
            TMetrics validationMetrics,
            ModelContainer modelContainer,
            Exception ex) : base(pipeline, score, runSucceeded)
        {
            ValidationMetrics = validationMetrics;
            ModelContainer = modelContainer;
            Exception = ex;
        }

        public RunDetails<TMetrics> ToIterationResult()
        {
            return new RunDetails<TMetrics>(Pipeline.Trainer.TrainerName.ToString(), Pipeline.ToEstimator(),
                Pipeline.ToPipeline(), ModelContainer, ValidationMetrics, Exception);
        }
    }
}
