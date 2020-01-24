// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.AutoML
{
    internal class SuggestedPipelineRunDetail
    {
        public readonly SuggestedPipeline Pipeline;
        public readonly bool RunSucceeded;
        public readonly double Score;

        public Exception Exception { get; set; }

        public SuggestedPipelineRunDetail(SuggestedPipeline pipeline, double score, bool runSucceeded)
        {
            Pipeline = pipeline;
            Score = score;
            RunSucceeded = runSucceeded;
        }

        public static SuggestedPipelineRunDetail FromPipelineRunResult(MLContext context, PipelineScore pipelineRunResult)
        {
            return new SuggestedPipelineRunDetail(SuggestedPipeline.FromPipeline(context, pipelineRunResult.Pipeline), pipelineRunResult.Score, pipelineRunResult.RunSucceeded);
        }

        public IRunResult ToRunResult(bool isMetricMaximizing)
        {
            return new RunResult(Pipeline.Trainer.HyperParamSet, Score, isMetricMaximizing);
        }
    }

    internal class SuggestedPipelineRunDetail<TMetrics> : SuggestedPipelineRunDetail
    {
        public readonly TMetrics ValidationMetrics;
        public readonly ModelContainer ModelContainer;

        internal SuggestedPipelineRunDetail(SuggestedPipeline pipeline,
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

        public RunDetail<TMetrics> ToIterationResult(IEstimator<ITransformer> preFeaturizer)
        {
            var estimator = SuggestedPipelineRunDetailUtil.PrependPreFeaturizer(Pipeline.ToEstimator(), preFeaturizer);
            return new RunDetail<TMetrics>(Pipeline.Trainer.TrainerName.ToString(), estimator,
                Pipeline.ToPipeline(), ModelContainer, ValidationMetrics, Exception);
        }
    }
}
