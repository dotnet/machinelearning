// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.AutoML
{
    internal sealed class SuggestedPipelineTrainResult<TMetrics>
    {
        public readonly TMetrics ValidationMetrics;
        public readonly ModelContainer ModelContainer;
        public readonly Exception Exception;
        public readonly double Score;

        internal SuggestedPipelineTrainResult(ModelContainer modelContainer,
            TMetrics metrics,
            Exception exception,
            double score)
        {
            ModelContainer = modelContainer;
            ValidationMetrics = metrics;
            Exception = exception;
            Score = score;
        }

        public TrainResult<TMetrics> ToTrainResult()
        {
            return new TrainResult<TMetrics>(ModelContainer, ValidationMetrics, Exception);
        }
    }

    internal sealed class SuggestedPipelineCrossValRunDetail<TMetrics> : SuggestedPipelineRunDetail
    {
        public readonly IEnumerable<SuggestedPipelineTrainResult<TMetrics>> Results;

        internal SuggestedPipelineCrossValRunDetail(SuggestedPipeline pipeline,
            double score,
            bool runSucceeded,
            IEnumerable<SuggestedPipelineTrainResult<TMetrics>> results) : base(pipeline, score, runSucceeded)
        {
            Results = results;
            Exception = Results.Select(r => r.Exception).FirstOrDefault(e => e != null);
        }

        public CrossValidationRunDetail<TMetrics> ToIterationResult(IEstimator<ITransformer> preFeaturizer)
        {
            var estimator = SuggestedPipelineRunDetailUtil.PrependPreFeaturizer(Pipeline.ToEstimator(), preFeaturizer);
            return new CrossValidationRunDetail<TMetrics>(Pipeline.Trainer.TrainerName.ToString(), estimator,
                Pipeline.ToPipeline(), Results.Select(r => r.ToTrainResult()));
        }
    }
}
