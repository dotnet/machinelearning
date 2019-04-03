// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Auto
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

    internal sealed class SuggestedPipelineCrossValRunDetails<TMetrics> : SuggestedPipelineRunDetails
    {
        public readonly IEnumerable<SuggestedPipelineTrainResult<TMetrics>> Results;

        internal SuggestedPipelineCrossValRunDetails(SuggestedPipeline pipeline,
            double score,
            bool runSucceeded,
            IEnumerable<SuggestedPipelineTrainResult<TMetrics>> results) : base(pipeline, score, runSucceeded)
        {
            Results = results;
        }

        public CrossValidationRunDetails<TMetrics> ToIterationResult()
        {
            return new CrossValidationRunDetails<TMetrics>(Pipeline.Trainer.TrainerName.ToString(), Pipeline.ToEstimator(),
                Pipeline.ToPipeline(), Results.Select(r => r.ToTrainResult()));
        }
    }
}
