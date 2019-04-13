// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Auto
{
    public sealed class RunDetail<TMetrics> : RunDetail
    {
        public TMetrics ValidationMetrics { get; private set; }
        public ITransformer Model { get { return _modelContainer.GetModel(); } }
        public Exception Exception { get; private set; }

        private readonly ModelContainer _modelContainer;

        internal RunDetail(string trainerName,
            IEstimator<ITransformer> estimator,
            Pipeline pipeline,
            ModelContainer modelContainer,
            TMetrics metrics,
            Exception exception) : base(trainerName, estimator, pipeline)
        {
            _modelContainer = modelContainer;
            ValidationMetrics = metrics;
            Exception = exception;
        }
    }

    public abstract class RunDetail
    {
        public string TrainerName { get; private set; }
        public double RuntimeInSeconds { get; internal set; }
        public IEstimator<ITransformer> Estimator { get; private set; }

        internal Pipeline Pipeline { get; private set; }
        internal double PipelineInferenceTimeInSeconds { get; set; }

        internal RunDetail(string trainerName,
            IEstimator<ITransformer> estimator,
            Pipeline pipeline)
        {
            TrainerName = trainerName;
            Estimator = estimator;
            Pipeline = pipeline;
        }
    }
}
