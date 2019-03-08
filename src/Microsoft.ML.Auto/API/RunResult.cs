// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public sealed class RunResult<T>
    {
        public T ValidationMetrics { get; private set; }
        public ITransformer Model { get { return _modelContainer.GetModel(); } }
        public Exception Exception { get; private set; }
        public string TrainerName { get; private set; }
        public int RuntimeInSeconds { get; private set; }
        public IEstimator<ITransformer> Estimator { get; private set; }

        internal Pipeline Pipeline { get; private set; }
        internal int PipelineInferenceTimeInSeconds { get; private set; }

        private readonly ModelContainer _modelContainer;

        internal RunResult(ModelContainer modelContainer,
            T metrics,
            IEstimator<ITransformer> estimator,
            Pipeline pipeline,
            Exception exception,
            int runtimeInSeconds,
            int pipelineInferenceTimeInSeconds)
        {
            _modelContainer = modelContainer;
            ValidationMetrics = metrics;
            Pipeline = pipeline;
            Estimator = estimator;
            Exception = exception;
            RuntimeInSeconds = runtimeInSeconds;
            PipelineInferenceTimeInSeconds = pipelineInferenceTimeInSeconds;

            TrainerName = pipeline?.Nodes.Where(n => n.NodeType == PipelineNodeType.Trainer).Last().Name;
        }
    }
}
