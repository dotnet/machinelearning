// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;

namespace Microsoft.ML.Auto
{
    public sealed class RunResult<T>
    {
        public T ValidationMetrics { get; private set; }
        public ITransformer Model { get; private set; }
        public Exception Exception { get; private set; }
        public string TrainerName { get; private set; }
        public int RuntimeInSeconds { get; private set; }

        internal Pipeline Pipeline { get; private set; }
        internal int PipelineInferenceTimeInSeconds { get; private set; }

        internal RunResult(
            ITransformer model,
            T metrics,
            Pipeline pipeline,
            Exception exception,
            int runtimeInSeconds,
            int pipelineInferenceTimeInSeconds)
        {
            Model = model;
            ValidationMetrics = metrics;
            Pipeline = pipeline;
            Exception = exception;
            RuntimeInSeconds = runtimeInSeconds;
            PipelineInferenceTimeInSeconds = pipelineInferenceTimeInSeconds;

            TrainerName = pipeline?.Nodes.Where(n => n.NodeType == PipelineNodeType.Trainer).Last().Name;
        }
    }
}
