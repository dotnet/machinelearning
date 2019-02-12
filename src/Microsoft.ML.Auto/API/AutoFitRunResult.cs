// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Auto
{
    public class AutoFitRunResult<T>
    {
        public readonly T Metrics;
        public readonly ITransformer Model;
        public readonly Exception Exception;
        public readonly string TrainerName;
        public readonly int RuntimeInSeconds;

        internal readonly Pipeline Pipeline;
        internal readonly int PipelineInferenceTimeInSeconds;

        internal AutoFitRunResult(
            ITransformer model,
            T metrics,
            Pipeline pipeline,
            Exception exception,
            int runtimeInSeconds,
            int pipelineInferenceTimeInSeconds)
        {
            Model = model;
            Metrics = metrics;
            Pipeline = pipeline;
            Exception = exception;
            RuntimeInSeconds = runtimeInSeconds;
            PipelineInferenceTimeInSeconds = pipelineInferenceTimeInSeconds;

            TrainerName = pipeline?.Nodes.Where(n => n.NodeType == PipelineNodeType.Trainer).Last().Name;
        }
    }
}
