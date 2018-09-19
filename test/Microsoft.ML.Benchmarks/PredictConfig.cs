// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;

namespace Microsoft.ML.Benchmarks
{
    internal class PredictConfig : ManualConfig
    {
        public PredictConfig()
        {
            Add(DefaultConfig.Instance
                .With(Job.Default
                    .WithWarmupCount(1) // for our time consuming benchmarks 1 warmup iteration is enough
                    .WithMaxIterationCount(20)
                    .With(Program.CreateToolchain()))
                .With(new ExtraMetricColumn())
                .With(MemoryDiagnoser.Default));
        }
    }
}
