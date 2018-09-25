// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;

namespace Microsoft.ML.Benchmarks
{
    public class TrainConfig : ManualConfig
    {
        public TrainConfig()
        {
            Add(DefaultConfig.Instance
                .With(Job.Default
                    .WithWarmupCount(0)
                    .WithIterationCount(1)
                    .WithLaunchCount(3)  // BDN will start 3 dedicated processes, each of them will just run given benchmark once, without any warm up to mimic the real world.
                    .With(Program.CreateToolchain()))
                .With(new ExtraMetricColumn())
                .With(MemoryDiagnoser.Default));
        }
    }
}
