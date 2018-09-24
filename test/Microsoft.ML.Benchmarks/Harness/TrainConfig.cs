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
                .With(Job.Dry // the "Dry" job runs the benchmark exactly once, without any warmup to mimic real-world scenario
                    .WithLaunchCount(3)  // BDN will run 3 dedicated processes, sequentially
                    .With(Program.CreateToolchain()))
                .With(new ExtraMetricColumn())
                .With(MemoryDiagnoser.Default));
        }
    }
}
