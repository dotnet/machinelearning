// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains.InProcess;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    class Program
    {
        public static void Main(string[] args)
        {
            BenchmarkSwitcher
                .FromAssembly(typeof(Program).Assembly)
                .Run(null, CreateClrVsCoreConfig());
        }

        private static IConfig CreateClrVsCoreConfig()
        {
            var config = DefaultConfig.Instance.With(
                Job.ShortRun.
                With(InProcessToolchain.Instance));
            return config;
        }
    }
}
