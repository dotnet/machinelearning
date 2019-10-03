// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Toolchains;
using BenchmarkDotNet.Toolchains.CsProj;
using BenchmarkDotNet.Toolchains.DotNetCli;
using Microsoft.ML.Benchmarks.Harness;

namespace Microsoft.ML.Benchmarks
{
    public class RecommendedConfig : ManualConfig
    {
        protected static readonly IReadOnlyList<MsBuildArgument> msbuildArguments = new List<MsBuildArgument>() { new MsBuildArgument($"/p:Configuration={GetBuildConfigurationName()}") };

        public RecommendedConfig()
        {
            Add(DefaultConfig.Instance); // this config contains all of the basic settings (exporters, columns etc)

            Add(GetJobDefinition() // job defines how many times given benchmark should be executed
                .With(msbuildArguments)
                .With(CreateToolchain())); // toolchain is responsible for generating, building and running dedicated executable per benchmark

            Add(new ExtraMetricColumn()); // an extra colum that can display additional metric reported by the benchmarks
        }

        protected virtual Job GetJobDefinition()
            => Job.Default
                .WithWarmupCount(1) // ML.NET benchmarks are typically CPU-heavy benchmarks, 1 warmup is usually enough
                .WithMaxIterationCount(20)
                .AsDefault(); // this way we tell BDN that it's a default config which can be overwritten

        /// <summary>
        /// we need our own toolchain because MSBuild by default does not copy recursive native dependencies to the output
        /// </summary>
        private IToolchain CreateToolchain()
        {
#if NETFRAMEWORK
            var tfm = "net461";
            var csProj = CsProjClassicNetToolchain.Net461;
#else
            var tfm = NetCoreAppSettings.Current.Value.TargetFrameworkMoniker;
            var csProj = CsProjCoreToolchain.Current.Value;
#endif
            return new Toolchain(
                tfm,
                new ProjectGenerator(tfm), // custom generator that copies native dependencies
                csProj.Builder,
                csProj.Executor);
        }

        private static string GetBuildConfigurationName()
        {
#if NETCOREAPP3_0
            return "Release-netcoreapp3_0";
#elif NET461
            return "Release-netfx";
#else
            return "Release";
#endif
        }
    }

    public class TrainConfig : RecommendedConfig
    {
        protected override Job GetJobDefinition()
            => Job.Dry // the "Dry" job runs the benchmark exactly once, without any warmup to mimic real-world scenario
                  .With(msbuildArguments)
                  .WithLaunchCount(3); // BDN will run 3 dedicated processes, sequentially
    }
}
