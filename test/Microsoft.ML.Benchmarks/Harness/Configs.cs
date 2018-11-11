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
        public RecommendedConfig()
        {
            Add(DefaultConfig.Instance); // this config contains all of the basic settings (exporters, columns etc)

            Add(GetJobDefinition() // job defines how many times given benchmark should be executed
                .WithCustomBuildConfiguration(GetBuildConfigurationName())
                .With(CreateToolchain())); // toolchain is responsible for generating, building and running dedicated executable per benchmark);

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
        protected IToolchain CreateToolchain()
        {
            var tfm = GetTargetFrameworkMoniker();
            var csProj = CsProjCoreToolchain.From(new NetCoreAppSettings(targetFrameworkMoniker: tfm, runtimeFrameworkVersion: null, name: tfm));

            return new Toolchain(
                tfm,
                new ProjectGenerator(tfm), // custom generator that copies native dependencies
                csProj.Builder,
                csProj.Executor);
        }

        private static string GetTargetFrameworkMoniker()
        {
#if NETCOREAPP3_0 // todo: remove the #IF DEFINES when BDN 0.11.2 gets released (BDN gains the 3.0 support)
            return "netcoreapp3.0";
#else
            return NetCoreAppSettings.Current.Value.TargetFrameworkMoniker;
#endif
        }

        protected static string GetBuildConfigurationName()
        {
#if NETCOREAPP3_0
            return "Release-Intrinsics";
#else
            return "Release";
#endif
        }
    }

    public class TrainConfig : RecommendedConfig
    {
        protected override Job GetJobDefinition()
            => Job.Dry // the "Dry" job runs the benchmark exactly once, without any warmup to mimic real-world scenario
                  .WithLaunchCount(3); // BDN will run 3 dedicated processes, sequentially
    }
}
