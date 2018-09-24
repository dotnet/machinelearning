using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
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
            Add(DefaultConfig.Instance
                .With(GetJobDefinition().With(CreateToolchain()))
                .With(new ExtraMetricColumn())
                .With(MemoryDiagnoser.Default));

            UnionRule = ConfigUnionRule.AlwaysUseLocal;
        }

        protected virtual Job GetJobDefinition()
            => Job.Default
                .WithWarmupCount(1) // ML.NET benchmarks are typically CPU-heavy benchmarks, 1 warmup is usually enough
                .WithMaxIterationCount(20);

        /// <summary>
        /// we need our own toolchain because MSBuild by default does not copy recursive native dependencies to the output
        /// </summary>
        private IToolchain CreateToolchain()
        {
            var csProj = CsProjCoreToolchain.Current.Value;
            var tfm = NetCoreAppSettings.Current.Value.TargetFrameworkMoniker;

            return new Toolchain(
                tfm,
                new ProjectGenerator(tfm),
                csProj.Builder,
                csProj.Executor);
        }
    }

    public class TrainConfig : RecommendedConfig
    {
        protected override Job GetJobDefinition()
            => Job.Dry // the "Dry" job runs the benchmark exactly once, without any warmup to mimic real-world scenario
                  .WithLaunchCount(3); // BDN will run 3 dedicated processes, sequentially
    }
}
