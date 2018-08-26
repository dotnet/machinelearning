// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains;
using BenchmarkDotNet.Toolchains.CsProj;
using BenchmarkDotNet.Toolchains.DotNetCli;
using Microsoft.ML.Benchmarks.Harness;
using System.IO;

namespace Microsoft.ML.Benchmarks
{
    class Program
    {
        /// <summary>
        /// execute dotnet run -c Release and choose the benchmarks you want to run
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args) 
            => BenchmarkSwitcher
                .FromAssembly(typeof(Program).Assembly)
                .Run(args, CreateCustomConfig());

        private static IConfig CreateCustomConfig() 
            => DefaultConfig.Instance
                .With(Job.Default
                    .WithMaxIterationCount(20)
                    .With(CreateToolchain()))
                .With(new ExtraMetricColumn())
                .With(MemoryDiagnoser.Default);

        private static IToolchain CreateToolchain()
        {
            var csProj = CsProjCoreToolchain.Current.Value;
            var tfm = NetCoreAppSettings.Current.Value.TargetFrameworkMoniker;

            return new Toolchain(
                tfm, 
                new ProjectGenerator(tfm), 
                csProj.Builder, 
                csProj.Executor);
        }

        internal static string GetDataPath(string name)
            => Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), "Input", name);
    }
}
