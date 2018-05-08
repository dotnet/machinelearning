// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains.CsProj;
using BenchmarkDotNet.Toolchains.InProcess;
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
        {
            BenchmarkSwitcher
                .FromAssembly(typeof(Program).Assembly)
                .Run(null, CreateClrVsCoreConfig());
        }

        private static IConfig CreateClrVsCoreConfig()
        {
            var config = DefaultConfig.Instance.With(
                Job.ShortRun.
                With(InProcessToolchain.Instance)).
                With(MemoryDiagnoser.Default);
            return config;
        }

        internal static string GetDataPath(string name)
            => Path.GetFullPath(Path.Combine(_dataRoot, name));

        static readonly string _dataRoot;
        static Program()
        {
            var currentAssemblyLocation = new FileInfo(typeof(Program).Assembly.Location);
            var rootDir = currentAssemblyLocation.Directory.Parent.Parent.Parent.Parent.FullName;
            _dataRoot = Path.Combine(rootDir, "test", "data");
        }
    }
}
