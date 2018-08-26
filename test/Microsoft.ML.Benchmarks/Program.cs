// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Toolchains;
using BenchmarkDotNet.Toolchains.CsProj;
using BenchmarkDotNet.Toolchains.DotNetCli;
using BenchmarkDotNet.Toolchains.InProcess;
using BenchmarkDotNet.Validators;
using Microsoft.ML.Benchmarks.Harness;
using System.IO;
using Microsoft.ML.Models;

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
                .With(new ClassificationMetricsColumn("AccuracyMacro", "Macro-average accuracy of the model"))
                    .With(CreateToolchain()))
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

    public class ClassificationMetricsColumn : IColumn
    {
        private readonly string _metricName;
        private readonly string _legend;

        public ClassificationMetricsColumn(string metricName, string legend)
        {
            _metricName = metricName;
            _legend = legend;
        }

        public string ColumnName => _metricName;
        public string Id => _metricName;
        public string Legend => _legend;
        public bool IsNumeric => true;
        public bool IsDefault(Summary summary, BenchmarkCase benchmark) => true;
        public bool IsAvailable(Summary summary) => true;
        public bool AlwaysShow => true;
        public ColumnCategory Category => ColumnCategory.Custom;
        public int PriorityInCategory => 1;
        public UnitType UnitType => UnitType.Dimensionless;

        public string GetValue(Summary summary, BenchmarkCase benchmark, ISummaryStyle style)
        {
            var property = typeof(ClassificationMetrics).GetProperty(_metricName);
            return property.GetValue(StochasticDualCoordinateAscentClassifierBench.s_metrics).ToString();
        }
        public string GetValue(Summary summary, BenchmarkCase benchmark) => GetValue(summary, benchmark, null);

        public override string ToString() => ColumnName;
    }
}
