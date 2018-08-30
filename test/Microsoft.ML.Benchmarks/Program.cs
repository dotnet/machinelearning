// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Toolchains.InProcess;
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
                    .With(InProcessToolchain.Instance))
                .With(new ClassificationMetricsColumn("AccuracyMacro", "Macro-average accuracy of the model"))
                .With(MemoryDiagnoser.Default);

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
