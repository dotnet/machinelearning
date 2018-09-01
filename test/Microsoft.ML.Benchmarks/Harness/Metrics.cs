// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Benchmarks
{
    public abstract class WithExtraMetrics
    {
        protected abstract IEnumerable<Metric> GetMetrics();

        /// <summary>
        ///  this method is executed after running the benchmrks
        ///  we use it as hack to simply print to console so ExtraMetricColumn can parse the output
        /// </summary>
        [GlobalCleanup]
        public void ReportMetrics()
        {
            foreach (var metric in GetMetrics())
            {
                Console.WriteLine(metric.ToParsableString());
            }
        }
    }

    public class ExtraMetricColumn : IColumn
    {
        public string ColumnName => "Extra Metric";
        public string Id => nameof(ExtraMetricColumn);
        public string Legend => "Value of the provided extra metric";
        public bool IsNumeric => true;
        public bool IsDefault(Summary summary, BenchmarkCase benchmark) => true;
        public bool IsAvailable(Summary summary) => true;
        public bool AlwaysShow => true;
        public ColumnCategory Category => ColumnCategory.Custom;
        public int PriorityInCategory => 1;
        public UnitType UnitType => UnitType.Dimensionless;
        public string GetValue(Summary summary, BenchmarkCase benchmark) => GetValue(summary, benchmark, null);
        public override string ToString() => ColumnName;

        public string GetValue(Summary summary, BenchmarkCase benchmark, ISummaryStyle style)
        {
            if (!summary.HasReport(benchmark))
                return "-";

            var results = summary[benchmark].ExecuteResults;
            if (results.Count != 1)
                return "-";

            var result = results.Single();
            var buffer = new StringBuilder();

            foreach (var line in result.ExtraOutput)
            {
                if (Metric.TryParse(line, out Metric metric))
                {
                    if (buffer.Length > 0)
                        buffer.Append(", ");

                    buffer.Append(metric.ToColumnValue());
                }
            }

            return buffer.Length > 0 ? buffer.ToString() : "-";
        }
    }

    public struct Metric
    {
        private const string Prefix = "// Metric";
        private const char Separator = '#';

        public string Name { get; }
        public string Value { get; }

        public Metric(string name, string value) : this()
        {
            Name = name;
            Value = value;
        }

        public string ToColumnValue()
            => $"{Name}: {Value}";

        public string ToParsableString()
            => $"{Prefix} {Separator} {Name} {Separator} {Value}";

        public static bool TryParse(string line, out Metric metric)
        {
            metric = default;

            if (!line.StartsWith(Prefix))
                return false;

            var splitted = line.Split(Separator);

            metric = new Metric(splitted[1].Trim(), splitted[2].Trim());

            return true;
        }
    }
}
