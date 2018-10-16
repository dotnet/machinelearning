// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Running;
using Microsoft.ML.Runtime.Internal.CpuMath;
using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Benchmarks.Tests
{
    public class TestConfig : RecommendedConfig
    {
        protected override Job GetJobDefinition() => Job.Dry; // Job.Dry runs the benchmark just once
    }

    public class BenchmarkTouchingNativeDependency
    {
        [Benchmark]
        public float Simple() => CpuMathUtils.Sum(Enumerable.Range(0, 1024).Select(Convert.ToSingle).ToArray());
    }

    public class BenchmarksTest
    {
        private const string SkipTheDebug =
#if DEBUG
            "BenchmarkDotNet does not allow running the benchmarks in Debug, so this test is disabled for DEBUG";
#else
            "";
#endif

        public BenchmarksTest(ITestOutputHelper output) => Output = output;

        private ITestOutputHelper Output { get; }

        [Fact(Skip = SkipTheDebug)]
        public void BenchmarksProjectIsNotBroken()
        {
            var summary = BenchmarkRunner.Run<BenchmarkTouchingNativeDependency>(new TestConfig().With(new OutputLogger(Output)));

            Assert.False(summary.HasCriticalValidationErrors, "The \"Summary\" should have NOT \"HasCriticalValidationErrors\"");

            Assert.True(summary.Reports.Any(), "The \"Summary\" should contain at least one \"BenchmarkReport\" in the \"Reports\" collection");

            Assert.True(summary.Reports.All(r => r.BuildResult.IsBuildSuccess),
                "The following benchmarks are failed to build: " +
                string.Join(", ", summary.Reports.Where(r => !r.BuildResult.IsBuildSuccess).Select(r => r.BenchmarkCase.DisplayInfo)));

            Assert.True(summary.Reports.All(r => r.ExecuteResults != null),
                "The following benchmarks don't have any execution results: " +
                string.Join(", ", summary.Reports.Where(r => r.ExecuteResults == null).Select(r => r.BenchmarkCase.DisplayInfo)));

            Assert.True(summary.Reports.All(r => r.ExecuteResults.Any(er => er.FoundExecutable && er.Data.Any())),
                "All reports should have at least one \"ExecuteResult\" with \"FoundExecutable\" = true and at least one \"Data\" item");

            Assert.True(summary.Reports.All(report => report.AllMeasurements.Any()),
                "All reports should have at least one \"Measurement\" in the \"AllMeasurements\" collection");
        }
    }

    public class OutputLogger : AccumulationLogger
    {
        private readonly ITestOutputHelper testOutputHelper;
        private string currentLine = "";

        public OutputLogger(ITestOutputHelper testOutputHelper)
        {
            this.testOutputHelper = testOutputHelper ?? throw new ArgumentNullException(nameof(testOutputHelper));
        }

        public override void Write(LogKind logKind, string text)
        {
            currentLine += text;
            base.Write(logKind, text);
        }

        public override void WriteLine()
        {
            testOutputHelper.WriteLine(currentLine);
            currentLine = "";
            base.WriteLine();
        }

        public override void WriteLine(LogKind logKind, string text)
        {
            testOutputHelper.WriteLine(currentLine + text);
            currentLine = "";
            base.WriteLine(logKind, text);
        }
    }
}
