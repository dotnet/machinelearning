// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Running;
using micro;
using Microsoft.ML.Benchmarks.Harness;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Benchmarks.Tests
{
    public class TestConfig : RecommendedConfig
    {
        protected override Job GetJobDefinition() => Job.Dry; // Job.Dry runs the benchmark just once
    }

    public class BenchmarksTest
    {
        public BenchmarksTest(ITestOutputHelper output) //: base(output)
        {
            this.output = output;
        }

        private ITestOutputHelper output { get; }

        public static IList<Type> GetBenchmarks()
        {
            IList<Type> benchmarks = new List<Type>();
            Assembly asm = typeof(StochasticDualCoordinateAscentClassifierBench).Assembly;

            var types = from type in asm.GetTypes()
                        where Attribute.IsDefined(type, typeof(CIBenchmark))
                        select type;

            foreach (Type type in types)
            {
                benchmarks.Add(type);
            }
            return benchmarks;
        }

        private void BenchmarksProjectIsNotBroken()
        {
            var types = GetBenchmarks();
            foreach (var type in types)
            {
                var config = new TestConfig();
                var c = config.With(new OutputLogger(output));
                var summary = BenchmarkRunner.Run(type, c);
                

                VisualStudio.TestTools.UnitTesting.Assert.IsFalse(summary.HasCriticalValidationErrors, "The \"Summary\" should have NOT \"HasCriticalValidationErrors\"");

                VisualStudio.TestTools.UnitTesting.Assert.IsTrue(summary.Reports.Any(), "The \"Summary\" should contain at least one \"BenchmarkReport\" in the \"Reports\" collection");

                VisualStudio.TestTools.UnitTesting.Assert.IsTrue(summary.Reports.All(r => r.BuildResult.IsBuildSuccess),
                    "The following benchmarks failed to build: " +
                    string.Join(", ", summary.Reports.Where(r => !r.BuildResult.IsBuildSuccess).Select(r => r.BenchmarkCase.DisplayInfo)));

                VisualStudio.TestTools.UnitTesting.Assert.IsTrue(summary.Reports.All(r => r.ExecuteResults != null),
                    "The following benchmarks don't have any execution results: " +
                    string.Join(", ", summary.Reports.Where(r => r.ExecuteResults == null).Select(r => r.BenchmarkCase.DisplayInfo)));

                VisualStudio.TestTools.UnitTesting.Assert.IsTrue(summary.Reports.All(r => r.ExecuteResults.Any(er => er.FoundExecutable && er.Data.Any())),
                    "All reports should have at least one \"ExecuteResult\" with \"FoundExecutable\" = true and at least one \"Data\" item");

                VisualStudio.TestTools.UnitTesting.Assert.IsTrue(summary.Reports.All(report => report.AllMeasurements.Any()),
                    "All reports should have at least one \"Measurement\" in the \"AllMeasurements\" collection");
            }
        }

        [Theory]
        [IterationData(iterations:10)]
        [Trait("Category", "RunSpecificTest")]
        public void CompletesBenchmarkInTime(int iterations)
        {
            //Output.WriteLine($"{iterations} - th");

            int timeout = 20 * 60 * 1000;

            var runTask = Task.Run(BenchmarksProjectIsNotBroken);
            var timeoutTask = Task.Delay(timeout + iterations);

            var finishedTask = Task.WhenAny(timeoutTask, runTask).Result;
            if (finishedTask == timeoutTask)
            {
                Console.WriteLine("Benchmark Hanging: fail to complete in 20 minutes");
                Environment.FailFast("Fail here to take memory dump");
            }
        }
    }

    public class OutputLogger : AccumulationLogger
    {
        private readonly ITestOutputHelper _testOutputHelper;
        private string _currentLine = "";

        public OutputLogger(ITestOutputHelper testOutputHelper)
        {
            this._testOutputHelper = testOutputHelper ?? throw new ArgumentNullException(nameof(testOutputHelper));
        }

        public override void Write(LogKind logKind, string text)
        {
            _currentLine += text;
            base.Write(logKind, text);
            Flush();
        }

        public override void WriteLine()
        {
            _testOutputHelper.WriteLine(_currentLine);
            _currentLine = "";
            base.WriteLine();
            Flush();
        }

        public override void WriteLine(LogKind logKind, string text)
        {
            _testOutputHelper.WriteLine(_currentLine + text);
            _currentLine = "";
            base.WriteLine(logKind, text);
            Flush();
        }
    }
}
