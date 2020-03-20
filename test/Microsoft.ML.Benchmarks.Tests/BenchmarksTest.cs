// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
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

    public class BenchmarksTest : BaseTestClass
    {
        public BenchmarksTest(ITestOutputHelper output) : base(output)
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

        [Fact]
        public void BenchmarksProjectIsNotBroken()
        {
            var types = GetBenchmarks();
            foreach (var type in types)
            {
                var summary = BenchmarkRunner.Run(type, new TestConfig().With(new OutputLogger(output)));

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

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupSentimentPipeline(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    PredictionEngineBench predictionEngineBench = new PredictionEngineBench();
        //    predictionEngineBench.SetupSentimentPipeline();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupIrisPipeline(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    PredictionEngineBench predictionEngineBench = new PredictionEngineBench();
        //    predictionEngineBench.SetupIrisPipeline();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupBreastCancerPipeline(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    PredictionEngineBench predictionEngineBench = new PredictionEngineBench();
        //    predictionEngineBench.SetupBreastCancerPipeline();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetup(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    CacheDataViewBench cacheDataViewBench = new CacheDataViewBench();
        //    cacheDataViewBench.Setup();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupHashScalarString(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    HashBench hashBench = new HashBench();
        //    hashBench.SetupHashScalarString();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupHashScalarFloat(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    HashBench hashBench = new HashBench();
        //    hashBench.SetupHashScalarFloat();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupHashScalarDouble(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    HashBench hashBench = new HashBench();
        //    hashBench.SetupHashScalarDouble();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupHashScalarKey(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    HashBench hashBench = new HashBench();
        //    hashBench.SetupHashScalarKey();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupHashVectorString(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    HashBench hashBench = new HashBench();
        //    hashBench.SetupHashVectorString();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupHashVectorFloat(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    HashBench hashBench = new HashBench();
        //    hashBench.SetupHashVectorFloat();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupHashVectorDouble(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    HashBench hashBench = new HashBench();
        //    hashBench.SetupHashVectorDouble();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupHashVectorKey(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    HashBench hashBench = new HashBench();
        //    hashBench.SetupHashVectorKey();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupTrainingSpeedTestsRff(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    RffTransformTrain rffTransformTrain = new RffTransformTrain();
        //    rffTransformTrain.SetupTrainingSpeedTests();
        //}

        //[Theory]
        //[IterationData(iterations: 1)]
        //[Trait("Category", "RunSpecificTest")]
        //public void TestBenchmarkHangingSetupPredictBenchmarks(int iteration)
        //{
        //    Output.WriteLine($"{iteration} -th");
        //    StochasticDualCoordinateAscentClassifierBench stochasticDualCoordinateAscentClassifierBench = 
        //        new StochasticDualCoordinateAscentClassifierBench();
        //    stochasticDualCoordinateAscentClassifierBench.SetupPredictBenchmarks();
        //}
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
        }

        public override void WriteLine()
        {
            _testOutputHelper.WriteLine(_currentLine);
            _currentLine = "";
            base.WriteLine();
        }

        public override void WriteLine(LogKind logKind, string text)
        {
            _testOutputHelper.WriteLine(_currentLine + text);
            _currentLine = "";
            base.WriteLine(logKind, text);
        }
    }
}
