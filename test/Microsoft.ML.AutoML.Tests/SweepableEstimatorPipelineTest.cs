// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class SweepableEstimatorPipelineTest : BaseTestClass
    {
        private readonly JsonSerializerOptions _jsonSerializerOptions;

        public SweepableEstimatorPipelineTest(ITestOutputHelper output)
            : base(output)
        {
            _jsonSerializerOptions = new JsonSerializerOptions()
            {
                WriteIndented = true,
                Converters =
                {
                    new JsonStringEnumConverter(), new DoubleToDecimalConverter(), new FloatToDecimalConverter(),
                },
            };

            if (Environment.GetEnvironmentVariable("HELIX_CORRELATION_ID") != null)
            {
                Approvals.UseAssemblyLocationForApprovedFiles();
            }
        }

        [Fact]
        public void SweepableEstimatorPipeline_append_test()
        {
            var e1 = new SweepableEstimator(CodeGen.EstimatorType.Concatenate);
            var e2 = new SweepableEstimator(CodeGen.EstimatorType.ConvertType);

            var pipeline = new SweepableEstimatorPipeline();
            pipeline = pipeline.Append(e1).Append(e2);
            pipeline.ToString().Should().Be("Concatenate=>ConvertType");
            pipeline.SearchSpace.FeatureSpaceDim.Should().Be(0);
        }

        [Fact]
        public void MultiModelPipeline_append_test()
        {
            var e1 = new SweepableEstimator(CodeGen.EstimatorType.Concatenate);
            var e2 = new SweepableEstimator(CodeGen.EstimatorType.ConvertType);
            var e3 = new SweepableEstimator(CodeGen.EstimatorType.ApplyOnnxModel);
            var e4 = new SweepableEstimator(CodeGen.EstimatorType.LightGbmBinary);

            var pipeline = new MultiModelPipeline();

            pipeline = pipeline.Append(e1, e2).AppendOrSkip(e3, e4);
            pipeline.Schema.ToString().Should().Be("(e0 + e1) * (e2 + e3 + Nil)");
            pipeline.BuildSweepableEstimatorPipeline("e0 * e2").ToString().Should().Be("Concatenate=>ApplyOnnxModel");
            pipeline.BuildSweepableEstimatorPipeline("e1 * Nil").ToString().Should().Be("ConvertType");
        }

        [Fact]
        public void MultiModelPipeline_append_pipeline_test()
        {
            var e1 = new SweepableEstimator(CodeGen.EstimatorType.Concatenate);
            var e2 = new SweepableEstimator(CodeGen.EstimatorType.ConvertType);
            var e3 = new SweepableEstimator(CodeGen.EstimatorType.ApplyOnnxModel);
            var e4 = new SweepableEstimator(CodeGen.EstimatorType.LightGbmBinary);
            var e5 = new SweepableEstimator(CodeGen.EstimatorType.FastTreeBinary);

            var pipeline1 = new MultiModelPipeline();
            var pipeline2 = new MultiModelPipeline();

            pipeline1 = pipeline1.Append(e1 + e2 * e3);
            pipeline2 = pipeline2.Append(e1 * (e3 + e4) + e5);

            pipeline1 = pipeline1.Append(pipeline2);

            pipeline1.Schema.ToString().Should().Be("(e0 + e1 * e2) * (e3 * (e4 + e5) + e6)");
        }

        [Fact]
        public void SweepableEstimatorPipeline_search_space_test()
        {
            var pipeline = CreateSweepbaleEstimatorPipeline();
            pipeline.SearchSpace.FeatureSpaceDim.Should().Be(15);

            // TODO
            // verify other properties in search space.
        }

        [Fact]
        public void SweepableEstimatorPipeline_can_be_created_from_MultiModelPipeline()
        {
            var multiModelPipeline = CreateMultiModelPipeline();
            var pipelines = multiModelPipeline.PipelineIds;

            pipelines.Should().BeEquivalentTo("e0 * e3 * e4", "e1 * e2 * e3 * e4", "e0 * Nil * e4", "e1 * e2 * Nil * e4", "Nil * e3 * e4", "e0 * e3 * e5", "e1 * e2 * e3 * e5", "e0 * Nil * e5", "e1 * e2 * Nil * e5", "Nil * e3 * e5", "Nil * Nil * e4", "Nil * Nil * e5");
            var singleModelPipeline = multiModelPipeline.BuildSweepableEstimatorPipeline(pipelines[0]);
            singleModelPipeline.ToString().Should().Be("ReplaceMissingValues=>Concatenate=>LightGbmBinary");
            singleModelPipeline = multiModelPipeline.BuildSweepableEstimatorPipeline(pipelines[2]);
            singleModelPipeline.ToString().Should().Be("ReplaceMissingValues=>LightGbmBinary");
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void SweepableEstimatorPipeline_search_space_init_value_test()
        {
            var singleModelPipeline = CreateSweepbaleEstimatorPipeline();
            var defaultParam = singleModelPipeline.SearchSpace.SampleFromFeatureSpace(singleModelPipeline.SearchSpace.Default);
            Approvals.Verify(JsonSerializer.Serialize(defaultParam, _jsonSerializerOptions));
        }

        private SweepableEstimatorPipeline CreateSweepbaleEstimatorPipeline()
        {
            var concat = SweepableEstimatorFactory.CreateConcatenate(new ConcatOption());
            var replaceMissingValue = SweepableEstimatorFactory.CreateReplaceMissingValues(new ReplaceMissingValueOption());
            var oneHot = SweepableEstimatorFactory.CreateOneHotEncoding(new OneHotOption());
            var lightGbm = SweepableEstimatorFactory.CreateLightGbmBinary(new LgbmOption());
            var fastTree = SweepableEstimatorFactory.CreateFastTreeBinary(new FastTreeOption());

            var pipeline = new SweepableEstimatorPipeline(new SweepableEstimator[] { concat, replaceMissingValue, oneHot, lightGbm, fastTree });
            return pipeline;
        }

        private MultiModelPipeline CreateMultiModelPipeline()
        {
            var concat = SweepableEstimatorFactory.CreateConcatenate(new ConcatOption());
            var replaceMissingValue = SweepableEstimatorFactory.CreateReplaceMissingValues(new ReplaceMissingValueOption());
            var oneHot = SweepableEstimatorFactory.CreateOneHotEncoding(new OneHotOption());
            var lightGbm = SweepableEstimatorFactory.CreateLightGbmBinary(new LgbmOption());
            var fastTree = SweepableEstimatorFactory.CreateFastTreeBinary(new FastTreeOption());

            var pipeline = new MultiModelPipeline();
            pipeline = pipeline.AppendOrSkip(replaceMissingValue + replaceMissingValue * oneHot);
            pipeline = pipeline.AppendOrSkip(concat);
            pipeline = pipeline.Append(lightGbm + fastTree);

            return pipeline;
        }
    }
}
