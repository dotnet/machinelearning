// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML.TestFramework;
using Newtonsoft.Json;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class SweepableEstimatorPipelineTest : BaseTestClass
    {
        public SweepableEstimatorPipelineTest(ITestOutputHelper output)
            : base(output)
        {
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
    }
}
