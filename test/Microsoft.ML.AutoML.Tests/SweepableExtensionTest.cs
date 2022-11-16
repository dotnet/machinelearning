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
    public class SweepableExtensionTest : BaseTestClass
    {
        private readonly JsonSerializerOptions _jsonSerializerOptions;

        public SweepableExtensionTest(ITestOutputHelper output)
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

            _jsonSerializerOptions.Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping;

            if (Environment.GetEnvironmentVariable("HELIX_CORRELATION_ID") != null)
            {
                Approvals.UseAssemblyLocationForApprovedFiles();
            }
        }

        [Fact]
        public void CreateSweepablePipelineFromIEstimatorTest()
        {
            var context = new MLContext();
            var estimator = context.Transforms.Concatenate("output", "input");
            var pipeline = estimator.Append(SweepableEstimatorFactory.CreateFastForestBinary(new FastForestOption()));

            pipeline.Should().BeOfType<SweepablePipeline>();
        }

        [Fact]
        public void CreateSweepablePipelineFromSweepableEstimatorTest()
        {
            var estimator = SweepableEstimatorFactory.CreateFastForestBinary(new FastForestOption());
            var pipeline = estimator.Append(estimator);

            pipeline.Should().BeOfType<SweepablePipeline>();
        }

        [Fact]
        public void CreateSweepablePipelineFromSweepableEstimatorAndIEstimatorTest()
        {
            var context = new MLContext();
            var estimator = SweepableEstimatorFactory.CreateFastForestBinary(new FastForestOption());
            var pipeline = estimator.Append(context.Transforms.Concatenate("output", "input"));

            pipeline.Should().BeOfType<SweepablePipeline>();
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void CreateSweepablePipelineFromIEstimatorAndBinaryClassifiers()
        {
            var context = new MLContext();
            SweepablePipeline pipeline = context.Transforms.Concatenate("output", "input")
                                .Append(context.Auto().BinaryClassification());

            var json = JsonSerializer.Serialize(pipeline, _jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void CreateSweepablePipelineFromIEstimatorAndMultiClassifiers()
        {
            var context = new MLContext();
            SweepablePipeline pipeline = context.Transforms.Concatenate("output", "input")
                                .Append(context.Auto().MultiClassification());

            var json = JsonSerializer.Serialize(pipeline, _jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void CreateSweepablePipelineFromIEstimatorAndRegressors()
        {
            var context = new MLContext();
            SweepablePipeline pipeline = context.Transforms.Concatenate("output", "input")
                                .Append(context.Auto().MultiClassification());

            var json = JsonSerializer.Serialize(pipeline, _jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void CreateSweepablePipelineFromSweepableEstimatorAndMultiClassifiers()
        {
            var context = new MLContext();
            var pipeline = SweepableEstimatorFactory.CreateFastForestBinary(new FastForestOption())
                                .Append(context.Auto().MultiClassification());

            var json = JsonSerializer.Serialize(pipeline, _jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void CreateSweepablePipelineFromSweepablePipelineAndMultiClassifiers()
        {
            var context = new MLContext();
            SweepablePipeline pipeline = context.Transforms.Concatenate("output", "input")
                                .Append(SweepableEstimatorFactory.CreateFeaturizeText(new FeaturizeTextOption()))
                                .Append(context.Auto().MultiClassification());

            var json = JsonSerializer.Serialize(pipeline, _jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void CreateSweepablePipelineFromSweepablePipelineAndSweepableEstimatorArray()
        {
            var context = new MLContext();
            SweepablePipeline pipeline = context.Transforms.Concatenate("output", "input")
                                .Append(SweepableEstimatorFactory.CreateFeaturizeText(new FeaturizeTextOption()))
                                .Append(context.Auto().MultiClassification().Estimators.Select(kv => kv.Value).ToArray());

            var json = JsonSerializer.Serialize(pipeline, _jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void CreateSweepablePipelineFromSweepableEstimatorAndSweepableEstimatorArray()
        {
            var context = new MLContext();
            SweepablePipeline pipeline = SweepableEstimatorFactory.CreateFeaturizeText(new FeaturizeTextOption())
                                .Append(context.Auto().MultiClassification().Estimators.Select(kv => kv.Value).ToArray());

            var json = JsonSerializer.Serialize(pipeline, _jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void CreateSweepablePipelineFromIEstimatorAndSweepableEstimatorArray()
        {
            var context = new MLContext();
            SweepablePipeline pipeline = context.Transforms.Concatenate("output", "input")
                                .Append(context.Auto().MultiClassification().Estimators.Select(kv => kv.Value).ToArray());

            var json = JsonSerializer.Serialize(pipeline, _jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AppendIEstimatorToSweepabePipelineTest()
        {
            var context = new MLContext();
            var estimator = context.Transforms.Concatenate("output", "input");
            var pipeline = estimator.Append(SweepableEstimatorFactory.CreateFastForestBinary(new FastForestOption()), SweepableEstimatorFactory.CreateFastForestBinary(new FastForestOption()));
            pipeline = pipeline.Append(context.Transforms.CopyColumns("output", "input"));

            pipeline.Should().BeOfType<SweepablePipeline>();
            var json = JsonSerializer.Serialize(pipeline, _jsonSerializerOptions);
            Approvals.Verify(json);
        }
    }
}
