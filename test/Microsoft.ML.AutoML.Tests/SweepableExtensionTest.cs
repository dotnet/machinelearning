// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using Microsoft.ML.AutoML.CodeGen;
using FluentAssertions;
using Microsoft.ML.TestFramework;
using Xunit.Abstractions;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using System.Text.Json.Serialization;
using System.Text.Json;
using ApprovalTests;

namespace Microsoft.ML.AutoML.Test
{
    public class SweepableExtensionTest : BaseTestClass
    {
        private readonly JsonSerializerOptions _jsonSerializerOptions;

        public SweepableExtensionTest(ITestOutputHelper output)
            : base(output)
        {
            this._jsonSerializerOptions = new JsonSerializerOptions()
            {
                WriteIndented = true,
                Converters =
                {
                    new JsonStringEnumConverter(), new DoubleToDecimalConverter(), new FloatToDecimalConverter(),
                },
            };

            this._jsonSerializerOptions.Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping;

            if (Environment.GetEnvironmentVariable("HELIX_CORRELATION_ID") != null)
            {
                Approvals.UseAssemblyLocationForApprovedFiles();
            }
        }

        [Fact]
        public void CreateSweepableEstimatorPipelineFromIEstimatorTest()
        {
            var context = new MLContext();
            var estimator = context.Transforms.Concatenate("output", "input");
            var pipeline = estimator.Append(SweepableEstimatorFactory.CreateFastForestBinary(new FastForestOption()));

            pipeline.ToString().Should().Be("Unknown=>FastForestBinary");
        }

        [Fact]
        public void AppendIEstimatorToSweepabeEstimatorPipelineTest()
        {
            var context = new MLContext();
            var estimator = context.Transforms.Concatenate("output", "input");
            var pipeline = estimator.Append(SweepableEstimatorFactory.CreateFastForestBinary(new FastForestOption()));
            pipeline = pipeline.Append(context.Transforms.CopyColumns("output", "input"));

            pipeline.ToString().Should().Be("Unknown=>FastForestBinary=>Unknown");
        }

        [Fact]
        public void CreateSweepableEstimatorPipelineFromSweepableEstimatorTest()
        {
            var estimator = SweepableEstimatorFactory.CreateFastForestBinary(new FastForestOption());
            var pipeline = estimator.Append(estimator);

            pipeline.ToString().Should().Be("FastForestBinary=>FastForestBinary");
        }

        [Fact]
        public void CreateSweepableEstimatorPipelineFromSweepableEstimatorAndIEstimatorTest()
        {
            var context = new MLContext();
            var estimator = SweepableEstimatorFactory.CreateFastForestBinary(new FastForestOption());
            var pipeline = estimator.Append(context.Transforms.Concatenate("output", "input"));

            pipeline.ToString().Should().Be("FastForestBinary=>Unknown");

        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void CreateMultiModelPipelineFromIEstimatorAndBinaryClassifiers()
        {
            var context = new MLContext();
            var pipeline = context.Transforms.Concatenate("output", "input")
                                .Append(context.Auto().BinaryClassification());

            var json = JsonSerializer.Serialize(pipeline, this._jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void CreateMultiModelPipelineFromIEstimatorAndMultiClassifiers()
        {
            var context = new MLContext();
            var pipeline = context.Transforms.Concatenate("output", "input")
                                .Append(context.Auto().MultiClassification());

            var json = JsonSerializer.Serialize(pipeline, this._jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void CreateMultiModelPipelineFromIEstimatorAndRegressors()
        {
            var context = new MLContext();
            var pipeline = context.Transforms.Concatenate("output", "input")
                                .Append(context.Auto().MultiClassification());

            var json = JsonSerializer.Serialize(pipeline, this._jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void CreateMultiModelPipelineFromSweepableEstimatorAndMultiClassifiers()
        {
            var context = new MLContext();
            var pipeline = SweepableEstimatorFactory.CreateFastForestBinary(new FastForestOption())
                                .Append(context.Auto().MultiClassification());

            var json = JsonSerializer.Serialize(pipeline, this._jsonSerializerOptions);
            Approvals.Verify(json);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void CreateMultiModelPipelineFromSweepableEstimatorPipelineAndMultiClassifiers()
        {
            var context = new MLContext();
            var pipeline = context.Transforms.Concatenate("output", "input")
                                .Append(SweepableEstimatorFactory.CreateFeaturizeText(new FeaturizeTextOption()))
                                .Append(context.Auto().MultiClassification());

            var json = JsonSerializer.Serialize(pipeline, this._jsonSerializerOptions);
            Approvals.Verify(json);
        }
    }
}
