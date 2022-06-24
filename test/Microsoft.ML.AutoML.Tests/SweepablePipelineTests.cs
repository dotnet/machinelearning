// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Microsoft.ML.AutoML.Test
{
    public class SweepablePipelineTests : BaseTestClass
    {
        private readonly JsonSerializerOptions _jsonSerializerOptions;

        public SweepablePipelineTests(ITestOutputHelper output) : base(output)
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
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void SweepablePipeline_Append_SweepableEstimator_Test()
        {
            var pipeline = new SweepablePipeline();
            var concatOption = new ConcatOption()
            {
                InputColumnNames = new List<string> { "a", "b", "c" }.ToArray(),
                OutputColumnName = "a",
            };
            var lgbmOption = new LgbmOption()
            {
                FeatureColumnName = "Feature",
                LabelColumnName = "Label",
            };

            // pipeline can append a single sweepable estimator
            pipeline = pipeline.Append(SweepableEstimatorFactory.CreateConcatenate(concatOption));

            // pipeline can append muliple sweepable estimators.
            pipeline = pipeline.Append(SweepableEstimatorFactory.CreateLightGbmBinary(lgbmOption), SweepableEstimatorFactory.CreateConcatenate(concatOption));

            // pipeline can append sweepable pipelines mixed with sweepble estimators
            pipeline = pipeline.Append(SweepableEstimatorFactory.CreateConcatenate(concatOption), pipeline);

            // pipeline can append sweepable pipelines.
            pipeline = pipeline.Append(pipeline, pipeline);

            Approvals.Verify(JsonSerializer.Serialize(pipeline, _jsonSerializerOptions));
        }
    }
}
