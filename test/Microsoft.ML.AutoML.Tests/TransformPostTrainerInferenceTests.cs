// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using ApprovalTests.Namers;
using ApprovalTests;
using ApprovalTests.Reporters;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Newtonsoft.Json;
using Xunit;
using Xunit.Abstractions;
using FluentAssertions;

namespace Microsoft.ML.AutoML.Test
{
    
    public class TransformPostTrainerInferenceTests : BaseTestClass
    {
        public TransformPostTrainerInferenceTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformPostTrainerMulticlassNonKeyLabel()
        {
            var json = TransformPostTrainerInferenceTestCore(TaskKind.MulticlassClassification,
                new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Label", NumberDataViewType.Single, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                });
            Approvals.Verify(json);
        }

        [Fact]
        public void TransformPostTrainerBinaryLabel()
        {
            var json = TransformPostTrainerInferenceTestCore(TaskKind.BinaryClassification,
                new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Label", NumberDataViewType.Single, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                });
            json.Should().Be("[]");
        }

        [Fact]
        public void TransformPostTrainerMulticlassKeyLabel()
        {
            var json = TransformPostTrainerInferenceTestCore(TaskKind.MulticlassClassification,
                new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Label", new KeyDataViewType(typeof(uint), 3), ColumnPurpose.Label, new ColumnDimensions(null, null)),
                });

            json.Should().Be("[]");
        }

        private static string TransformPostTrainerInferenceTestCore(
            TaskKind task,
            DatasetColumnInfo[] columns)
        {
            var transforms = TransformInferenceApi.InferTransformsPostTrainer(new MLContext(1), task, columns);
            var pipelineNodes = transforms.Select(t => t.PipelineNode);
            return JsonConvert.SerializeObject(pipelineNodes, Formatting.Indented);
        }
    }
}
