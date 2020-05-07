// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    
    public class TransformPostTrainerInferenceTests : BaseTestClass
    {
        public TransformPostTrainerInferenceTests(ITestOutputHelper output) : base(output)
        {
        }

        [Theory]
        [IterationData(iterations: 20)]
        [Trait("Category", "RunSpecificTest")]
        public void CompletesTransformPostTrainerMulticlassNonKeyLabel(int iterations)
        {
            Output.WriteLine($"{iterations} - th");

            int timeout = 20 * 60 * 1000;

            var runTask = Task.Run(TransformPostTrainerMulticlassNonKeyLabel);
            var timeoutTask = Task.Delay(timeout + iterations);
            var finishedTask = Task.WhenAny(timeoutTask, runTask).Result;
            if (finishedTask == timeoutTask)
            {
                Console.WriteLine("TransformPostTrainerMulticlassNonKeyLabel test Hanging: fail to complete in 20 minutes");
                Environment.FailFast("Fail here to take memory dump");
            }
        }

        [Fact]
        public void TransformPostTrainerMulticlassNonKeyLabel()
        {
            TransformPostTrainerInferenceTestCore(TaskKind.MulticlassClassification,
                new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Label", NumberDataViewType.Single, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""KeyToValueMapping"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""PredictedLabel""
    ],
    ""OutColumns"": [
      ""PredictedLabel""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformPostTrainerBinaryLabel()
        {
            TransformPostTrainerInferenceTestCore(TaskKind.BinaryClassification,
                new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Label", NumberDataViewType.Single, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                }, @"[]");
        }

        [Fact]
        public void TransformPostTrainerMulticlassKeyLabel()
        {
            TransformPostTrainerInferenceTestCore(TaskKind.MulticlassClassification,
                new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Label", new KeyDataViewType(typeof(uint), 3), ColumnPurpose.Label, new ColumnDimensions(null, null)),
                }, @"[]");
        }

        private static void TransformPostTrainerInferenceTestCore(
            TaskKind task,
            DatasetColumnInfo[] columns,
            string expectedJson)
        {
            var transforms = TransformInferenceApi.InferTransformsPostTrainer(new MLContext(1), task, columns);
            var pipelineNodes = transforms.Select(t => t.PipelineNode);
            Util.AssertObjectMatchesJson(expectedJson, pipelineNodes);
        }
    }
}
