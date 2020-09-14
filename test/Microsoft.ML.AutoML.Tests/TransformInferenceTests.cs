// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Newtonsoft.Json;
using Xunit;
using ApprovalTests.Namers;
using ApprovalTests;
using ApprovalTests.Reporters;
using Xunit.Abstractions;
using FluentAssertions;

namespace Microsoft.ML.AutoML.Test
{
    
    public class TransformInferenceTests : BaseTestClass
    {
        public TransformInferenceTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceNumAndCatCols()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Categorical1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    new DatasetColumnInfo("Categorical2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    new DatasetColumnInfo("LargeCat1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                    new DatasetColumnInfo("LargeCat2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceNumCatAndFeatCols()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Categorical1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    new DatasetColumnInfo("Categorical2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    new DatasetColumnInfo("LargeCat1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                    new DatasetColumnInfo("LargeCat2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                });
            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceCatAndFeatCols()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Categorical1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    new DatasetColumnInfo("LargeCat1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                });
            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceNumericCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                });
            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceNumericCols()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Numeric2", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceFeatColScalar()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        public void TransformInferenceFeatColVector()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                });

            json.Should().Be("[]");
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void NumericAndFeatCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void NumericScalarCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                });
            
            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void NumericVectorCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric", new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceTextCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Text", TextDataViewType.Instance, ColumnPurpose.TextFeature, new ColumnDimensions(null, null)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceTextAndFeatCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Text", TextDataViewType.Instance, ColumnPurpose.TextFeature, new ColumnDimensions(null, null)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceBoolCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Bool", BooleanDataViewType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceBoolAndNumCols()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Bool", BooleanDataViewType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceBoolAndFeatCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Bool", BooleanDataViewType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceNumericMissingCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Missing", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceNumericMissingCols()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Missing1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    new DatasetColumnInfo("Missing2", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceIgnoreCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.Ignore, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Numeric2", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                });

            Approvals.Verify(json);
        }

        [Fact]
        public void TransformInferenceDefaultLabelCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo(DefaultColumnNames.Label, NumberDataViewType.Single, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                });

            json.Should().Be("[]");
        }

        [Fact]
        public void TransformInferenceCustomLabelCol()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("CustomLabel", NumberDataViewType.Single, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                });

            json.Should().Be("[]");
        }

        [Theory]
        [InlineData(true, @"[
  {
    ""Name"": ""ValueToKeyMapping"",
    ""NodeType"": 0,
    ""InColumns"": [
      ""CustomName""
    ],
    ""OutColumns"": [
      ""CustomName""
    ],
    ""SerializedProperties"": {}
  }
]")]
        [InlineData(false, @"[]")]
        public void TransformInferenceCustomTextForRecommendation(bool useRecommendationTask, string expectedJson)
        {
            foreach (var columnPurpose in new[] { ColumnPurpose.UserId, ColumnPurpose.ItemId })
            {
                var json = TransformInferenceTestCore(new[]
                    {
                    new DatasetColumnInfo(DefaultColumnNames.Features, new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("CustomName", TextDataViewType.Instance, columnPurpose, new ColumnDimensions(null, null)),
                }, task: useRecommendationTask ? TaskKind.Recommendation : TaskKind.MulticlassClassification);

                json.Should().Be(expectedJson);
            }
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceCustomTextLabelColMulticlass()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("CustomLabel", TextDataViewType.Instance, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                }, TaskKind.MulticlassClassification);

            Approvals.Verify(json);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TransformInferenceMissingNameCollision()
        {
            var json = TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Missing", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    new DatasetColumnInfo("Missing_MissingIndicator", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
                    new DatasetColumnInfo("Missing_MissingIndicator0", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
                });

            Approvals.Verify(json);
        }

        private static string TransformInferenceTestCore(
            DatasetColumnInfo[] columns,
            TaskKind task = TaskKind.BinaryClassification)
        {
            var transforms = TransformInferenceApi.InferTransforms(new MLContext(1), task, columns);
            TestApplyTransformsToRealDataView(transforms, columns);
            var pipelineNodes = transforms.Select(t => t.PipelineNode);
            return JsonConvert.SerializeObject(pipelineNodes, Formatting.Indented);
        }

        private static void TestApplyTransformsToRealDataView(IEnumerable<SuggestedTransform> transforms,
            IEnumerable<DatasetColumnInfo> columns)
        {
            // create a dummy data view from input columns
            var data = DataViewTestFixture.BuildDummyDataView(columns);

            // iterate through suggested transforms and apply it to a real data view
            foreach (var transform in transforms.Select(t => t.Estimator))
            {
                data = transform.Fit(data).Transform(data);
            }

            // assert Features column of type 'R4' exists
            var featuresCol = data.Schema.GetColumnOrNull(DefaultColumnNames.Features);
            Assert.NotNull(featuresCol);
            Assert.True(featuresCol.Value.Type.IsVector());
            Assert.Equal(NumberDataViewType.Single, featuresCol.Value.Type.GetItemType());
        }
    }
}
