// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{

    public class TransformInferenceTests : BaseTestClass
    {
        public TransformInferenceTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TransformInferenceNumAndCatCols()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Categorical1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    new DatasetColumnInfo("Categorical2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    new DatasetColumnInfo("LargeCat1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                    new DatasetColumnInfo("LargeCat2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                }, @"[
  {
    ""Name"": ""OneHotEncoding"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Categorical1"",
      ""Categorical2""
    ],
    ""OutColumns"": [
      ""Categorical1"",
      ""Categorical2""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""OneHotHashEncoding"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""LargeCat1"",
      ""LargeCat2""
    ],
    ""OutColumns"": [
      ""LargeCat1"",
      ""LargeCat2""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Categorical1"",
      ""Categorical2"",
      ""LargeCat1"",
      ""LargeCat2"",
      ""Numeric1""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceNumCatAndFeatCols()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Categorical1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    new DatasetColumnInfo("Categorical2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    new DatasetColumnInfo("LargeCat1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                    new DatasetColumnInfo("LargeCat2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                }, @"[
  {
    ""Name"": ""OneHotEncoding"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Categorical1"",
      ""Categorical2""
    ],
    ""OutColumns"": [
      ""Categorical1"",
      ""Categorical2""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""OneHotHashEncoding"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""LargeCat1"",
      ""LargeCat2""
    ],
    ""OutColumns"": [
      ""LargeCat1"",
      ""LargeCat2""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Categorical1"",
      ""Categorical2"",
      ""LargeCat1"",
      ""LargeCat2"",
      ""Features"",
      ""Numeric1""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceCatAndFeatCols()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Categorical1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    new DatasetColumnInfo("LargeCat1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                }, @"[
  {
    ""Name"": ""OneHotEncoding"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Categorical1""
    ],
    ""OutColumns"": [
      ""Categorical1""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""OneHotHashEncoding"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""LargeCat1""
    ],
    ""OutColumns"": [
      ""LargeCat1""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Categorical1"",
      ""LargeCat1"",
      ""Features""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceNumericCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                },
                @"[
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Numeric""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceNumericCols()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Numeric2", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Numeric1"",
      ""Numeric2""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceFeatColScalar()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Features""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceFeatColVector()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[]");
        }

        [Fact]
        public void NumericAndFeatCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Features"",
      ""Numeric""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void NumericScalarCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Numeric""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void NumericVectorCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric", new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""ColumnCopying"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Numeric""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceTextCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Text", TextDataViewType.Instance, ColumnPurpose.TextFeature, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""TextFeaturizing"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Text""
    ],
    ""OutColumns"": [
      ""Text_tf""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""ColumnCopying"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Text_tf""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceTextAndFeatCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Text", TextDataViewType.Instance, ColumnPurpose.TextFeature, new ColumnDimensions(null, null)),
                },
                @"[
  {
    ""Name"": ""TextFeaturizing"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Text""
    ],
    ""OutColumns"": [
      ""Text_tf""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Text_tf"",
      ""Features""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceBoolCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Bool", BooleanDataViewType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""TypeConverting"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Bool""
    ],
    ""OutColumns"": [
      ""Bool""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Bool""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceBoolAndNumCols()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Bool", BooleanDataViewType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""TypeConverting"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Bool""
    ],
    ""OutColumns"": [
      ""Bool""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Bool"",
      ""Numeric""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceBoolAndFeatCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Bool", BooleanDataViewType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""TypeConverting"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Bool""
    ],
    ""OutColumns"": [
      ""Bool""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Bool"",
      ""Features""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceNumericMissingCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Missing", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
                }, @"[
  {
    ""Name"": ""MissingValueIndicating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing""
    ],
    ""OutColumns"": [
      ""Missing_MissingIndicator""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""TypeConverting"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing_MissingIndicator""
    ],
    ""OutColumns"": [
      ""Missing_MissingIndicator""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""MissingValueReplacing"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing""
    ],
    ""OutColumns"": [
      ""Missing""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing_MissingIndicator"",
      ""Missing"",
      ""Numeric""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceNumericMissingCols()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Missing1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    new DatasetColumnInfo("Missing2", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    new DatasetColumnInfo("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
                }, @"[
  {
    ""Name"": ""MissingValueIndicating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing1"",
      ""Missing2""
    ],
    ""OutColumns"": [
      ""Missing1_MissingIndicator"",
      ""Missing2_MissingIndicator""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""TypeConverting"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing1_MissingIndicator"",
      ""Missing2_MissingIndicator""
    ],
    ""OutColumns"": [
      ""Missing1_MissingIndicator"",
      ""Missing2_MissingIndicator""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""MissingValueReplacing"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing1"",
      ""Missing2""
    ],
    ""OutColumns"": [
      ""Missing1"",
      ""Missing2""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing1_MissingIndicator"",
      ""Missing2_MissingIndicator"",
      ""Missing1"",
      ""Missing2"",
      ""Numeric""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceIgnoreCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Numeric1", NumberDataViewType.Single, ColumnPurpose.Ignore, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("Numeric2", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Numeric2""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        [Fact]
        public void TransformInferenceDefaultLabelCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo(DefaultColumnNames.Label, NumberDataViewType.Single, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                }, @"[]");
        }

        [Fact]
        public void TransformInferenceCustomLabelCol()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("CustomLabel", NumberDataViewType.Single, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                }, @"[]");
        }

        [Theory]
        [InlineData(true, @"[
  {
    ""Name"": ""ValueToKeyMapping"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""CustomName""
    ],
    ""OutColumns"": [
      ""CustomName""
    ],
    ""Properties"": {}
  }
]")]
        [InlineData(false, @"[]")]
        public void TransformInferenceCustomTextForRecommendation(bool useRecommendationTask, string expectedJson)
        {
            foreach (var columnPurpose in new[] { ColumnPurpose.UserId, ColumnPurpose.ItemId })
            {
                TransformInferenceTestCore(new[]
                    {
                    new DatasetColumnInfo(DefaultColumnNames.Features, new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("CustomName", TextDataViewType.Instance, columnPurpose, new ColumnDimensions(null, null)),
                }, expectedJson, useRecommendationTask ? TaskKind.Recommendation : TaskKind.MulticlassClassification);
            }
        }

        [Fact]
        public void TransformInferenceCustomTextLabelColMulticlass()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo(DefaultColumnNames.Features, new VectorDataViewType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    new DatasetColumnInfo("CustomLabel", TextDataViewType.Instance, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""ValueToKeyMapping"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""CustomLabel""
    ],
    ""OutColumns"": [
      ""CustomLabel""
    ],
    ""Properties"": {}
  }
]", TaskKind.MulticlassClassification);
        }

        [Fact]
        public void TransformInferenceMissingNameCollision()
        {
            TransformInferenceTestCore(new[]
                {
                    new DatasetColumnInfo("Missing", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    new DatasetColumnInfo("Missing_MissingIndicator", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
                    new DatasetColumnInfo("Missing_MissingIndicator0", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
                }, @"[
  {
    ""Name"": ""MissingValueIndicating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing""
    ],
    ""OutColumns"": [
      ""Missing_MissingIndicator1""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""TypeConverting"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing_MissingIndicator1""
    ],
    ""OutColumns"": [
      ""Missing_MissingIndicator1""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""MissingValueReplacing"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing""
    ],
    ""OutColumns"": [
      ""Missing""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""Missing_MissingIndicator1"",
      ""Missing"",
      ""Missing_MissingIndicator"",
      ""Missing_MissingIndicator0""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]");
        }

        private static void TransformInferenceTestCore(
            DatasetColumnInfo[] columns,
            string expectedJson,
            TaskKind task = TaskKind.BinaryClassification)
        {
            var transforms = TransformInferenceApi.InferTransforms(new MLContext(1), task, columns);
            TestApplyTransformsToRealDataView(transforms, columns);
            var pipelineNodes = transforms.Select(t => t.PipelineNode);
            Util.AssertObjectMatchesJson(expectedJson, pipelineNodes);
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
