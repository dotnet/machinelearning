// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class TransformInferenceTests
    {
        [TestMethod]
        public void TransformInferenceNumAndCatCols()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Categorical1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("Categorical2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("LargeCat1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                    ("LargeCat2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
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

        [TestMethod]
        public void TransformInferenceNumCatAndFeatCols()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Categorical1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("Categorical2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("LargeCat1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                    ("LargeCat2", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
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

        [TestMethod]
        public void TransformInferenceCatAndFeatCols()
        {
            TransformInferenceTestCore(new(string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Categorical1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("LargeCat1", TextDataViewType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
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

        [TestMethod]
        public void TransformInferenceNumericCol()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void TransformInferenceNumericCols()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Numeric2", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void TransformInferenceFeatColScalar()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void TransformInferenceFeatColVector()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, new VectorType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[]");
        }

        [TestMethod]
        public void NumericAndFeatCol()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void NumericScalarCol()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void NumericVectorCol()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric", new VectorType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void TransformInferenceTextCol()
        {
            TransformInferenceTestCore(new(string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Text", TextDataViewType.Instance, ColumnPurpose.TextFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void TransformInferenceTextAndFeatCol()
        {
            TransformInferenceTestCore(new(string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Text", TextDataViewType.Instance, ColumnPurpose.TextFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void TransformInferenceBoolCol()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Bool", BooleanDataViewType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void TransformInferenceBoolAndNumCols()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Bool", BooleanDataViewType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void TransformInferenceBoolAndFeatCol()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Bool", BooleanDataViewType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void TransformInferenceNumericMissingCol()
        {
            TransformInferenceTestCore(new(string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Missing", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    ("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
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

        [TestMethod]
        public void TransformInferenceNumericMissingCols()
        {
            TransformInferenceTestCore(new(string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Missing1", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    ("Missing2", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    ("Numeric", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
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

        [TestMethod]
        public void TransformInferenceIgnoreCol()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric1", NumberDataViewType.Single, ColumnPurpose.Ignore, new ColumnDimensions(null, null)),
                    ("Numeric2", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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

        [TestMethod]
        public void TransformInferenceDefaultLabelCol()
        {
            TransformInferenceTestCore(new(string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, new VectorType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    (DefaultColumnNames.Label, NumberDataViewType.Single, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                }, @"[]");
        }

        [TestMethod]
        public void TransformInferenceCustomLabelCol()
        {
            TransformInferenceTestCore(new(string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, new VectorType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("CustomLabel", NumberDataViewType.Single, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                }, @"[]");
        }

        [TestMethod]
        public void TransformInferenceCustomTextLabelCol()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, new VectorType(NumberDataViewType.Single), ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("CustomLabel", TextDataViewType.Instance, ColumnPurpose.Label, new ColumnDimensions(null, null)),
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
]");
        }

        [TestMethod]
        public void TransformInferenceMissingNameCollision()
        {
            TransformInferenceTestCore(new (string, DataViewType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Missing", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    ("Missing_MissingIndicator", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
                    ("Missing_MissingIndicator0", NumberDataViewType.Single, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
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
            (string name, DataViewType type, ColumnPurpose purpose, ColumnDimensions dimensions)[] columns,
            string expectedJson)
        {
            var transforms = TransformInferenceApi.InferTransforms(new MLContext(), columns);
            TestApplyTransformsToRealDataView(transforms, columns);
            var pipelineNodes = transforms.Select(t => t.PipelineNode);
            Util.AssertObjectMatchesJson(expectedJson, pipelineNodes);
        }

        private static void TestApplyTransformsToRealDataView(IEnumerable<SuggestedTransform> transforms,
            IEnumerable<(string name, DataViewType type, ColumnPurpose purpose, ColumnDimensions dimensions)> columns)
        {
            // create a dummy data view from input columns
            var data = BuildDummyDataView(columns);

            // iterate thru suggested transforms and apply it to a real data view
            foreach (var transform in transforms.Select(t => t.Estimator))
            {
                data = transform.Fit(data).Transform(data);
            }

            // assert Features column of type 'R4' exists
            var featuresCol = data.Schema.GetColumnOrNull(DefaultColumnNames.Features);
            Assert.IsNotNull(featuresCol);
            Assert.AreEqual(true, featuresCol.Value.Type.IsVector());
            Assert.AreEqual(NumberDataViewType.Single, featuresCol.Value.Type.GetItemType());
        }

        private static IDataView BuildDummyDataView(
            IEnumerable<(string name, DataViewType type, ColumnPurpose purpose, ColumnDimensions dimensions)> columns)
        {
            return BuildDummyDataView(columns.Select(c => (c.name, c.type)));
        }

        private static IDataView BuildDummyDataView(IEnumerable<(string name, DataViewType type)> columns)
        {
            var dataBuilder = new ArrayDataViewBuilder(new MLContext());
            foreach(var column in columns)
            {
                if (column.type == NumberDataViewType.Single)
                {
                    dataBuilder.AddColumn(column.name, NumberDataViewType.Single, new float[] { 0 });
                }
                else if (column.type == BooleanDataViewType.Instance)
                {
                    dataBuilder.AddColumn(column.name, BooleanDataViewType.Instance, new bool[] { false });
                }
                else if (column.type == TextDataViewType.Instance)
                {
                    dataBuilder.AddColumn(column.name, new string[] { "a" });
                }
                else if (column.type.IsVector() && column.type.GetItemType() == NumberDataViewType.Single)
                {
                    dataBuilder.AddColumn(column.name, Util.GetKeyValueGetter(new[] { "1", "2" }), 
                        NumberDataViewType.Single, new float[] { 0, 0 });
                }
            }
            return dataBuilder.GetDataView();
        }
    }
}
