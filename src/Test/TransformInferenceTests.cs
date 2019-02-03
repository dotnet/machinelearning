// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
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
            TransformInferenceTestCore(new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric1", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Categorical1", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("Categorical2", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("LargeCat1", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                    ("LargeCat2", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
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
            TransformInferenceTestCore(new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Numeric1", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Categorical1", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("Categorical2", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("LargeCat1", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                    ("LargeCat2", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
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
            TransformInferenceTestCore(new(string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Categorical1", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("LargeCat1", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
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
            TransformInferenceTestCore(new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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
            TransformInferenceTestCore(new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric1", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Numeric2", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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
        public void TransformInferenceFeatCol()
        {
            TransformInferenceTestCore(new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                }, @"[]");
        }

        [TestMethod]
        public void NumericAndFeatCol()
        {
            TransformInferenceTestCore(new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Numeric", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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
        public void TransformInferenceTextCol()
        {
            TransformInferenceTestCore(new(string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Text", TextType.Instance, ColumnPurpose.TextFeature, new ColumnDimensions(null, null)),
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
    ""Name"": ""ColumnConcatenating"",
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
            TransformInferenceTestCore(new(string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Text", TextType.Instance, ColumnPurpose.TextFeature, new ColumnDimensions(null, null)),
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
            TransformInferenceTestCore(new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Bool", BoolType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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
            TransformInferenceTestCore(new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Bool", BoolType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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
            TransformInferenceTestCore(new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Bool", BoolType.Instance, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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
            TransformInferenceTestCore(new(string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Missing", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    ("Numeric", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
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
            TransformInferenceTestCore(new(string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Missing1", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    ("Missing2", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    ("Numeric", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
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
            TransformInferenceTestCore(new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Numeric1", NumberType.R4, ColumnPurpose.Ignore, new ColumnDimensions(null, null)),
                    ("Numeric2", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
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
            TransformInferenceTestCore(new(string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    (DefaultColumnNames.Label, NumberType.R4, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                }, @"[]");
        }

        [TestMethod]
        public void TransformInferenceCustomLabelCol()
        {
            TransformInferenceTestCore(new(string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("CustomLabel", NumberType.R4, ColumnPurpose.Label, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""ColumnCopying"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""CustomLabel""
    ],
    ""OutColumns"": [
      ""Label""
    ],
    ""Properties"": {}
  }
]");
        }

        [TestMethod]
        public void TransformInferenceDefaultGroupIdCol()
        {
            TransformInferenceTestCore(new(string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    (DefaultColumnNames.GroupId, NumberType.R4, ColumnPurpose.Group, new ColumnDimensions(null, null)),
                }, @"[]");
        }

        [TestMethod]
        public void TransformInferenceCustomGroupIdCol()
        {
            TransformInferenceTestCore(new(string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    (DefaultColumnNames.Features, NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("CustomGroupId", NumberType.R4, ColumnPurpose.Group, new ColumnDimensions(null, null)),
                }, @"[
  {
    ""Name"": ""ColumnCopying"",
    ""NodeType"": ""Transform"",
    ""InColumns"": [
      ""CustomGroupId""
    ],
    ""OutColumns"": [
      ""GroupId""
    ],
    ""Properties"": {}
  }
]");
        }

        [TestMethod]
        public void TransformInferenceMissingNameCollision()
        {
            TransformInferenceTestCore(new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Missing", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, true)),
                    ("Missing_MissingIndicator", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
                    ("Missing_MissingIndicator0", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, false)),
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
            (string name, ColumnType type, ColumnPurpose purpose, ColumnDimensions dimensions)[] columns,
            string expectedJson)
        {
            var transforms = TransformInferenceApi.InferTransforms(new MLContext(), columns);
            TestApplyTransformsToRealDataView(transforms, columns);
            var pipelineNodes = transforms.Select(t => t.PipelineNode);
            Util.AssertObjectMatchesJson(expectedJson, pipelineNodes);
        }

        private static void TestApplyTransformsToRealDataView(IEnumerable<SuggestedTransform> transforms,
            IEnumerable<(string name, ColumnType type, ColumnPurpose purpose, ColumnDimensions dimensions)> columns)
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
            Assert.AreEqual(NumberType.R4, featuresCol.Value.Type.ItemType());
        }

        private static IDataView BuildDummyDataView(
            IEnumerable<(string name, ColumnType type, ColumnPurpose purpose, ColumnDimensions dimensions)> columns)
        {
            return BuildDummyDataView(columns.Select(c => (c.name, c.type)));
        }

        private static IDataView BuildDummyDataView(IEnumerable<(string name, ColumnType type)> columns)
        {
            var dataBuilder = new ArrayDataViewBuilder(new MLContext());
            foreach(var column in columns)
            {
                if (column.type == NumberType.R4)
                {
                    dataBuilder.AddColumn(column.name, NumberType.R4, new float[] { 0 });
                }
                else if (column.type == BoolType.Instance)
                {
                    dataBuilder.AddColumn(column.name, BoolType.Instance, new bool[] { false });
                }
                else if (column.type == TextType.Instance)
                {
                    dataBuilder.AddColumn(column.name, new string[] { "a" });
                }
            }
            return dataBuilder.GetDataView();
        }
    }
}
