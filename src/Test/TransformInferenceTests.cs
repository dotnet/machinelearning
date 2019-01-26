using System.Linq;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class TransformInferenceTests
    {
        [TestMethod]
        public void TransformInferenceCategoricalColumns()
        {
            var transforms = TransformInferenceApi.InferTransforms(new MLContext(),
                new (string, ColumnType, ColumnPurpose, ColumnDimensions)[]
                {
                    ("Num1", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Num2", NumberType.R4, ColumnPurpose.NumericFeature, new ColumnDimensions(null, null)),
                    ("Cat1", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("Cat2", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(7, null)),
                    ("LargeCat1", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                    ("LargeCat2", TextType.Instance, ColumnPurpose.CategoricalFeature, new ColumnDimensions(500, null)),
                });

            var actualNodes = transforms.Select(t => t.PipelineNode);

            var expectedNodesJson = @"
[
  {
    ""Name"": ""ColumnConcatenating"",
    ""NodeType"": 0,
    ""InColumns"": [
      ""Num1"",
      ""Num2""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""OneHotEncoding"",
    ""NodeType"": 0,
    ""InColumns"": [
      ""Cat1"",
      ""Cat2""
    ],
    ""OutColumns"": [
      ""Cat1"",
      ""Cat2""
    ],
    ""Properties"": {}
  },
  {
    ""Name"": ""OneHotHashEncoding"",
    ""NodeType"": 0,
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
    ""NodeType"": 0,
    ""InColumns"": [
      ""Cat1"",
      ""Cat2"",
      ""LargeCat1"",
      ""LargeCat2""
    ],
    ""OutColumns"": [
      ""Features""
    ],
    ""Properties"": {}
  }
]";
            Util.AssertObjectMatchesJson(expectedNodesJson, actualNodes);
        }
    }
}
