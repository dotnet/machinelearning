using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace mlnet.Tests
{
    [TestClass]
    public class TransformGeneratorTests
    {
        [TestMethod]
        public void MissingValueReplacingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();//categorical
            PipelineNode node = new PipelineNode("MissingValueReplacing", PipelineNodeType.Transform, new string[] { "categorical_column_1" }, new string[] { "categorical_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            var expectedTransform = "ReplaceMissingValues(new []{new InputOutputColumnPair(\"categorical_column_1\",\"categorical_column_1\")})";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }

        [TestMethod]
        public void OneHotEncodingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();//categorical
            PipelineNode node = new PipelineNode("OneHotEncoding", PipelineNodeType.Transform, new string[] { "categorical_column_1" }, new string[] { "categorical_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Categorical.OneHotEncoding(new []{new InputOutputColumnPair(\"categorical_column_1\",\"categorical_column_1\")})";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }

        [TestMethod]
        public void NormalizingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("Normalizing", PipelineNodeType.Transform, new string[] { "numeric_column_1" }, new string[] { "numeric_column_1_copy" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "NormalizeMinMax(\"numeric_column_1_copy\",\"numeric_column_1\")";
            string expectedUsings = null;
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void ColumnConcatenatingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("ColumnConcatenating", PipelineNodeType.Transform, new string[] { "numeric_column_1", "numeric_column_2" }, new string[] { "Features" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Concatenate(\"Features\",new []{\"numeric_column_1\",\"numeric_column_2\"})";
            string expectedUsings = null;
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void ColumnCopyingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();//nume to num feature 2
            PipelineNode node = new PipelineNode("ColumnCopying", PipelineNodeType.Transform, new string[] { "numeric_column_1" }, new string[] { "numeric_column_2" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "CopyColumns(\"numeric_column_2\",\"numeric_column_1\")";
            string expectedUsings = null;
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void KeyToValueMappingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("KeyToValueMapping", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Conversion.MapKeyToValue(\"Label\",\"Label\")";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }

        [TestMethod]
        public void MissingValueIndicatingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();//numeric feature
            PipelineNode node = new PipelineNode("MissingValueIndicating", PipelineNodeType.Transform, new string[] { "numeric_column_1" }, new string[] { "numeric_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "IndicateMissingValues(new []{new InputOutputColumnPair(\"numeric_column_1\",\"numeric_column_1\")})";
            string expectedUsings = null;
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void OneHotHashEncodingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("OneHotHashEncoding", PipelineNodeType.Transform, new string[] { "Categorical_column_1" }, new string[] { "Categorical_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Categorical.OneHotHashEncoding(new []{new InputOutputColumnPair(\"Categorical_column_1\",\"Categorical_column_1\")})";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }

        [TestMethod]
        public void TextFeaturizingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("TextFeaturizing", PipelineNodeType.Transform, new string[] { "Text_column_1" }, new string[] { "Text_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Text.FeaturizeText(\"Text_column_1\",\"Text_column_1\")";
            string expectedUsings = null;
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void TypeConvertingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("TypeConverting", PipelineNodeType.Transform, new string[] { "I4_column_1" }, new string[] { "R4_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Conversion.ConvertType(new []{new InputOutputColumnPair(\"R4_column_1\",\"I4_column_1\")})";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }

        [TestMethod]
        public void ValueToKeyMappingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("ValueToKeyMapping", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Conversion.MapValueToKey(\"Label\",\"Label\")";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }

        [TestMethod]
        public void ImageLoadingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>()
            {
                {"imageFolder", @"C:\\Test" },
            };
            PipelineNode node = new PipelineNode("ImageLoading", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "LoadImages(\"Label\", @\"C:\\\\Test\", \"Label\")";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }

        [TestMethod]
        public void ImageResizingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>()
            {
                { "imageWidth", 224 },
                { "imageHeight", 224 },
            };
            PipelineNode node = new PipelineNode("ImageResizing", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "ResizeImages(\"Label\", 224, 224, \"Label\")";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }

        [TestMethod]
        public void PixelExtractingTest()
        {
            var context = new MLContext();
            PipelineNode node = new PipelineNode("PixelExtracting", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" });
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "ExtractPixels(\"Label\", \"Label\")";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }

        [TestMethod]
        public void ResNet18Featurizing()
        {
            var context = new MLContext();
            PipelineNode node = new PipelineNode("ResNet18Featurizing", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" });
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "DnnFeaturizeImage(\"Label\", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), \"Label\")";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }
    }
}
