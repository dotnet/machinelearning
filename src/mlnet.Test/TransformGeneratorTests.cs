using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.CodeGenerator.CSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace mlnet.Test
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
            var actual = codeGenerator.GenerateTransformsAndUsings();
            var expectedTransform = "ReplaceMissingValues(new []{new MissingValueReplacingTransformer.ColumnInfo(\"categorical_column_1\",\"categorical_column_1\")})";
            string expectedUsings = "using Microsoft.ML.Transforms;\r\n";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void OneHotEncodingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();//categorical
            PipelineNode node = new PipelineNode("OneHotEncoding", PipelineNodeType.Transform, new string[] { "categorical_column_1" }, new string[] { "categorical_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expectedTransform = "Categorical.OneHotEncoding(new []{new OneHotEncodingEstimator.ColumnInfo(\"categorical_column_1\",\"categorical_column_1\")})";
            var expectedUsings = "using Microsoft.ML.Transforms.Categorical;\r\n";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void NormalizingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("Normalizing", PipelineNodeType.Transform, new string[] { "numeric_column_1" }, new string[] { "numeric_column_1_copy" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expectedTransform = "Normalize(\"numeric_column_1_copy\",\"numeric_column_1\")";
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
            var actual = codeGenerator.GenerateTransformsAndUsings();
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
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expectedTransform = "CopyColumns(\"numeric_column_2\",\"numeric_column_1\")";
            string expectedUsings = null;
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void MissingValueIndicatingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();//numeric feature
            PipelineNode node = new PipelineNode("MissingValueIndicating", PipelineNodeType.Transform, new string[] { "numeric_column_1" }, new string[] { "numeric_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expectedTransform = "IndicateMissingValues(new []{(\"numeric_column_1\",\"numeric_column_1\")})";
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
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expectedTransform = "Categorical.OneHotHashEncoding(new []{new OneHotHashEncodingEstimator.ColumnInfo(\"Categorical_column_1\",\"Categorical_column_1\")})";
            var expectedUsings = "using Microsoft.ML.Transforms.Categorical;\r\n";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void TextFeaturizingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("TextFeaturizing", PipelineNodeType.Transform, new string[] { "Text_column_1" }, new string[] { "Text_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings();
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
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expectedTransform = "Conversion.ConvertType(new []{new TypeConvertingTransformer.ColumnInfo(\"R4_column_1\",DataKind.Single,\"I4_column_1\")})";
            string expectedUsings = "using Microsoft.ML.Transforms.Conversions;\r\n";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void ValueToKeyMappingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("ValueToKeyMapping", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expectedTransform = "Conversion.MapValueToKey(\"Label\",\"Label\")";
            var expectedUsings = "using Microsoft.ML.Transforms.Conversions;\r\n";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

    }
}
