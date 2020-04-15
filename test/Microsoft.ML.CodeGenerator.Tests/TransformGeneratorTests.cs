// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace mlnet.Tests
{
    public class TransformGeneratorTests : BaseTestClass
    {
        public TransformGeneratorTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void MissingValueReplacingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();//categorical
            PipelineNode node = new PipelineNode("MissingValueReplacing", PipelineNodeType.Transform, new string[] { "categorical_column_1" }, new string[] { "categorical_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            var expectedTransform = "ReplaceMissingValues(new []{new InputOutputColumnPair(\"categorical_column_1\",\"categorical_column_1\")})";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void OneHotEncodingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();//categorical
            PipelineNode node = new PipelineNode("OneHotEncoding", PipelineNodeType.Transform, new string[] { "categorical_column_1" }, new string[] { "categorical_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Categorical.OneHotEncoding(new []{new InputOutputColumnPair(\"categorical_column_1\",\"categorical_column_1\")})";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void NormalizingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("Normalizing", PipelineNodeType.Transform, new string[] { "numeric_column_1" }, new string[] { "numeric_column_1_copy" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "NormalizeMinMax(\"numeric_column_1_copy\",\"numeric_column_1\")";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void ColumnConcatenatingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("ColumnConcatenating", PipelineNodeType.Transform, new string[] { "numeric_column_1", "numeric_column_2" }, new string[] { "Features" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Concatenate(\"Features\",new []{\"numeric_column_1\",\"numeric_column_2\"})";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void ColumnCopyingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();//nume to num feature 2
            PipelineNode node = new PipelineNode("ColumnCopying", PipelineNodeType.Transform, new string[] { "numeric_column_1" }, new string[] { "numeric_column_2" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "CopyColumns(\"numeric_column_2\",\"numeric_column_1\")";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void KeyToValueMappingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("KeyToValueMapping", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Conversion.MapKeyToValue(\"Label\",\"Label\")";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void MissingValueIndicatingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();//numeric feature
            PipelineNode node = new PipelineNode("MissingValueIndicating", PipelineNodeType.Transform, new string[] { "numeric_column_1" }, new string[] { "numeric_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "IndicateMissingValues(new []{new InputOutputColumnPair(\"numeric_column_1\",\"numeric_column_1\")})";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void OneHotHashEncodingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("OneHotHashEncoding", PipelineNodeType.Transform, new string[] { "Categorical_column_1" }, new string[] { "Categorical_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Categorical.OneHotHashEncoding(new []{new InputOutputColumnPair(\"Categorical_column_1\",\"Categorical_column_1\")})";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void TextFeaturizingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("TextFeaturizing", PipelineNodeType.Transform, new string[] { "Text_column_1" }, new string[] { "Text_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Text.FeaturizeText(\"Text_column_1\",\"Text_column_1\")";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void TypeConvertingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("TypeConverting", PipelineNodeType.Transform, new string[] { "I4_column_1" }, new string[] { "R4_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Conversion.ConvertType(new []{new InputOutputColumnPair(\"R4_column_1\",\"I4_column_1\")})";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void ValueToKeyMappingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("ValueToKeyMapping", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "Conversion.MapValueToKey(\"Label\",\"Label\")";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void ImageLoadingTest()
        {
            PipelineNode node = new PipelineNode("ImageLoading", PipelineNodeType.Transform,
                new string[] { "Label" }, new string[] { "Label" }, null);

            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "LoadImages(\"Label\", null, \"Label\")";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void ImageLoadingRawBytes()
        {
            PipelineNode node = new PipelineNode("RawByteImageLoading", PipelineNodeType.Transform,
                new string[] { "Label" }, new string[] { "Label" }, null);

            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new PipelineNode[] { node });
            string expectedTransform = "LoadRawImageBytes(\"Label\", null, \"Label\")";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }
    }
}
