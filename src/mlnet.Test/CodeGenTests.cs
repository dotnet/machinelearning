using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.CodeGenerator.CSharp;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace mlnet.Test
{
    [TestClass]
    public class CodeGeneratorTests
    {
        [TestMethod]
        public void TrainerGeneratorBasicNamedParameterTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"LearningRate", 0.1f },
                {"NumLeaves", 1 },
            };
            PipelineNode node = new PipelineNode("LightGbmBinary", PipelineNodeType.Trainer, new string[] { "Label" }, default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expected = "LightGbm(learningRate:0.1f,numLeaves:1,labelColumn:\"Label\",featureColumn:\"Features\")";
            Assert.AreEqual(expected, actual.Item1);
            Assert.IsNull(actual.Item2);
        }

        [TestMethod]
        public void TrainerGeneratorBasicAdvancedParameterTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"LearningRate", 0.1f },
                {"NumLeaves", 1 },
                {"UseSoftmax", true }
            };
            PipelineNode node = new PipelineNode("LightGbmBinary", PipelineNodeType.Trainer, new string[] { "Label" }, default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainer = "LightGbm(new Options(){LearningRate=0.1f,NumLeaves=1,UseSoftmax=true,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            string expectedUsing = "using Microsoft.ML.LightGBM;\r\n";
            Assert.AreEqual(expectedTrainer, actual.Item1);
            Assert.AreEqual(expectedUsing, actual.Item2);
        }

        [TestMethod]
        public void TransformGeneratorBasicTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("Normalizing", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expected = "Normalize(\"Label\",\"Label\")";
            Assert.AreEqual(expected, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }

        [TestMethod]
        public void TransformGeneratorUsingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("OneHotEncoding", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expectedTransform = "Categorical.OneHotEncoding(new []{new OneHotEncodingEstimator.ColumnInfo(\"Label\",\"Label\")})";
            var expectedUsings = "using Microsoft.ML.Transforms.Categorical;\r\n";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void ClassLabelGenerationBasicTest()
        {
            var columns = new TextLoader.Column[]
            {
                new TextLoader.Column(){ Name = DefaultColumnNames.Label, Source = new TextLoader.Range[]{new TextLoader.Range(0) }, Type = DataKind.Bool },
            };

            var purposes = new List<(string, ColumnPurpose)>()
            {
                (DefaultColumnNames.Label, ColumnPurpose.Label),
            };

            var result = (new TextLoader.Arguments()
            {
                Column = columns,
                AllowQuoting = false,
                AllowSparse = false,
                Separators = new[] { ',' },
                HasHeader = true,
                TrimWhitespace = true
            }, purposes);

            CodeGenerator codeGenerator = new CodeGenerator(null, result, null);
            var actual = codeGenerator.GenerateClassLabels();
            var expected1 = "[ColumnName(\"Label\"), LoadColumn(0)]";
            var expected2 = "public bool Label{get; set;}";

            Assert.AreEqual(expected1, actual[0]);
            Assert.AreEqual(expected2, actual[1]);
        }

        [TestMethod]
        public void ColumnGenerationTest()
        {
            var columns = new TextLoader.Column[]
            {
                new TextLoader.Column(){ Name = DefaultColumnNames.Label, Source = new TextLoader.Range[]{new TextLoader.Range(0) }, Type = DataKind.Bool },
                new TextLoader.Column(){ Name = DefaultColumnNames.Features, Source = new TextLoader.Range[]{new TextLoader.Range(1) }, Type = DataKind.R4 },
            };

            var purposes = new List<(string, ColumnPurpose)>()
            {
                (DefaultColumnNames.Label, ColumnPurpose.Label),
                (DefaultColumnNames.Features, ColumnPurpose.NumericFeature),
            };

            var result = (new TextLoader.Arguments()
            {
                Column = columns,
                AllowQuoting = false,
                AllowSparse = false,
                Separators = new[] { ',' },
                HasHeader = true,
                TrimWhitespace = true
            }, purposes);

            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("Normalizing", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, result, null);
            var actual = codeGenerator.GenerateColumns();
            Assert.AreEqual(actual.Count, 2);
            string expectedColumn1 = "new Column(\"Label\",DataKind.BL,0),";
            string expectedColumn2 = "new Column(\"Features\",DataKind.R4,1),";
            Assert.AreEqual(expectedColumn1, actual[0]);
            Assert.AreEqual(expectedColumn2, actual[1]);
        }

        [TestMethod]
        public void TrainerComplexParameterTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"Booster", new CustomProperty(){Properties= new Dictionary<string, object>(), Name = "TreeBooster"} },
            };
            PipelineNode node = new PipelineNode("LightGbmBinary", PipelineNodeType.Trainer, new string[] { "Label" }, default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainer = "LightGbm(new Options(){Booster=new TreeBooster(){},LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            var expectedUsings = "using Microsoft.ML.LightGBM;\r\n";
            Assert.AreEqual(expectedTrainer, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2);

        }

        #region Transform Tests
        [TestMethod]
        public void OneHotEncodingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();//categorical
            PipelineNode node = new PipelineNode("OneHotEncoding", PipelineNodeType.Transform, new string[] { "categorical_column_1" }, new string[] { "categorical_column_1" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expectedTransform = "Conversion.ConvertType(new []{new TypeConvertingTransformer.ColumnInfo(\"R4_column_1\",DataKind.R4,\"I4_column_1\")})";
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, (null, null), null);
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expectedTransform = "Conversion.MapValueToKey(\"Label\",\"Label\")";
            var expectedUsings = "using Microsoft.ML.Transforms.Conversions;\r\n";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        #endregion

    }
}
