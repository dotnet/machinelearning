using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Microsoft.ML.CLI;
using System;

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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expected = "LightGbm(learningRate:0.1f,numLeaves:1,labelColumn:\"Label\",featureColumn:\"Features\");";
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainer = "LightGbm(new Options(){LearningRate=0.1f,NumLeaves=1,UseSoftmax=true,LabelColumn=\"Label\",FeatureColumn=\"Features\"});";
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null);
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null);
            var actual = codeGenerator.GenerateTransformsAndUsings();
            string expectedTransform = "Categorical.OneHotEncoding(new []{new OneHotEncodingEstimator.ColumnInfo(\"Label\",\"Label\")})";
            var expectedUsings = "using Microsoft.ML.Transforms.Categorical;\r\n";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.AreEqual(expectedUsings, actual[0].Item2);
        }

        [TestMethod]
        public void ClassLabelGenerationBasicTest()
        {
            List<(TextLoader.Column, ColumnPurpose)> list = new List<(TextLoader.Column, ColumnPurpose)>()
            {
                (new TextLoader.Column(){ Name = "Label", Source = new TextLoader.Range[]{new TextLoader.Range(0) }, Type = DataKind.Bool }, ColumnPurpose.Label),
            };
            ColumnInferenceResult result = new ColumnInferenceResult(list, false, false, ",", true, true);

            CodeGenerator codeGenerator = new CodeGenerator(null, result);
            var actual = codeGenerator.GenerateClassLabels();
            var expected1 = "[ColumnName(\"Label\")]";
            var expected2 = "public bool Label{get; set;}";

            Assert.AreEqual(expected1, actual[0]);
            Assert.AreEqual(expected2, actual[1]);
        }

        [TestMethod]
        public void ColumnGenerationTest()
        {
            List<(TextLoader.Column, ColumnPurpose)> list = new List<(TextLoader.Column, ColumnPurpose)>()
            {
                (new TextLoader.Column(){ Name = "Label", Source = new TextLoader.Range[]{new TextLoader.Range(0) }, Type = DataKind.Bool }, ColumnPurpose.Label),
                (new TextLoader.Column(){ Name = "Features", Source = new TextLoader.Range[]{new TextLoader.Range(1) }, Type = DataKind.R4 }, ColumnPurpose.NumericFeature),
            };
            ColumnInferenceResult result = new ColumnInferenceResult(list, false, false, ",", true, true);

            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("Normalizing", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, result);
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainer = "LightGbm(new Options(){Booster=new TreeBooster(){},LabelColumn=\"Label\",FeatureColumn=\"Features\"});";
            var expectedUsings = "using Microsoft.ML.LightGBM;\r\n";
            Assert.AreEqual(expectedTrainer, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2);

        }

    }
}
