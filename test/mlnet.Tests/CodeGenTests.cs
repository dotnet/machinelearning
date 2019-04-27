// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.CLI.CodeGenerator.CSharp;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace mlnet.Tests
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
            PipelineNode node = new PipelineNode("LightGbmBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expected = "LightGbm(new LightGbmBinaryTrainer.Options(){LearningRate=0.1f,NumLeaves=1,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.AreEqual(expected, actual.Item1);
            Assert.AreEqual(1, actual.Item2.Count());
            Assert.AreEqual("using Microsoft.ML.Trainers.LightGbm;\r\n", actual.Item2.First());
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
            PipelineNode node = new PipelineNode("LightGbmBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainer = "LightGbm(new LightGbmBinaryTrainer.Options(){LearningRate=0.1f,NumLeaves=1,UseSoftmax=true,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            string expectedUsing = "using Microsoft.ML.Trainers.LightGbm;\r\n";
            Assert.AreEqual(expectedTrainer, actual.Item1);
            Assert.AreEqual(expectedUsing, actual.Item2[0]);
        }

        [TestMethod]
        public void TransformGeneratorBasicTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("Normalizing", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new List<PipelineNode>() { node });
            string expected = "NormalizeMinMax(\"Label\",\"Label\")";
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new List<PipelineNode>() { node });
            string expectedTransform = "Categorical.OneHotEncoding(new []{new InputOutputColumnPair(\"Label\",\"Label\")})";
            Assert.AreEqual(expectedTransform, actual[0].Item1);
            Assert.IsNull(actual[0].Item2);
        }

        [TestMethod]
        public void ClassLabelGenerationBasicTest()
        {
            var columns = new TextLoader.Column[]
            {
                new TextLoader.Column(){ Name = DefaultColumnNames.Label, Source = new TextLoader.Range[]{new TextLoader.Range(0) }, DataKind = DataKind.Boolean },
            };

            var result = new ColumnInferenceResults()
            {
                TextLoaderOptions = new TextLoader.Options()
                {
                    Columns = columns,
                    AllowQuoting = false,
                    AllowSparse = false,
                    Separators = new[] { ',' },
                    HasHeader = true,
                    TrimWhitespace = true
                },
                ColumnInformation = new ColumnInformation()
            };

            CodeGenerator codeGenerator = new CodeGenerator(null, result, null);
            var actual = codeGenerator.GenerateClassLabels();
            var expected1 = "[ColumnName(\"Label\"), LoadColumn(0)]";
            var expected2 = "public bool Label{get; set;}";

            Assert.AreEqual(expected1, actual[0]);
            Assert.AreEqual(expected2, actual[1]);
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
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainer = "LightGbm(new LightGbmBinaryTrainer.Options(){Booster=new TreeBooster(){},LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            var expectedUsings = "using Microsoft.ML.Trainers.LightGbm;\r\n";
            Assert.AreEqual(expectedTrainer, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);
        }
    }
}
