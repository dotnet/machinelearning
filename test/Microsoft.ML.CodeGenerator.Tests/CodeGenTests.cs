// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.Data;
using Xunit;

namespace mlnet.Tests
{
    public class CodeGeneratorTests
    {
        [Fact]
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
            Assert.Equal(expected, actual.Item1);
            Assert.Single(actual.Item2);
            Assert.Equal("using Microsoft.ML.Trainers.LightGbm;\r\n", actual.Item2.First());
        }

        [Fact]
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
            Assert.Equal(expectedTrainer, actual.Item1);
            Assert.Equal(expectedUsing, actual.Item2[0]);
        }

        [Fact]
        public void TransformGeneratorBasicTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("Normalizing", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new List<PipelineNode>() { node });
            string expected = "NormalizeMinMax(\"Label\",\"Label\")";
            Assert.Equal(expected, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void TransformGeneratorUsingTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("OneHotEncoding", PipelineNodeType.Transform, new string[] { "Label" }, new string[] { "Label" }, elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTransformsAndUsings(new List<PipelineNode>() { node });
            string expectedTransform = "Categorical.OneHotEncoding(new []{new InputOutputColumnPair(\"Label\",\"Label\")})";
            Assert.Equal(expectedTransform, actual[0].Item1);
            Assert.Null(actual[0].Item2);
        }

        [Fact]
        public void ClassLabelGenerationBasicTest()
        {
            var columns = new TextLoader.Column[]
            {
                new TextLoader.Column(){ Name = "Label", Source = new TextLoader.Range[]{new TextLoader.Range(0) }, DataKind = DataKind.Boolean },
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

            Assert.Equal(expected1, actual[0]);
            Assert.Equal(expected2, actual[1]);
        }

        [Fact]
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
            Assert.Equal(expectedTrainer, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);
        }

        [Fact]
        public void NormalizeTest()
        {
            var testStrArray = new string[] { "Abc Abc", "abc ABC" };
            var expectedStrArray = new string[] { "Abc_Abc", "Abc_ABC" };
            for (int i = 0; i != expectedStrArray.Count(); ++i)
            {
                var actualStr = Microsoft.ML.CodeGenerator.Utilities.Utils.Normalize(testStrArray[i]);
                Assert.Equal(expectedStrArray[i], actualStr);
            }
        }
    }
}
