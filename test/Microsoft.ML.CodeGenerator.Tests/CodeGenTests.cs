// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace mlnet.Tests
{
    public class CodeGeneratorTests : BaseTestClass
    {
        public CodeGeneratorTests(ITestOutputHelper output) : base(output)
        {
        }

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
        public void ClassLabelGenerationTest()
        {
            Assert.Equal(CodeGenTestData.inputColumns.Count, CodeGenTestData.expectedLabels.Count);
            for (int i = 0; i < CodeGenTestData.inputColumns.Count; i++)
            {
                var result = new ColumnInferenceResults()
                {
                    TextLoaderOptions = new TextLoader.Options()
                    {
                        Columns = CodeGenTestData.inputColumns[i],
                        AllowQuoting = false,
                        AllowSparse = false,
                        Separators = new[] { ',' },
                        HasHeader = true,
                        TrimWhitespace = true
                    },
                    ColumnInformation = new ColumnInformation()
                };

                CodeGenerator codeGenerator = new CodeGenerator(null, result, null);
                var actualLabels = codeGenerator.GenerateClassLabels();
                Assert.Equal(actualLabels, CodeGenTestData.expectedLabels[i]);
            }
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


    }
    public class CodeGenTestData
    {
        public static List<TextLoader.Column[]> inputColumns = new List<TextLoader.Column[]>
        {
            new TextLoader.Column[]
            {
                new TextLoader.Column(){ Name = "Label", Source = new TextLoader.Range[]{new TextLoader.Range(0) }, DataKind = DataKind.Boolean },
            },
            new TextLoader.Column[]
            {
                new TextLoader.Column(){ Name = "id", Source = new TextLoader.Range[]{new TextLoader.Range(0) }, DataKind = DataKind.Single },
                new TextLoader.Column(){ Name = "country", Source = new TextLoader.Range[]{new TextLoader.Range(1) }, DataKind = DataKind.Single },
                new TextLoader.Column(){ Name = "Country", Source = new TextLoader.Range[]{new TextLoader.Range(2) }, DataKind = DataKind.String }
            },
            new TextLoader.Column[]
            {
                new TextLoader.Column(){ Name = "id", Source = new TextLoader.Range[]{new TextLoader.Range(0) }, DataKind = DataKind.Int32 },
                new TextLoader.Column(){ Name = "shape", Source = new TextLoader.Range[]{new TextLoader.Range(1) }, DataKind = DataKind.Int32 },
                new TextLoader.Column(){ Name = "Shape", Source = new TextLoader.Range[]{new TextLoader.Range(2) }, DataKind = DataKind.String },
                new TextLoader.Column(){ Name = "color", Source = new TextLoader.Range[]{new TextLoader.Range(3) }, DataKind = DataKind.String },
                new TextLoader.Column(){ Name = "price", Source = new TextLoader.Range[]{new TextLoader.Range(4) }, DataKind = DataKind.Double },
            },
            new TextLoader.Column[]
            {
                new TextLoader.Column(){ Name = "vin", Source = new TextLoader.Range[]{new TextLoader.Range(0) }, DataKind = DataKind.String },
                new TextLoader.Column(){ Name = "make", Source = new TextLoader.Range[]{new TextLoader.Range(1) }, DataKind = DataKind.String },
                new TextLoader.Column(){ Name = "model", Source = new TextLoader.Range[]{new TextLoader.Range(2) }, DataKind = DataKind.String },
                new TextLoader.Column(){ Name = "color", Source = new TextLoader.Range[]{new TextLoader.Range(3) }, DataKind = DataKind.String },
                new TextLoader.Column(){ Name = "MSRP", Source = new TextLoader.Range[]{new TextLoader.Range(4) }, DataKind = DataKind.Single },
                new TextLoader.Column(){ Name = "engine size", Source = new TextLoader.Range[]{new TextLoader.Range(5) }, DataKind = DataKind.Double },
                new TextLoader.Column(){ Name = "isElectric", Source = new TextLoader.Range[]{new TextLoader.Range(6) }, DataKind = DataKind.Boolean },
            },
            new TextLoader.Column[]
            {
                new TextLoader.Column(){ Name = "var_text", Source = new TextLoader.Range[]{new TextLoader.Range(0) }, DataKind = DataKind.String },
                new TextLoader.Column(){ Name = "var_text", Source = new TextLoader.Range[]{new TextLoader.Range(1) }, DataKind = DataKind.String },
                new TextLoader.Column(){ Name = "var_num", Source = new TextLoader.Range[]{new TextLoader.Range(2) }, DataKind = DataKind.Int32 },
                new TextLoader.Column(){ Name = "var_num", Source = new TextLoader.Range[]{new TextLoader.Range(3) }, DataKind = DataKind.Int32 },
                new TextLoader.Column(){ Name = "var_num", Source = new TextLoader.Range[]{new TextLoader.Range(4) }, DataKind = DataKind.Int32 },
                new TextLoader.Column(){ Name = "var_text", Source = new TextLoader.Range[]{new TextLoader.Range(5) }, DataKind = DataKind.String },
                new TextLoader.Column(){ Name = "var_num", Source = new TextLoader.Range[]{new TextLoader.Range(6) }, DataKind = DataKind.Int32 },
                new TextLoader.Column(){ Name = "var_text", Source = new TextLoader.Range[]{new TextLoader.Range(7) }, DataKind = DataKind.String },
                new TextLoader.Column(){ Name = "var_num", Source = new TextLoader.Range[]{new TextLoader.Range(8) }, DataKind = DataKind.Double },
            }
        };
        public static List<List<string>> expectedLabels = new List<List<string>>
        {
            new List<string>
            {
                "[ColumnName(\"Label\"), LoadColumn(0)]",
                "public bool Label{get; set;}",
                "\r\n"
            },
            new List<string>
            {
                "[ColumnName(\"id\"), LoadColumn(0)]",
                "public float Id_col_0{get; set;}",
                "\r\n",
                "[ColumnName(\"country\"), LoadColumn(1)]",
                "public float Country_col_1{get; set;}",
                "\r\n",
                "[ColumnName(\"Country\"), LoadColumn(2)]",
                "public string Country_col_2{get; set;}",
                "\r\n"
            },
            new List<string>
            {
                "[ColumnName(\"id\"), LoadColumn(0)]",
                "public int Id_col_0{get; set;}",
                "\r\n",
                "[ColumnName(\"shape\"), LoadColumn(1)]",
                "public int Shape_col_1{get; set;}",
                "\r\n",
                "[ColumnName(\"Shape\"), LoadColumn(2)]",
                "public string Shape_col_2{get; set;}",
                "\r\n",
                "[ColumnName(\"color\"), LoadColumn(3)]",
                "public string Color_col_3{get; set;}",
                "\r\n",
                "[ColumnName(\"price\"), LoadColumn(4)]",
                "public double Price_col_4{get; set;}",
                "\r\n"
            },
            new List<string>
            {
                "[ColumnName(\"vin\"), LoadColumn(0)]",
                "public string Vin{get; set;}",
                "\r\n",
                "[ColumnName(\"make\"), LoadColumn(1)]",
                "public string Make{get; set;}",
                "\r\n",
                "[ColumnName(\"model\"), LoadColumn(2)]",
                "public string Model{get; set;}",
                "\r\n",
                "[ColumnName(\"color\"), LoadColumn(3)]",
                "public string Color{get; set;}",
                "\r\n",
                "[ColumnName(\"MSRP\"), LoadColumn(4)]",
                "public float MSRP{get; set;}",
                "\r\n",
                "[ColumnName(\"engine size\"), LoadColumn(5)]",
                "public double Engine_size{get; set;}",
                "\r\n",
                "[ColumnName(\"isElectric\"), LoadColumn(6)]",
                "public bool IsElectric{get; set;}",
                "\r\n"
            },
            new List<string>
            {
                "[ColumnName(\"var_text\"), LoadColumn(0)]",
                "public string Var_text_col_0{get; set;}",
                "\r\n",
                "[ColumnName(\"var_text\"), LoadColumn(1)]",
                "public string Var_text_col_1{get; set;}",
                "\r\n",
                "[ColumnName(\"var_num\"), LoadColumn(2)]",
                "public int Var_num_col_2{get; set;}",
                "\r\n",
                "[ColumnName(\"var_num\"), LoadColumn(3)]",
                "public int Var_num_col_3{get; set;}",
                "\r\n",
                "[ColumnName(\"var_num\"), LoadColumn(4)]",
                "public int Var_num_col_4{get; set;}",
                "\r\n",
                "[ColumnName(\"var_text\"), LoadColumn(5)]",
                "public string Var_text_col_5{get; set;}",
                "\r\n",
                "[ColumnName(\"var_num\"), LoadColumn(6)]",
                "public int Var_num_col_6{get; set;}",
                "\r\n",
                "[ColumnName(\"var_text\"), LoadColumn(7)]",
                "public string Var_text_col_7{get; set;}",
                "\r\n",
                "[ColumnName(\"var_num\"), LoadColumn(8)]",
                "public double Var_num_col_8{get; set;}",
                "\r\n"
            }
        };
    }
}
