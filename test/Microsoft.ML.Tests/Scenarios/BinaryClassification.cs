using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using System.IO;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        public class BreastCancerData
        {
            [Column(ordinal: "0")]
            public float Label;
            [Column(ordinal: "1-9")]
            [VectorType(9)]
            public float[] Features;
        }

        public class BreastCancerPrediction
        {
            [ColumnName("PredictedLabel")]
            public DvBool Cancerous;
        }

        [Fact]
        public void SaveModelToOnnxTest()
        {
            string dataPath = GetDataPath(@"breast-cancer.txt");
            var pipeline = new LearningPipeline();

            pipeline.Add(new Data.TextLoader(dataPath)
            {
                Arguments = new TextLoaderArguments
                {
                    Separator = new[] { '\t' },
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoaderColumn()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoaderRange(0) },
                            Type = DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Features",
                            Source = new [] { new TextLoaderRange(1, 9) },
                            Type = DataKind.Num
                        }
                    }
                }
            });

            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
            
            PredictionModel<BreastCancerData, BreastCancerPrediction> model = pipeline.Train<BreastCancerData, BreastCancerPrediction>();
            
            var modelOutpath = GetOutputPath(Path.Combine("..", "Common", 
                "Scenario", "BinaryClassification", "BreastCancer"), "SaveModelToOnnxTest.zip");

            DeleteOutputPath(modelOutpath);

            var onnxPath = GetOutputPath(Path.Combine("..", "Common",
                "Scenario", "BinaryClassification", "BreastCancer"), "SaveModelToOnnxTest.pb");

            DeleteOutputPath(onnxPath);

            var onnxAsJsonPath = GetOutputPath(Path.Combine("..", "Common",
                "Scenario", "BinaryClassification", "BreastCancer"), "SaveModelToOnnxTest.json");

            DeleteOutputPath(onnxAsJsonPath);

            model.WriteAsync(modelOutpath);
            SaveAsOnnx.Save(new Runtime.Model.Onnx.SaveOnnxCommand.Arguments
            {
                InputModelFile = modelOutpath,
                OutputsToDrop = "Label,Features",
                Onnx = onnxPath,
                Json = onnxAsJsonPath,
                Domain = "Onnx"
            });

            Assert.True(CheckEquality(Path.Combine("..", "Common", "Scenario", "BinaryClassification", "BreastCancer"),
                "SaveModelToOnnxTest.json"));
        }
    }
}
