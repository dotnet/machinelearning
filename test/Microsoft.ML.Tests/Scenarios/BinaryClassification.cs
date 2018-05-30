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
            public float Label;

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

            var model = pipeline.Train<BreastCancerData, BreastCancerPrediction>();
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Scenario", "BinaryClassification", "BreastCancer");
            var modelOutpath = GetOutputPath(subDir, "SaveModelToOnnxTest.zip");
            DeleteOutputPath(modelOutpath);

            var onnxPath = GetOutputPath(subDir, "SaveModelToOnnxTest.pb");
            DeleteOutputPath(onnxPath);

            var onnxAsJsonPath = GetOutputPath(subDir, "SaveModelToOnnxTest.json");
            DeleteOutputPath(onnxAsJsonPath);

            OnnxConverter converter = new OnnxConverter()
            {
                InputsToDrop = new[] { "Label" },
                OutputsToDrop = new[] { "Label", "Features" },
                Onnx = onnxPath,
                Json = onnxAsJsonPath,
                Domain = "Onnx"
            };

            converter.Convert(model);

            CheckEquality(subDir, "SaveModelToOnnxTest.json");
            Done();
        }
    }
}
