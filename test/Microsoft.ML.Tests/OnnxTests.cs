// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Trainers;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class OnnxTests : BaseTestBaseline
    {
        public OnnxTests(ITestOutputHelper output) : base(output)
        {
        }

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
        public void BinaryClassificationSaveModelToOnnxTest()
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
                            Type = Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Features",
                            Source = new [] { new TextLoaderRange(1, 9) },
                            Type = Data.DataKind.Num
                        }
                    }
                }
            });

            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            var model = pipeline.Train<BreastCancerData, BreastCancerPrediction>();
            var subDir = Path.Combine("..", "..", TestFolder, CommonFolder, "Onnx", "BinaryClassification", "BreastCancer");
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
