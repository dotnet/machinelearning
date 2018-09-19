// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using System;
using System.IO;
using System.Text.RegularExpressions;
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

            public float F1;
            public ReadOnlyMemory<char> F2;
        }

        public class BreastCancerDataAllColumns
        {
            public float Label;

            [VectorType(9)]
            public float[] Features;
        }

        public class BreastCancerPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Cancerous;
        }

        public class BreastCancerMCPrediction
        {
            [ColumnName("Score")]
            public float[] Scores;
        }

        [Fact]
        public void BinaryClassificationFastTreeSaveModelToOnnxTest()
        {
            string dataPath = GetDataPath(@"breast-cancer.txt");
            var pipeline = new Legacy.LearningPipeline();

            pipeline.Add(new Legacy.Data.TextLoader(dataPath)
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
                            Type = Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "F1",
                            Source = new [] { new TextLoaderRange(1, 1) },
                            Type = Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "F2",
                            Source = new [] { new TextLoaderRange(2, 2) },
                            Type = Legacy.Data.DataKind.TX
                        }
                    }
                }
            });

            pipeline.Add(new MissingValueSubstitutor("F1"));
            pipeline.Add(new MinMaxNormalizer("F1"));
            pipeline.Add(new CategoricalOneHotVectorizer("F2"));
            pipeline.Add(new ColumnConcatenator("Features", "F1", "F2"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 2, NumTrees = 1, MinDocumentsInLeafs = 2 });

            var model = pipeline.Train<BreastCancerData, BreastCancerPrediction>();
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "BinaryClassification", "BreastCancer");
            var onnxPath = GetOutputPath(subDir, "BinaryClassificationFastTreeSaveModelToOnnxTest.onnx");
            DeleteOutputPath(onnxPath);

            var onnxAsJsonPath = GetOutputPath(subDir, "BinaryClassificationFastTreeSaveModelToOnnxTest.json");
            DeleteOutputPath(onnxAsJsonPath);

            OnnxConverter converter = new OnnxConverter()
            {
                InputsToDrop = new[] { "Label" },
                OutputsToDrop = new[] { "Label", "F1", "F2", "Features" },
                Onnx = onnxPath,
                Json = onnxAsJsonPath,
                Domain = "Onnx"
            };

            converter.Convert(model);

            // Strip the version.
            var fileText = File.ReadAllText(onnxAsJsonPath);
            fileText = Regex.Replace(fileText, "\"producerVersion\": \"([^\"]+)\"", "\"producerVersion\": \"##VERSION##\"");
            File.WriteAllText(onnxAsJsonPath, fileText);

            CheckEquality(subDir, "BinaryClassificationFastTreeSaveModelToOnnxTest.json");
            Done();
        }

        [Fact]
        public void BinaryClassificationLightGBMSaveModelToOnnxTest()
        {
            string dataPath = GetDataPath(@"breast-cancer.txt");
            var pipeline = new Legacy.LearningPipeline();

            pipeline.Add(new Legacy.Data.TextLoader(dataPath)
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
                            Type = Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Features",
                            Source = new [] { new TextLoaderRange(1, 9) },
                            Type = Legacy.Data.DataKind.Num
                        }
                    }
                }
            });

            pipeline.Add(new LightGbmBinaryClassifier() { NumLeaves = 2, NumBoostRound = 1, MinDataPerLeaf = 2 });

            var model = pipeline.Train<BreastCancerDataAllColumns, BreastCancerPrediction>();
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "BinaryClassification", "BreastCancer");
            var onnxPath = GetOutputPath(subDir, "BinaryClassificationLightGBMSaveModelToOnnxTest.onnx");
            DeleteOutputPath(onnxPath);

            var onnxAsJsonPath = GetOutputPath(subDir, "BinaryClassificationLightGBMSaveModelToOnnxTest.json");
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

            // Strip the version.
            var fileText = File.ReadAllText(onnxAsJsonPath);
            fileText = Regex.Replace(fileText, "\"producerVersion\": \"([^\"]+)\"", "\"producerVersion\": \"##VERSION##\"");
            File.WriteAllText(onnxAsJsonPath, fileText);

            CheckEquality(subDir, "BinaryClassificationLightGBMSaveModelToOnnxTest.json");
            Done();
        }

        [Fact]
        public void BinaryClassificationLRSaveModelToOnnxTest()
        {
            string dataPath = GetDataPath(@"breast-cancer.txt");
            var pipeline = new Legacy.LearningPipeline();

            pipeline.Add(new Legacy.Data.TextLoader(dataPath)
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
                            Type = Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Features",
                            Source = new [] { new TextLoaderRange(1, 9) },
                            Type = Legacy.Data.DataKind.Num
                        }
                    }
                }
            });

            pipeline.Add(new LogisticRegressionBinaryClassifier() { UseThreads = false });

            var model = pipeline.Train<BreastCancerDataAllColumns, BreastCancerPrediction>();
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "BinaryClassification", "BreastCancer");
            var onnxPath = GetOutputPath(subDir, "BinaryClassificationLRSaveModelToOnnxTest.onnx");
            DeleteOutputPath(onnxPath);

            var onnxAsJsonPath = GetOutputPath(subDir, "BinaryClassificationLRSaveModelToOnnxTest.json");
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

            // Strip the version.
            var fileText = File.ReadAllText(onnxAsJsonPath);
            fileText = Regex.Replace(fileText, "\"producerVersion\": \"([^\"]+)\"", "\"producerVersion\": \"##VERSION##\"");
            File.WriteAllText(onnxAsJsonPath, fileText);

            CheckEquality(subDir, "BinaryClassificationLRSaveModelToOnnxTest.json");
            Done();
        }

        [Fact]
        public void MultiClassificationLRSaveModelToOnnxTest()
        {
            string dataPath = GetDataPath(@"breast-cancer.txt");
            var pipeline = new Legacy.LearningPipeline();

            pipeline.Add(new Legacy.Data.TextLoader(dataPath)
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
                            Type = Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Features",
                            Source = new [] { new TextLoaderRange(1, 9) },
                            Type = Legacy.Data.DataKind.Num
                        }
                    }
                }
            });

            pipeline.Add(new Dictionarizer("Label"));
            pipeline.Add(new LogisticRegressionClassifier() { UseThreads = false });

            var model = pipeline.Train<BreastCancerDataAllColumns, BreastCancerMCPrediction>();
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "MultiClassClassification", "BreastCancer");
            var onnxPath = GetOutputPath(subDir, "MultiClassificationLRSaveModelToOnnxTest.onnx");
            DeleteOutputPath(onnxPath);

            var onnxAsJsonPath = GetOutputPath(subDir, "MultiClassificationLRSaveModelToOnnxTest.json");
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

            // Strip the version.
            var fileText = File.ReadAllText(onnxAsJsonPath);
            fileText = Regex.Replace(fileText, "\"producerVersion\": \"([^\"]+)\"", "\"producerVersion\": \"##VERSION##\"");
            File.WriteAllText(onnxAsJsonPath, fileText);

            CheckEquality(subDir, "MultiClassificationLRSaveModelToOnnxTest.json");
            Done();
        }

    }
}
