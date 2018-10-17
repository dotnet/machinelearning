// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.RunTests;
using System;
using System.Collections.Generic;
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

        public class EmbeddingsData
        {
            [VectorType(4)]
            public string[] Cat;
        }

        public class EmbeddingsResult
        {
            [ColumnName("Cat")]
            public float[] Cat;
        }

        public class BreastNumericalColumns
        {
            [VectorType(9)]
            public float[] Features;
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

        public class BreastCancerClusterPrediction
        {
            [ColumnName("PredictedLabel")]
            public uint NearestCluster;
            [ColumnName("Score")]
            public float[] Distances;
        }

        [Fact]
        public void InitializerCreationTest()
        {
            using (var env = new ConsoleEnvironment())
            {
                // Create the actual implementation
                var ctxImpl = new OnnxContextImpl(env, "model", "ML.NET", "0", 0, "com.test", Runtime.Model.Onnx.OnnxVersion.Stable);

                // Use implementation as in the actual conversion code
                var ctx = ctxImpl as OnnxContext;
                ctx.AddInitializer(9.4f, "float");
                ctx.AddInitializer(17L, "int64");
                ctx.AddInitializer("36", "string");
                ctx.AddInitializer(new List<float> { 9.4f, 1.7f, 3.6f }, new List<long> { 1, 3 }, "floats");
                ctx.AddInitializer(new List<long> { 94L, 17L, 36L }, new List<long> { 1, 3 }, "int64s");
                ctx.AddInitializer(new List<string> { "94" , "17", "36" }, new List<long> { 1, 3 }, "strings");

                var model = ctxImpl.MakeModel();

                var floatScalar = model.Graph.Initializer[0];
                Assert.True(floatScalar.Name == "float");
                Assert.True(floatScalar.Dims.Count == 0);
                Assert.True(floatScalar.FloatData.Count == 1);
                Assert.True(floatScalar.FloatData[0] == 9.4f);

                var int64Scalar = model.Graph.Initializer[1];
                Assert.True(int64Scalar.Name == "int64");
                Assert.True(int64Scalar.Dims.Count == 0);
                Assert.True(int64Scalar.Int64Data.Count == 1);
                Assert.True(int64Scalar.Int64Data[0] == 17L);

                var stringScalar = model.Graph.Initializer[2];
                Assert.True(stringScalar.Name == "string");
                Assert.True(stringScalar.Dims.Count == 0);
                Assert.True(stringScalar.StringData.Count == 1);
                Assert.True(stringScalar.StringData[0].ToStringUtf8() == "36");

                var floatsTensor = model.Graph.Initializer[3];
                Assert.True(floatsTensor.Name == "floats");
                Assert.True(floatsTensor.Dims.Count == 2);
                Assert.True(floatsTensor.Dims[0] == 1);
                Assert.True(floatsTensor.Dims[1] == 3);
                Assert.True(floatsTensor.FloatData.Count == 3);
                Assert.True(floatsTensor.FloatData[0] == 9.4f);
                Assert.True(floatsTensor.FloatData[1] == 1.7f);
                Assert.True(floatsTensor.FloatData[2] == 3.6f);

                var int64sTensor = model.Graph.Initializer[4];
                Assert.True(int64sTensor.Name == "int64s");
                Assert.True(int64sTensor.Dims.Count == 2);
                Assert.True(int64sTensor.Dims[0] == 1);
                Assert.True(int64sTensor.Dims[1] == 3);
                Assert.True(int64sTensor.Int64Data.Count == 3);
                Assert.True(int64sTensor.Int64Data[0] == 94L);
                Assert.True(int64sTensor.Int64Data[1] == 17L);
                Assert.True(int64sTensor.Int64Data[2] == 36L);

                var stringsTensor = model.Graph.Initializer[5];
                Assert.True(stringsTensor.Name == "strings");
                Assert.True(stringsTensor.Dims.Count == 2);
                Assert.True(stringsTensor.Dims[0] == 1);
                Assert.True(stringsTensor.Dims[1] == 3);
                Assert.True(stringsTensor.StringData.Count == 3);
                Assert.True(stringsTensor.StringData[0].ToStringUtf8() == "94");
                Assert.True(stringsTensor.StringData[1].ToStringUtf8() == "17");
                Assert.True(stringsTensor.StringData[2].ToStringUtf8() == "36");
            }
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
        public void KeyToVectorWithBagTest()
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

            var vectorizer = new CategoricalOneHotVectorizer();
            var categoricalColumn = new CategoricalTransformColumn() {
                OutputKind = CategoricalTransformOutputKind.Bag, Name = "F2", Source = "F2" };
            vectorizer.Column = new CategoricalTransformColumn[1] { categoricalColumn };
            pipeline.Add(vectorizer);
            pipeline.Add(new ColumnConcatenator("Features", "F1", "F2"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 2, NumTrees = 1, MinDocumentsInLeafs = 2 });

            var model = pipeline.Train<BreastCancerData, BreastCancerPrediction>();
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "BinaryClassification", "BreastCancer");
            var onnxPath = GetOutputPath(subDir, "KeyToVectorBag.onnx");
            DeleteOutputPath(onnxPath);

            var onnxAsJsonPath = GetOutputPath(subDir, "KeyToVectorBag.json");
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

            CheckEquality(subDir, "KeyToVectorBag.json");
            Done();
        }

        [Fact]
        public void WordEmbeddingsTest()
        {
            string dataPath = GetDataPath(@"small-sentiment-test.tsv");
            var pipeline = new Legacy.LearningPipeline(0);

            pipeline.Add(new Legacy.Data.TextLoader(dataPath)
            {
                Arguments = new TextLoaderArguments
                {
                    Separator = new[] { '\t' },
                    HasHeader = false,
                    Column = new []
                    {
                        new TextLoaderColumn()
                        {
                            Name = "Cat",
                            Source = new [] { new TextLoaderRange(0, 3) },
                            Type = Legacy.Data.DataKind.TX
                        },
                    }
                }
            });

            //var embed = new WordEmbeddings(new string[1] { "Features" });
            var modelPath = GetDataPath(@"shortsentiment.emd");
            var embed = new WordEmbeddings() { CustomLookupTable = modelPath };
            embed.AddColumn("Cat", "Cat");
            pipeline.Add(embed);
            var model = pipeline.Train<EmbeddingsData, EmbeddingsResult>();

            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "WordEmbeddings");
            var onnxPath = GetOutputPath(subDir, "WordEmbeddings.onnx");
            DeleteOutputPath(onnxPath);

            var onnxAsJsonPath = GetOutputPath(subDir, "WordEmbeddings.json");
            DeleteOutputPath(onnxAsJsonPath);

            OnnxConverter converter = new OnnxConverter()
            {
                Onnx = onnxPath,
                Json = onnxAsJsonPath,
                Domain = "Onnx"
            };

            converter.Convert(model);

            var fileText = File.ReadAllText(onnxAsJsonPath);
            fileText = Regex.Replace(fileText, "\"producerVersion\": \"([^\"]+)\"", "\"producerVersion\": \"##VERSION##\"");
            File.WriteAllText(onnxAsJsonPath, fileText);

            CheckEquality(subDir, "WordEmbeddings.json");
            Done();
        }

        [Fact]
        public void KmeansTest()
        {
            string dataPath = GetDataPath(@"breast-cancer.txt");
            var pipeline = new Legacy.LearningPipeline(0);

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
                            Name = "Features",
                            Source = new [] { new TextLoaderRange(1, 9) },
                            Type = Legacy.Data.DataKind.R4
                        },
                    }
                }
            });

            pipeline.Add(new KMeansPlusPlusClusterer() { K = 2, MaxIterations = 1, NumThreads = 1, InitAlgorithm = KMeansPlusPlusTrainerInitAlgorithm.Random });
            var model = pipeline.Train<BreastNumericalColumns, BreastCancerClusterPrediction>();
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "Cluster", "BreastCancer");
            var onnxPath = GetOutputPath(subDir, "Kmeans.onnx");
            DeleteOutputPath(onnxPath);

            var onnxAsJsonPath = GetOutputPath(subDir, "Kmeans.json");
            DeleteOutputPath(onnxAsJsonPath);

            OnnxConverter converter = new OnnxConverter()
            {
                Onnx = onnxPath,
                Json = onnxAsJsonPath,
                Domain = "Onnx"
            };

            converter.Convert(model);

            // Strip the version.
            var fileText = File.ReadAllText(onnxAsJsonPath);
            fileText = Regex.Replace(fileText, "\"producerVersion\": \"([^\"]+)\"", "\"producerVersion\": \"##VERSION##\"");
            File.WriteAllText(onnxAsJsonPath, fileText);

            CheckEquality(subDir, "Kmeans.json");
            Done();
        }


        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
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
