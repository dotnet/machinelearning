// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Google.Protobuf;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Tools;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Onnx;
using Newtonsoft.Json;
using Xunit;
using Xunit.Abstractions;
using static Microsoft.ML.Model.OnnxConverter.OnnxCSharpToProtoWrapper;

namespace Microsoft.ML.Tests
{
    public class OnnxConversionTest : BaseTestBaseline
    {
        private class AdultData
        {
            [LoadColumn(0, 10), ColumnName("FeatureVector")]
            public float Features { get; set; }

            [LoadColumn(11)]
            public float Target { get; set; }
        }

        public OnnxConversionTest(ITestOutputHelper output) : base(output)
        {
        }

        private bool IsOnnxRuntimeSupported()
        {
            return Environment.Is64BitProcess && (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || AttributeHelpers.CheckLibcVersionGreaterThanMinimum(new System.Version(2, 23)));
        }

        /// <summary>
        /// In this test, we convert a trained <see cref="TransformerChain"/> into ONNX <see cref="ModelProto"/> file and then
        /// call <see cref="OnnxScoringEstimator"/> to evaluate that file. The outputs of <see cref="OnnxScoringEstimator"/> are checked against the original
        /// ML.NET model's outputs.
        /// </summary>
        [Fact]
        public void SimpleEndToEndOnnxConversionTest()
        {
            // Step 1: Create and train a ML.NET pipeline.
            var trainDataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var mlContext = new MLContext(seed: 1);
            var data = mlContext.Data.LoadFromTextFile<AdultData>(trainDataPath,
                separatorChar: ';'
,
                hasHeader: true);
            var cachedTrainData = mlContext.Data.Cache(data);
            var dynamicPipeline =
                mlContext.Transforms.NormalizeMinMax("FeatureVector")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.Sdca(new SdcaRegressionTrainer.Options() {
                    LabelColumnName = "Target",
                    FeatureColumnName = "FeatureVector",
                    NumberOfThreads = 1
                }));
            var model = dynamicPipeline.Fit(data);
            var transformedData = model.Transform(data);

            // Step 2: Convert ML.NET model to ONNX format and save it as a file.
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, data);
            var onnxFileName = "model.onnx";
            var onnxModelPath = GetOutputPath(onnxFileName);
            SaveOnnxModel(onnxModel, onnxModelPath, null);

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && Environment.Is64BitProcess)
            {
                // Step 3: Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                var onnxTransformer = onnxEstimator.Fit(data);
                var onnxResult = onnxTransformer.Transform(data);

                // Step 4: Compare ONNX and ML.NET results.
                CompareSelectedR4ScalarColumns("Score", "Score0", transformedData, onnxResult, 1);
            }

            // Step 5: Check ONNX model's text format. This test will be not necessary if Step 3 and Step 4 can run on Linux and
            // Mac to support cross-platform tests.
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "Regression", "Adult");
            var onnxTextName = "SimplePipeline.txt";
            var onnxTextPath = GetOutputPath(subDir, onnxTextName);
            SaveOnnxModel(onnxModel, null, onnxTextPath);
            CheckEquality(subDir, onnxTextName, digitsOfPrecision: 3);

            Done();
        }

        private class BreastCancerFeatureVector
        {
            [LoadColumn(1, 9), VectorType(9)]
            public float[] Features;
        }

        private class BreastCancerCatFeatureExample
        {
            [LoadColumn(0)]
            public bool Label;

            [LoadColumn(1)]
            public float F1;

            [LoadColumn(2)]
            public string F2;
        }

        private class BreastCancerMulticlassExample
        {
            [LoadColumn(1)]
            public string Label;

            [LoadColumn(2, 9), VectorType(8)]
            public float[] Features;
        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline. Tracked by https://github.com/dotnet/machinelearning/issues/2087")]
        public void KmeansOnnxConversionTest()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath("breast-cancer.txt");
            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var data = mlContext.Data.LoadFromTextFile<BreastCancerFeatureVector>(dataPath,
                separatorChar: '\t',
                hasHeader: true);

            var pipeline = mlContext.Transforms.NormalizeMinMax("Features").
                Append(mlContext.Clustering.Trainers.KMeans(new Trainers.KMeansTrainer.Options
                {
                    FeatureColumnName = DefaultColumnNames.Features,
                    MaximumNumberOfIterations = 1,
                    NumberOfClusters = 4,
                    NumberOfThreads = 1,
                    InitializationAlgorithm = Trainers.KMeansTrainer.InitializationAlgorithm.Random
                }));

            var model = pipeline.Fit(data);
            var transformedData = model.Transform(data);

            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, data);

            // Compare results produced by ML.NET and ONNX's runtime.
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && Environment.Is64BitProcess)
            {
                var onnxFileName = "model.onnx";
                var onnxModelPath = GetOutputPath(onnxFileName);
                SaveOnnxModel(onnxModel, onnxModelPath, null);

                // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                var onnxTransformer = onnxEstimator.Fit(data);
                var onnxResult = onnxTransformer.Transform(data);
                CompareSelectedR4VectorColumns("Score", "Score0", transformedData, onnxResult, 3);
            }

            // Check ONNX model's text format. We save the produced ONNX model as a text file and compare it against
            // the associated file in ML.NET repo. Such a comparison can be retired if ONNXRuntime ported to ML.NET
            // can support Linux and Mac.
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "Cluster", "BreastCancer");
            var onnxTextName = "Kmeans.txt";
            var onnxTextPath = GetOutputPath(subDir, onnxTextName);
            SaveOnnxModel(onnxModel, null, onnxTextPath);
            CheckEquality(subDir, onnxTextName, digitsOfPrecision: 2);
            Done();
        }

        private class DataPoint
        {
            [VectorType(3)]
            public float[] Features { get; set; }
        }

        [Fact]
        void LpNormOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);

            var samples = new List<DataPoint>()
            {
                new DataPoint() { Features = new float[3] {0.01f, 0.02f, 0.03f} },
                new DataPoint() { Features = new float[3] {0.04f, 0.05f, 0.06f} },
                new DataPoint() { Features = new float[3] {0.07f, 0.08f, 0.09f} },
                new DataPoint() { Features = new float[3] {0.10f, 0.11f, 0.12f} },
                new DataPoint() { Features = new float[3] {0.13f, 0.14f, 0.15f} }
            };
            var dataView = mlContext.Data.LoadFromEnumerable(samples);

            LpNormNormalizingEstimatorBase.NormFunction[] norms =
            {
                LpNormNormalizingEstimatorBase.NormFunction.L1,
                LpNormNormalizingEstimatorBase.NormFunction.L2,
                LpNormNormalizingEstimatorBase.NormFunction.Infinity,
                LpNormNormalizingEstimatorBase.NormFunction.StandardDeviation
            };

            bool[] ensureZeroMeans = { true, false};
            foreach (var ensureZeroMean in ensureZeroMeans)
            {
                foreach (var norm in norms)
                {
                    var pipe = mlContext.Transforms.NormalizeLpNorm(nameof(DataPoint.Features), norm:norm, ensureZeroMean: ensureZeroMean);

                    var model = pipe.Fit(dataView);
                    var transformedData = model.Transform(dataView);
                    var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);

                    var onnxFileName = $"LpNorm-{norm.ToString()}-{ensureZeroMean}.onnx";
                    var onnxModelPath = GetOutputPath(onnxFileName);

                    SaveOnnxModel(onnxModel, onnxModelPath, null);

                    // Compare results produced by ML.NET and ONNX's runtime.
                    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && Environment.Is64BitProcess)
                    {
                        // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                        string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                        string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                        var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                        var onnxTransformer = onnxEstimator.Fit(dataView);
                        var onnxResult = onnxTransformer.Transform(dataView);
                        CompareSelectedR4VectorColumns(nameof(DataPoint.Features), outputNames[0], transformedData, onnxResult, 3);
                    }
                }
            }

            Done();
        }

        [Fact]
        void CommandLineOnnxConversionTest()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            string modelPath = GetOutputPath("ModelWithLessIO.zip");
            var trainingPathArgs = $"data={dataPath} out={modelPath}";
            var trainingArgs = " loader=text{col=Label:BL:0 col=F1:R4:1-8 col=F2:TX:9} xf=Cat{col=F2} xf=Concat{col=Features:F1,F2} tr=ft{numberOfThreads=1 numberOfLeaves=8 numberOfTrees=3} seed=1";
            Assert.Equal(0, Maml.Main(new[] { "train " + trainingPathArgs + trainingArgs }));

            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "BinaryClassification", "BreastCancer");
            var onnxTextName = "ModelWithLessIO.txt";
            var onnxFileName = "ModelWithLessIO.onnx";
            var onnxTextPath = GetOutputPath(subDir, onnxTextName);
            var onnxFilePath = GetOutputPath(subDir, onnxFileName);
            string conversionCommand = $"saveonnx in={modelPath} onnx={onnxFilePath} json={onnxTextPath} domain=machinelearning.dotnet name=modelWithLessIO inputsToDrop=Label outputsToDrop=F1,F2,Features,Label";
            Assert.Equal(0, Maml.Main(new[] { conversionCommand }));

            var fileText = File.ReadAllText(onnxTextPath);
            fileText = Regex.Replace(fileText, "\"producerVersion\": \".*\"", "\"producerVersion\": \"##VERSION##\"");
            File.WriteAllText(onnxTextPath, fileText);

            CheckEquality(subDir, onnxTextName);
            Done();
        }

        [Fact]
        public void KeyToVectorWithBagOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath("breast-cancer.txt");

            var data = mlContext.Data.LoadFromTextFile<BreastCancerCatFeatureExample>(dataPath,
                separatorChar: '\t',
                hasHeader: true);

            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("F2", "F2", Transforms.OneHotEncodingEstimator.OutputKind.Bag)
            .Append(mlContext.Transforms.ReplaceMissingValues(new MissingValueReplacingEstimator.ColumnOptions("F2")))
            .Append(mlContext.Transforms.Concatenate("Features", "F1", "F2"))
            .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features", numberOfLeaves: 2, numberOfTrees: 1, minimumExampleCountPerLeaf: 2));

            var model = pipeline.Fit(data);
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, data);

            // Check ONNX model's text format. We save the produced ONNX model as a text file and compare it against
            // the associated file in ML.NET repo. Such a comparison can be retired if ONNXRuntime ported to ML.NET
            // can support Linux and Mac.
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "BinaryClassification", "BreastCancer");
            var onnxTextName = "OneHotBagPipeline.txt";
            var onnxFileName = "OneHotBagPipeline.onnx";
            var onnxTextPath = GetOutputPath(subDir, onnxTextName);
            var onnxFilePath = GetOutputPath(subDir, onnxFileName);
            SaveOnnxModel(onnxModel, onnxFilePath, onnxTextPath);
            CheckEquality(subDir, onnxTextName);
            Done();
        }

        [Fact]
        public void InitializerCreationTest()
        {
            var env = new MLContext();
            // Create the actual implementation
            var ctxImpl = new OnnxContextImpl(env, "model", "ML.NET", "0", 0, "com.test", Model.OnnxConverter.OnnxVersion.Stable);

            // Use implementation as in the actual conversion code
            var ctx = ctxImpl as OnnxContext;
            ctx.AddInitializer(9.4f, "float");
            ctx.AddInitializer(17L, "int64");
            ctx.AddInitializer("36", "string");
            ctx.AddInitializer(new List<float> { 9.4f, 1.7f, 3.6f }, new List<long> { 1, 3 }, "floats");
            ctx.AddInitializer(new List<long> { 94L, 17L, 36L }, new List<long> { 1, 3 }, "int64s");
            ctx.AddInitializer(new List<string> { "94", "17", "36" }, new List<long> { 1, 3 }, "strings");

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

        [Fact]
        public void LogisticRegressionOnnxConversionTest()
        {
            // Step 1: Create and train a ML.NET pipeline.
            var trainDataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var mlContext = new MLContext(seed: 1);
            var data = mlContext.Data.LoadFromTextFile<AdultData>(trainDataPath,
                separatorChar: ';'
,
                hasHeader: true);
            var cachedTrainData = mlContext.Data.Cache(data);
            var dynamicPipeline =
                mlContext.Transforms.NormalizeMinMax("FeatureVector")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.Sdca(new SdcaRegressionTrainer.Options() {
                    LabelColumnName = "Target",
                    FeatureColumnName = "FeatureVector",
                    NumberOfThreads = 1
                }));
            var model = dynamicPipeline.Fit(data);

            // Step 2: Convert ML.NET model to ONNX format and save it as a file.
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, data);

            // Step 3: Save ONNX model as binary and text files.
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "BinaryClassification", "BreastCancer");
            var onnxFileName = "LogisticRegressionSaveModelToOnnxTest.onnx";
            var onnxFilePath = GetOutputPath(subDir, onnxFileName);
            var onnxTextName = "LogisticRegressionSaveModelToOnnxTest.txt";
            var onnxTextPath = GetOutputPath(subDir, onnxTextName);
            SaveOnnxModel(onnxModel, onnxFilePath, onnxTextPath);

            // Step 4: Check ONNX model's text format.
            CheckEquality(subDir, onnxTextName, digitsOfPrecision: 3);
            Done();
        }

        [LightGBMFact]
        public void LightGbmBinaryClassificationOnnxConversionTest()
        {
            // Step 1: Create and train a ML.NET pipeline.
            var trainDataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var mlContext = new MLContext(seed: 1);
            var data = mlContext.Data.LoadFromTextFile<AdultData>(trainDataPath,
                separatorChar: ';'
,
                hasHeader: true);
            var cachedTrainData = mlContext.Data.Cache(data);
            var dynamicPipeline =
                mlContext.Transforms.NormalizeMinMax("FeatureVector")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.LightGbm(labelColumnName: "Target", featureColumnName: "FeatureVector", numberOfIterations: 3, numberOfLeaves: 16, minimumExampleCountPerLeaf: 100));
            var model = dynamicPipeline.Fit(data);

            // Step 2: Convert ML.NET model to ONNX format and save it as a file.
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, data);

            // Step 3: Save ONNX model as binary and text files.
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "BinaryClassification", "BreastCancer");
            var onnxFileName = "LightGbmBinaryClassificationOnnxConversionTest.onnx";
            var onnxFilePath = GetOutputPath(subDir, onnxFileName);
            var onnxTextName = "LightGbmBinaryClassificationOnnxConversionTest.txt";
            var onnxTextPath = GetOutputPath(subDir, onnxTextName);
            SaveOnnxModel(onnxModel, onnxFilePath, onnxTextPath);

            // Step 4: Check ONNX model's text format.
            CheckEquality(subDir, onnxTextName, digitsOfPrecision: 3);
            Done();
        }

        [Fact]
        public void MulticlassLogisticRegressionOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath("breast-cancer.txt");
            var data = mlContext.Data.LoadFromTextFile<BreastCancerMulticlassExample>(dataPath,
                separatorChar: '\t',
                hasHeader: true);

            var pipeline = mlContext.Transforms.NormalizeMinMax("Features").
                Append(mlContext.Transforms.Conversion.MapValueToKey("Label")).
                Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(new LbfgsMaximumEntropyMulticlassTrainer.Options() { NumberOfThreads = 1 }));

            var model = pipeline.Fit(data);
            var transformedData = model.Transform(data);
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, data);

            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "MultiClassClassification", "BreastCancer");
            var onnxFileName = "MultiClassificationLogisticRegressionSaveModelToOnnxTest.onnx";
            var onnxFilePath = GetOutputPath(subDir, onnxFileName);
            var onnxTextName = "MultiClassificationLogisticRegressionSaveModelToOnnxTest.txt";
            var onnxTextPath = GetOutputPath(subDir, onnxTextName);

            SaveOnnxModel(onnxModel, onnxFilePath, onnxTextPath);

            CheckEquality(subDir, onnxTextName, digitsOfPrecision: 2);
            Done();
        }

        [Fact]
        public void LoadingPredictorModelAndOnnxConversionTest()
        {
            string dataPath = GetDataPath("iris.txt");
            string modelPath = Path.GetTempPath() + Guid.NewGuid().ToString() + ".model.bin";
            string onnxPath = Path.GetTempPath() + Guid.NewGuid().ToString() + ".model.onnx";
            string onnxJsonPath = Path.GetTempPath() + Guid.NewGuid().ToString() + ".model.onnx.json";

            string inputGraph = string.Format(@"
            {{
                'Inputs': {{
                    'inputFile': '{0}'
                }},
                'Nodes': [
                    {{
                        'Name': 'Data.TextLoader',
                        'Inputs':
                        {{
                            'InputFile': '$inputFile',
                            'Arguments':
                            {{
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': true,
                                'AllowSparse': true,
                                'InputSize': null,
                                'TrimWhitespace': false,
                                'HasHeader': false,
                                'Column':
                                [
                                    {{'Name':'Sepal_Width','Type':null,'Source':[{{'Min':2,'Max':2,'AutoEnd':false,'VariableEnd':false,'AllOther':false,'ForceVector':false}}],'KeyCount':null}},
                                    {{'Name':'Petal_Length','Type':null,'Source':[{{'Min':3,'Max':4,'AutoEnd':false,'VariableEnd':false,'AllOther':false,'ForceVector':false}}],'KeyCount':null}},
                                ]
                            }}
                        }},
                        'Outputs':
                        {{
                            'Data': '$training_data'
                        }}
                    }},
                    {{
                        'Inputs': {{
                            'FeatureColumnName': 'Petal_Length',
                            'LabelColumnName': 'Sepal_Width',
                            'TrainingData': '$training_data',
                        }},
                        'Name': 'Trainers.StochasticDualCoordinateAscentRegressor',
                        'Outputs': {{
                            'PredictorModel': '$output_model'
                        }}
                    }}
                ],
                'Outputs': {{
                    'output_model': '{1}'
                }}
            }}", dataPath.Replace("\\", "\\\\"), modelPath.Replace("\\", "\\\\"));

            // Write entry point graph into file so that it can be invoke by graph runner below.
            var jsonPath = DeleteOutputPath("graph.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            // Execute the saved entry point graph to produce a predictive model.
            var args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            var cmd = new ExecuteGraphCommand(Env, args);
            cmd.Run();

            // Make entry point graph to conduct ONNX conversion.
            inputGraph = string.Format(@"
            {{
                'Inputs': {{
                    'model': '{0}'
                }},
                'Nodes': [
                    {{
                        'Inputs': {{
                            'Domain': 'com.microsoft.models',
                            'Json': '{1}',
                            'PredictiveModel': '$model',
                            'Onnx': '{2}',
                            'OnnxVersion': 'Experimental'
                        }},
                        'Name': 'Models.OnnxConverter',
                        'Outputs': {{}}
                    }}
                ],
                'Outputs': {{}}
            }}
            ", modelPath.Replace("\\", "\\\\"), onnxJsonPath.Replace("\\", "\\\\"), onnxPath.Replace("\\", "\\\\"));

            // Write entry point graph for ONNX conversion into file so that it can be invoke by graph runner below.
            jsonPath = DeleteOutputPath("graph.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            // Onnx converter's assembly is not loaded by default, so we need to register it before calling it.
            Env.ComponentCatalog.RegisterAssembly(typeof(OnnxExportExtensions).Assembly);

            // Execute the saved entry point graph to convert the saved model to ONNX format.
            args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            cmd = new ExecuteGraphCommand(Env, args);
            cmd.Run();

            // Load the resulted ONNX model from the file so that we can check if the conversion looks good.
            var model = new OnnxCSharpToProtoWrapper.ModelProto();
            using (var modelStream = File.OpenRead(onnxPath))
                model = OnnxCSharpToProtoWrapper.ModelProto.Parser.ParseFrom(modelStream);

            // Make sure a PredictorModel is loaded by seeing if a predictive model exists. In this the
            // predictive model is "LinearRegressor" (converted from StochasticDualCoordinateAscentRegressor
            // in the original training entry-point graph.
            Assert.Equal("Scaler", model.Graph.Node[0].OpType);
            Assert.Equal("LinearRegressor", model.Graph.Node[1].OpType);

            File.Delete(modelPath);
            File.Delete(onnxPath);
            File.Delete(onnxJsonPath);

            Done();
        }


        [Fact]
        public void RemoveVariablesInPipelineTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath("breast-cancer.txt");
            var data = mlContext.Data.LoadFromTextFile<BreastCancerCatFeatureExample>(dataPath,
                separatorChar: '\t',
                hasHeader: true);

            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("F2", "F2", Transforms.OneHotEncodingEstimator.OutputKind.Bag)
            .Append(mlContext.Transforms.ReplaceMissingValues(new MissingValueReplacingEstimator.ColumnOptions("F2")))
            .Append(mlContext.Transforms.Concatenate("Features", "F1", "F2"))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features", numberOfLeaves: 2, numberOfTrees: 1, minimumExampleCountPerLeaf: 2));

            var model = pipeline.Fit(data);
            var transformedData = model.Transform(data);

            var onnxConversionContext = new OnnxContextImpl(mlContext, "A Simple Pipeline", "ML.NET", "0", 0, "machinelearning.dotnet", OnnxVersion.Stable);

            LinkedList<ITransformCanSaveOnnx> transforms = null;
            using (var conversionChannel = (mlContext as IChannelProvider).Start("ONNX conversion"))
            {
                SaveOnnxCommand.GetPipe(onnxConversionContext, conversionChannel, transformedData, out IDataView root, out IDataView sink, out transforms);
                // Input columns' names to be excluded in the resulted ONNX model.
                var redundantInputColumnNames = new HashSet<string> { "Label" };
                // Output columns' names to be excluded in the resulted ONNX model.
                var redundantOutputColumnNames = new HashSet<string> { "Label", "F1", "F2", "Features" };
                var onnxModel = SaveOnnxCommand.ConvertTransformListToOnnxModel(onnxConversionContext, conversionChannel, root, sink, transforms,
                    redundantInputColumnNames, redundantOutputColumnNames);

                // Check ONNX model's text format. We save the produced ONNX model as a text file and compare it against
                // the associated file in ML.NET repo. Such a comparison can be retired if ONNXRuntime ported to ML.NET
                // can support Linux and Mac.
                var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "BinaryClassification", "BreastCancer");
                var onnxTextName = "ExcludeVariablesInOnnxConversion.txt";
                var onnxFileName = "ExcludeVariablesInOnnxConversion.onnx";
                var onnxTextPath = GetOutputPath(subDir, onnxTextName);
                var onnxFilePath = GetOutputPath(subDir, onnxFileName);
                SaveOnnxModel(onnxModel, onnxFilePath, onnxTextPath);
                CheckEquality(subDir, onnxTextName, digitsOfPrecision: 3);
            }
            Done();
        }

        private class SmallSentimentExample
        {
            [LoadColumn(0, 3), VectorType(4)]
            public string[] Tokens;
        }

        [Fact]
        public void WordEmbeddingsTest()
        {
            var mlContext = new MLContext(seed: 1);
            var dataPath = GetDataPath(@"small-sentiment-test.tsv");
            var embedNetworkPath = GetDataPath(@"shortsentiment.emd");
            var data = mlContext.Data.LoadFromTextFile<SmallSentimentExample>(dataPath, separatorChar: '\t', hasHeader: false);

            var pipeline = mlContext.Transforms.Text.ApplyWordEmbedding("Embed", embedNetworkPath, "Tokens");
            var model = pipeline.Fit(data);
            var transformedData = model.Transform(data);

            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "Transforms", "Sentiment");
            var onnxTextName = "SmallWordEmbed.txt";
            var onnxFileName = "SmallWordEmbed.onnx";
            var onnxTextPath = GetOutputPath(subDir, onnxTextName);
            var onnxFilePath = GetOutputPath(subDir, onnxFileName);
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, data);
            SaveOnnxModel(onnxModel, onnxFilePath, onnxTextPath);

            CheckEquality(subDir, onnxTextName, parseOption: NumberParseOption.UseSingle);
            Done();
        }

        [Fact]
        public void OnnxTypeConversionTest()
        {
            var mlContext = new MLContext(seed: 1);
            string filePath = GetDataPath("type-conversion.txt");

            // These are the supported conversions
            // ML.NET does not allow any conversions between signed and unsigned numeric types
            // Onnx does not seem to support casting a string to any type
            // Though the onnx docs claim support for byte and sbyte, 
            // CreateNamedOnnxValue in OnnxUtils.cs throws a NotImplementedException for those two
            DataKind[,] supportedConversions = new[,]
            {
                { DataKind.Int16, DataKind.Int16},
                { DataKind.Int16, DataKind.Int32},
                { DataKind.Int16, DataKind.Int64},
                { DataKind.Int16, DataKind.Single},
                { DataKind.Int16, DataKind.Double},
                { DataKind.UInt16, DataKind.UInt16},
                { DataKind.UInt16, DataKind.UInt32},
                { DataKind.UInt16, DataKind.UInt64},
                { DataKind.UInt16, DataKind.Single},
                { DataKind.UInt16, DataKind.Double},
                { DataKind.Int32, DataKind.Int16},
                { DataKind.Int32, DataKind.Int32},
                { DataKind.Int32, DataKind.Int64},
                { DataKind.Int32, DataKind.Single},
                { DataKind.Int32, DataKind.Double},
                { DataKind.Int64, DataKind.Int16},
                { DataKind.Int64, DataKind.Int32},
                { DataKind.Int64, DataKind.Int64},
                { DataKind.Int64, DataKind.Single},
                { DataKind.Int64, DataKind.Double},
                { DataKind.UInt64, DataKind.UInt16},
                { DataKind.UInt64, DataKind.UInt32},
                { DataKind.UInt64, DataKind.UInt64},
                { DataKind.UInt64, DataKind.Single},
                { DataKind.UInt64, DataKind.Double},
                { DataKind.Single, DataKind.Single},
                { DataKind.Single, DataKind.Double},
                { DataKind.Double, DataKind.Single},
                { DataKind.Double, DataKind.Double}
            };

            for (int i = 0; i < supportedConversions.GetLength(0); i++)
            {
                var fromKind = supportedConversions[i, 0];
                var toKind = supportedConversions[i, 1];

                TextLoader.Column[] columns = new []
                {
                    new TextLoader.Column("Value", fromKind, 0, 0)
                };
                var dataView = mlContext.Data.LoadFromTextFile(filePath, columns);

                var pipeline = mlContext.Transforms.Conversion.ConvertType("ValueConverted", "Value", outputKind: toKind);
                var model = pipeline.Fit(dataView);
                var mlnetResult = model.Transform(dataView);

                var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);
                var onnxFileName = "typeconversion.onnx";
                var onnxModelPath = GetOutputPath(onnxFileName);
                SaveOnnxModel(onnxModel, onnxModelPath, null);

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && Environment.Is64BitProcess)
                {
                    string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                    var onnxTransformer = onnxEstimator.Fit(dataView);
                    var onnxResult = onnxTransformer.Transform(dataView);

                    CompareResults(model.ColumnPairs[0].outputColumnName, outputNames[1], mlnetResult, onnxResult);
                }
            }
            Done();
        }

        [Fact]
        public void PcaOnnxConversionTest()
        {
            var dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);

            var mlContext = new MLContext(seed: 1);
            var dataView = mlContext.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.Single, 0, 10)
            }, hasHeader: true, separatorChar: ';');

            bool[] zeroMeans = { true, false };
            foreach (var zeroMean in zeroMeans)
            {
                var pipeline = ML.Transforms.ProjectToPrincipalComponents("pca", "features", rank: 5, seed: 1, ensureZeroMean: zeroMean);
                var model = pipeline.Fit(dataView);
                var transformedData = model.Transform(dataView);
                var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);

                var onnxFileName = "pca.onnx";
                var onnxModelPath = GetOutputPath(onnxFileName);

                SaveOnnxModel(onnxModel, onnxModelPath, null);

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && Environment.Is64BitProcess)
                {
                    // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                    string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                    var onnxTransformer = onnxEstimator.Fit(dataView);
                    var onnxResult = onnxTransformer.Transform(dataView);
                    CompareSelectedR4VectorColumns(model.ColumnPairs[0].outputColumnName, outputNames[2], transformedData, onnxResult);
                }
            }

            Done();
        }

        private class TransformedDataPoint : DataPoint, IEquatable<TransformedDataPoint>
        {
            [VectorType(3)]
            public int[] MissingIndicator { get; set; }

            public bool Equals(TransformedDataPoint other)
            {
                return Enumerable.SequenceEqual(MissingIndicator, other.MissingIndicator);
            }
        }

        [Fact]
        void IndicateMissingValuesOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);

            var samples = new List<DataPoint>()
            {
                new DataPoint() { Features = new float[3] {1, 1, 0}, },
                new DataPoint() { Features = new float[3] {0, float.NaN, 1}, },
                new DataPoint() { Features = new float[3] {-1, float.NaN, float.PositiveInfinity}, },
            };
            var dataView = mlContext.Data.LoadFromEnumerable(samples);

            // IsNaN outputs a binary tensor. Support for this has been added in the latest version
            // of Onnxruntime, but that hasn't been released yet.
            // So we need to convert its type to Int32 until then. 
            // ConvertType part of the pipeline can be removed once we pick up a new release of the Onnx runtime

            var pipeline = mlContext.Transforms.IndicateMissingValues(new[] { new InputOutputColumnPair("MissingIndicator", "Features"), })
                            .Append(mlContext.Transforms.Conversion.ConvertType("MissingIndicator", outputKind: DataKind.Int32));

            var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);
            var mlnetData = mlContext.Data.CreateEnumerable<TransformedDataPoint>(transformedData, false);
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);

            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "Transforms");
            var onnxFileName = "IndicateMissingValues.onnx";
            var onnxTextName = "IndicateMissingValues.txt";
            var onnxModelPath = GetOutputPath(onnxFileName);
            var onnxTextPath = GetOutputPath(subDir, onnxTextName);

            SaveOnnxModel(onnxModel, onnxModelPath, onnxTextPath);

            // Compare results produced by ML.NET and ONNX's runtime.
            if (IsOnnxRuntimeSupported())
            {
                // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                var onnxTransformer = onnxEstimator.Fit(dataView);
                var onnxResult = onnxTransformer.Transform(dataView);
                CompareSelectedVectorColumns<int>(model.LastTransformer.ColumnPairs[0].outputColumnName, outputNames[1], transformedData, onnxResult);
            }

            CheckEquality(subDir, onnxTextName, parseOption: NumberParseOption.UseSingle);
            Done();
        }

        private void CreateDummyExamplesToMakeComplierHappy()
        {
            var dummyExample = new BreastCancerFeatureVector() { Features = null };
            var dummyExample1 = new BreastCancerCatFeatureExample() { Label = false, F1 = 0, F2 = "Amy" };
            var dummyExample2 = new BreastCancerMulticlassExample() { Label = "Amy", Features = null };
            var dummyExample3 = new SmallSentimentExample() { Tokens = null };
        }

        private void CompareResults(string leftColumnName, string rightColumnName, IDataView left, IDataView right)
        {
            var leftColumn = left.Schema[leftColumnName];
            var rightColumn = right.Schema[rightColumnName];
            var leftType = leftColumn.Type.GetItemType();
            var rightType = rightColumn.Type.GetItemType();
            Assert.Equal(leftType, rightType);

            if (leftType == NumberDataViewType.SByte)
                CompareSelectedVectorColumns<sbyte>(leftColumnName, rightColumnName, left, right);
            else if (leftType == NumberDataViewType.Byte)
                CompareSelectedVectorColumns<byte>(leftColumnName, rightColumnName, left, right);
            else if (leftType == NumberDataViewType.Int16)
                CompareSelectedVectorColumns<short>(leftColumnName, rightColumnName, left, right);
            else if (leftType == NumberDataViewType.UInt16)
                CompareSelectedVectorColumns<ushort>(leftColumnName, rightColumnName, left, right);
            else if (leftType == NumberDataViewType.Int32)
                CompareSelectedVectorColumns<int>(leftColumnName, rightColumnName, left, right);
            else if (leftType == NumberDataViewType.UInt32)
                CompareSelectedVectorColumns<uint>(leftColumnName, rightColumnName, left, right);
            else if (leftType == NumberDataViewType.Int64)
                CompareSelectedVectorColumns<long>(leftColumnName, rightColumnName, left, right);
            else if (leftType == NumberDataViewType.UInt64)
                CompareSelectedVectorColumns<ulong>(leftColumnName, rightColumnName, left, right);
            else if (leftType == NumberDataViewType.Single)
                CompareSelectedR4VectorColumns(leftColumnName, rightColumnName, left, right);
            else if (leftType == NumberDataViewType.Double)
                CompareSelectedVectorColumns<double>(leftColumnName, rightColumnName, left, right);
        }

        private void CompareSelectedVectorColumns<T>(string leftColumnName, string rightColumnName, IDataView left, IDataView right)
        {
            var leftColumn = left.Schema[leftColumnName];
            var rightColumn = right.Schema[rightColumnName];

            using (var expectedCursor = left.GetRowCursor(leftColumn))
            using (var actualCursor = right.GetRowCursor(rightColumn))
            {
                VBuffer<T> expected = default;
                VBuffer<T> actual = default;
                var expectedGetter = expectedCursor.GetGetter<VBuffer<T>>(leftColumn);
                var actualGetter = actualCursor.GetGetter<VBuffer<T>>(rightColumn);
                while (expectedCursor.MoveNext() && actualCursor.MoveNext())
                {
                    expectedGetter(ref expected);
                    actualGetter(ref actual);

                    Assert.Equal(expected.Length, actual.Length);
                    for (int i = 0; i < expected.Length; ++i)
                        Assert.Equal(expected.GetItemOrDefault(i), actual.GetItemOrDefault(i));
                }
            }
        }

        private void CompareSelectedR8VectorColumns(string leftColumnName, string rightColumnName, IDataView left, IDataView right, int precision = 6)
        {
            var leftColumn = left.Schema[leftColumnName];
            var rightColumn = right.Schema[rightColumnName];

            using (var expectedCursor = left.GetRowCursor(leftColumn))
            using (var actualCursor = right.GetRowCursor(rightColumn))
            {
                VBuffer<double> expected = default;
                VBuffer<double> actual = default;
                var expectedGetter = expectedCursor.GetGetter<VBuffer<double>>(leftColumn);
                var actualGetter = actualCursor.GetGetter<VBuffer<double>>(rightColumn);
                while (expectedCursor.MoveNext() && actualCursor.MoveNext())
                {
                    expectedGetter(ref expected);
                    actualGetter(ref actual);

                    Assert.Equal(expected.Length, actual.Length);
                    for (int i = 0; i < expected.Length; ++i)
                        Assert.Equal(expected.GetItemOrDefault(i), actual.GetItemOrDefault(i), precision);
                }
            }
        }

        private void CompareSelectedR4VectorColumns(string leftColumnName, string rightColumnName, IDataView left, IDataView right, int precision = 6)
        {
            var leftColumn = left.Schema[leftColumnName];
            var rightColumn = right.Schema[rightColumnName];

            using (var expectedCursor = left.GetRowCursor(leftColumn))
            using (var actualCursor = right.GetRowCursor(rightColumn))
            {
                VBuffer<float> expected = default;
                VBuffer<float> actual = default;
                var expectedGetter = expectedCursor.GetGetter<VBuffer<float>>(leftColumn);
                var actualGetter = actualCursor.GetGetter<VBuffer<float>>(rightColumn);
                while (expectedCursor.MoveNext() && actualCursor.MoveNext())
                {
                    expectedGetter(ref expected);
                    actualGetter(ref actual);

                    Assert.Equal(expected.Length, actual.Length);
                    for (int i = 0; i < expected.Length; ++i)
                    {
                        // We are using float values. But the Assert.Equal function takes doubles.
                        // And sometimes the converted doubles are different in their precision.
                        // So make sure we compare floats
                        float exp = expected.GetItemOrDefault(i);
                        float act = actual.GetItemOrDefault(i);
                        CompareNumbersWithTolerance(exp, act, null, precision);
                    }
                }
            }
        }

        private void CompareSelectedR4ScalarColumns(string leftColumnName, string rightColumnName, IDataView left, IDataView right, int precision = 6)
        {
            var leftColumn = left.Schema[leftColumnName];
            var rightColumn = right.Schema[rightColumnName];

            using (var expectedCursor = left.GetRowCursor(leftColumn))
            using (var actualCursor = right.GetRowCursor(rightColumn))
            {
                float expected = default;
                VBuffer<float> actual = default;
                var expectedGetter = expectedCursor.GetGetter<float>(leftColumn);
                var actualGetter = actualCursor.GetGetter<VBuffer<float>>(rightColumn);
                while (expectedCursor.MoveNext() && actualCursor.MoveNext())
                {
                    expectedGetter(ref expected);
                    actualGetter(ref actual);

                    // Scalar such as R4 (float) is converted to [1, 1]-tensor in ONNX format for consitency of making batch prediction.
                    Assert.Equal(1, actual.Length);
                    Assert.Equal(expected, actual.GetItemOrDefault(0), precision);
                }
            }
        }

        private void SaveOnnxModel(ModelProto model, string binaryFormatPath, string textFormatPath)
        {
            DeleteOutputPath(binaryFormatPath); // Clean if such a file exists.
            DeleteOutputPath(textFormatPath);

            if (binaryFormatPath != null)
                using (var file = Env.CreateOutputFile(binaryFormatPath))
                using (var stream = file.CreateWriteStream())
                    model.WriteTo(stream);

            if (textFormatPath != null)
            {
                using (var file = Env.CreateOutputFile(textFormatPath))
                using (var stream = file.CreateWriteStream())
                using (var writer = new StreamWriter(stream))
                {
                    var parsedJson = JsonConvert.DeserializeObject(model.ToString());
                    writer.Write(JsonConvert.SerializeObject(parsedJson, Formatting.Indented));
                }

                // Strip the version information.
                var fileText = File.ReadAllText(textFormatPath);

                fileText = Regex.Replace(fileText, "\"producerVersion\": \".*\"", "\"producerVersion\": \"##VERSION##\"");
                File.WriteAllText(textFormatPath, fileText);
            }
        }
    }
}
