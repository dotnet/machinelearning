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
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.ML.Tools;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Onnx;
using Microsoft.ML.Transforms.Text;
using Newtonsoft.Json;
using Xunit;
using Xunit.Abstractions;
using static Microsoft.ML.Model.OnnxConverter.OnnxCSharpToProtoWrapper;

#pragma warning disable CS0649 // Field 'fieldName' is never assigned to, and will always have its default value null

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
            return OnnxFactAttribute.IsOnnxRuntimeSupported;
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

        private class BreastCancerBinaryClassification
        {
            [LoadColumn(0)]
            public bool Label;

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

        [Fact]
        //Skipping test temporarily. This test will be re-enabled once the cause of failures has been determined
        [Trait("Category", "SkipInCI")]
        public void RegressionTrainersOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);
            string dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);

            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var dataView = mlContext.Data.LoadFromTextFile<AdultData>(dataPath,
                separatorChar: ';',
                hasHeader: true);
            List<IEstimator<ITransformer>> estimators = new List<IEstimator<ITransformer>>()
            {
                mlContext.Regression.Trainers.Sdca("Target","FeatureVector"),
                mlContext.Regression.Trainers.Ols("Target","FeatureVector"),
                mlContext.Regression.Trainers.OnlineGradientDescent("Target","FeatureVector"),
                mlContext.Regression.Trainers.FastForest("Target", "FeatureVector"),
                mlContext.Regression.Trainers.FastTree("Target", "FeatureVector"),
                mlContext.Regression.Trainers.FastTreeTweedie("Target", "FeatureVector"),
                mlContext.Regression.Trainers.LbfgsPoissonRegression("Target", "FeatureVector"),
            };
            if (Environment.Is64BitProcess)
            {
                estimators.Add(mlContext.Regression.Trainers.LightGbm("Target", "FeatureVector"));
            }
            foreach (var estimator in estimators)
            {
                var model = estimator.Fit(dataView);
                var transformedData = model.Transform(dataView);
                var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);
                // Compare model scores produced by ML.NET and ONNX's runtime
                if (IsOnnxRuntimeSupported())
                {
                    var onnxFileName = $"{estimator.ToString()}.onnx";
                    var onnxModelPath = GetOutputPath(onnxFileName);
                    SaveOnnxModel(onnxModel, onnxModelPath, null);
                    // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                    string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                    var onnxTransformer = onnxEstimator.Fit(dataView);
                    var onnxResult = onnxTransformer.Transform(dataView);
                    CompareSelectedR4ScalarColumns(transformedData.Schema[2].Name, outputNames[2], transformedData, onnxResult, 3); // compare score results 
                }
                // Compare the Onnx graph to a baseline if OnnxRuntime is not supported
                else
                {
                    var onnxFileName = $"{estimator.ToString()}.txt";
                    var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "Regression", "Adult");
                    var onnxTextModelPath = GetOutputPath(subDir, onnxFileName);
                    SaveOnnxModel(onnxModel, null, onnxTextModelPath);
                    CheckEquality(subDir, onnxFileName, digitsOfPrecision: 1);
                }
            }
            Done();
        }

        [Fact]
        public void BinaryClassificationTrainersOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);
            string dataPath = GetDataPath("breast-cancer.txt");
            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var dataView = mlContext.Data.LoadFromTextFile<BreastCancerBinaryClassification>(dataPath, separatorChar: '\t', hasHeader: true);
            List<IEstimator<ITransformer>> estimators = new List<IEstimator<ITransformer>>()
            {
                mlContext.BinaryClassification.Trainers.AveragedPerceptron(),
                mlContext.BinaryClassification.Trainers.FastForest(),
                mlContext.BinaryClassification.Trainers.FastTree(),
                mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(),
                mlContext.BinaryClassification.Trainers.LinearSvm(),
                mlContext.BinaryClassification.Trainers.Prior(),
                mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(),
                mlContext.BinaryClassification.Trainers.SdcaNonCalibrated(),
                mlContext.BinaryClassification.Trainers.SgdCalibrated(),
                mlContext.BinaryClassification.Trainers.SgdNonCalibrated(),
                mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression(),
            };
            if (Environment.Is64BitProcess)
            {
                estimators.Add(mlContext.BinaryClassification.Trainers.LightGbm());
            }

            var initialPipeline = mlContext.Transforms.ReplaceMissingValues("Features").
                Append(mlContext.Transforms.NormalizeMinMax("Features"));
            foreach (var estimator in estimators)
            {
                var pipeline = initialPipeline.Append(estimator);
                var model = pipeline.Fit(dataView);
                var transformedData = model.Transform(dataView);
                var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);
                // Compare model scores produced by ML.NET and ONNX's runtime.
                if (IsOnnxRuntimeSupported())
                {
                    var onnxFileName = $"{estimator.ToString()}.onnx";
                    var onnxModelPath = GetOutputPath(onnxFileName);
                    SaveOnnxModel(onnxModel, onnxModelPath, null);
                    // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                    string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                    var onnxTransformer = onnxEstimator.Fit(dataView);
                    var onnxResult = onnxTransformer.Transform(dataView);
                    CompareSelectedR4ScalarColumns(transformedData.Schema[5].Name, outputNames[3], transformedData, onnxResult, 3); //compare scores
                    CompareSelectedScalarColumns<Boolean>(transformedData.Schema[4].Name, outputNames[2], transformedData, onnxResult); //compare predicted labels
                }
            }
            Done();
        }

        [Fact]
        public void TestVectorWhiteningOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);
            string dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataView = mlContext.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.Single, 0, 10)
            }, hasHeader: true, separatorChar: ';');

            var pipeline = new VectorWhiteningEstimator(mlContext, "whitened1", "features")
                .Append(new VectorWhiteningEstimator(mlContext, "whitened2", "features", kind: WhiteningKind.PrincipalComponentAnalysis, rank: 5));
            var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);

            // Compare model scores produced by ML.NET and ONNX's runtime.
            if (IsOnnxRuntimeSupported())
            {
                var onnxFileName = $"VectorWhitening.onnx";
                var onnxModelPath = GetOutputPath(onnxFileName);
                SaveOnnxModel(onnxModel, onnxModelPath, null);
                // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                var onnxTransformer = onnxEstimator.Fit(dataView);
                var onnxResult = onnxTransformer.Transform(dataView);
                CompareSelectedR4VectorColumns(transformedData.Schema[2].Name, outputNames[2], transformedData, onnxResult); // whitened1
                CompareSelectedR4VectorColumns(transformedData.Schema[3].Name, outputNames[3], transformedData, onnxResult); // whitened2
            }
            Done();
        }

        [Fact]
        public void PlattCalibratorOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);
            string dataPath = GetDataPath("breast-cancer.txt");
            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var dataView = mlContext.Data.LoadFromTextFile<BreastCancerBinaryClassification>(dataPath, separatorChar: '\t', hasHeader: true);
            List<IEstimator<ITransformer>> estimators = new List<IEstimator<ITransformer>>()
            {
                mlContext.BinaryClassification.Trainers.AveragedPerceptron(),
                mlContext.BinaryClassification.Trainers.FastForest(),
                mlContext.BinaryClassification.Trainers.FastTree(),
                mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(),
                mlContext.BinaryClassification.Trainers.LinearSvm(),
                mlContext.BinaryClassification.Trainers.Prior(),
                mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(),
                mlContext.BinaryClassification.Trainers.SdcaNonCalibrated(),
                mlContext.BinaryClassification.Trainers.SgdCalibrated(),
                mlContext.BinaryClassification.Trainers.SgdNonCalibrated(),
                mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression(),
            };
            if (Environment.Is64BitProcess)
            {
                estimators.Add(mlContext.BinaryClassification.Trainers.LightGbm());
            }

            var initialPipeline = mlContext.Transforms.ReplaceMissingValues("Features").
                Append(mlContext.Transforms.NormalizeMinMax("Features"));
            foreach (var estimator in estimators)
            {
                var pipeline = initialPipeline.Append(estimator).Append(mlContext.BinaryClassification.Calibrators.Platt());
                var model = pipeline.Fit(dataView);
                var outputSchema = model.GetOutputSchema(dataView.Schema);
                var transformedData = model.Transform(dataView);
                var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);

                // Compare model scores produced by ML.NET and ONNX's runtime. 
                if (IsOnnxRuntimeSupported())
                {
                    var onnxFileName = $"{estimator.ToString()}-WithPlattCalibrator.onnx";
                    var onnxModelPath = GetOutputPath(onnxFileName);
                    SaveOnnxModel(onnxModel, onnxModelPath, null);
                    // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                    string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                    var onnxTransformer = onnxEstimator.Fit(dataView);
                    var onnxResult = onnxTransformer.Transform(dataView);
                    CompareSelectedR4ScalarColumns(transformedData.Schema[5].Name, outputNames[3], transformedData, onnxResult, 3); //compare scores
                    CompareSelectedScalarColumns<Boolean>(transformedData.Schema[4].Name, outputNames[2], transformedData, onnxResult); //compare predicted labels
                    CompareSelectedR4ScalarColumns(transformedData.Schema.Last().Name, outputNames.Last(), transformedData, onnxResult, 3); //compare probabilities
                }
            }
            Done();
        }

        class PlattModelInput
        {
            public bool Label { get; set; }
            public float Score { get; set; }
        }

        static IEnumerable<PlattModelInput> PlattGetData()
        {
            for (int i = 0; i < 100; i++)
            {
                yield return new PlattModelInput { Score = i, Label = i % 2 == 0 };
            }
        }

        [Fact]
        public void PlattCalibratorOnnxConversionTest2()
        {
            // Test PlattCalibrator without any binary prediction trainer
            var mlContext = new MLContext(seed: 0);

            IDataView data = mlContext.Data.LoadFromEnumerable(PlattGetData());

            var calibratorEstimator = mlContext.BinaryClassification.Calibrators
                .Platt();

            var calibratorTransformer = calibratorEstimator.Fit(data);
            var transformedData = calibratorTransformer.Transform(data);
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(calibratorTransformer, data);

            // Compare model scores produced by ML.NET and ONNX's runtime.
            if (IsOnnxRuntimeSupported())
            {
                var onnxFileName = $"{calibratorTransformer.ToString()}.onnx";
                var onnxModelPath = GetOutputPath(onnxFileName);
                SaveOnnxModel(onnxModel, onnxModelPath, null);

                // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                var onnxTransformer = onnxEstimator.Fit(data);
                var onnxResult = onnxTransformer.Transform(data);
                CompareSelectedR4ScalarColumns(transformedData.Schema.Last().Name, outputNames.Last(), transformedData, onnxResult, 3); //compare probabilities
            }
            Done();
        }

        private class DataPoint
        {
            [VectorType(3)]
            public float[] Features { get; set; }
        }

        [Theory]
        [CombinatorialData]
        public void LpNormOnnxConversionTest(
            bool ensureZeroMean,
            LpNormNormalizingEstimatorBase.NormFunction norm)
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

            Done();
        }

        [Fact]
        public void CommandLineOnnxConversionTest()
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
        //Skipping test temporarily. This test will be re-enabled once the cause of failures has been determined
        [Trait("Category", "SkipInCI")]
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

        [Theory]
        // These are the supported conversions
        // ML.NET does not allow any conversions between signed and unsigned numeric types
        // Onnx does not seem to support casting a string to any type
        // Though the onnx docs claim support for byte and sbyte, 
        // CreateNamedOnnxValue in OnnxUtils.cs throws a NotImplementedException for those two
        [InlineData(DataKind.Int16, DataKind.Int16)]
        [InlineData(DataKind.Int16, DataKind.Int32)]
        [InlineData(DataKind.Int16, DataKind.Int64)]
        [InlineData(DataKind.Int16, DataKind.Single)]
        [InlineData(DataKind.Int16, DataKind.Double)]
        [InlineData(DataKind.UInt16, DataKind.UInt16)]
        [InlineData(DataKind.UInt16, DataKind.UInt32)]
        [InlineData(DataKind.UInt16, DataKind.UInt64)]
        [InlineData(DataKind.UInt16, DataKind.Single)]
        [InlineData(DataKind.UInt16, DataKind.Double)]
        [InlineData(DataKind.Int32, DataKind.Int16)]
        [InlineData(DataKind.Int32, DataKind.Int32)]
        [InlineData(DataKind.Int32, DataKind.Int64)]
        [InlineData(DataKind.Int32, DataKind.Single)]
        [InlineData(DataKind.Int32, DataKind.Double)]
        [InlineData(DataKind.Int64, DataKind.Int16)]
        [InlineData(DataKind.Int64, DataKind.Int32)]
        [InlineData(DataKind.Int64, DataKind.Int64)]
        [InlineData(DataKind.Int64, DataKind.Single)]
        [InlineData(DataKind.Int64, DataKind.Double)]
        [InlineData(DataKind.UInt64, DataKind.UInt16)]
        [InlineData(DataKind.UInt64, DataKind.UInt32)]
        [InlineData(DataKind.UInt64, DataKind.UInt64)]
        [InlineData(DataKind.UInt64, DataKind.Single)]
        [InlineData(DataKind.UInt64, DataKind.Double)]
        [InlineData(DataKind.Single, DataKind.Single)]
        [InlineData(DataKind.Single, DataKind.Double)]
        [InlineData(DataKind.Double, DataKind.Single)]
        [InlineData(DataKind.Double, DataKind.Double)]
        public void OnnxTypeConversionTest(DataKind fromKind, DataKind toKind)
        {
            var mlContext = new MLContext(seed: 1);
            string filePath = GetDataPath("type-conversion.txt");

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
        public void IndicateMissingValuesOnnxConversionTest()
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

        [Theory]
        [InlineData(DataKind.Single)]
        [InlineData(DataKind.String)]
        public void ValueToKeyMappingOnnxConversionTest(DataKind valueType)
        {
            var mlContext = new MLContext(seed: 1);
            string filePath = GetDataPath("type-conversion.txt");

            TextLoader.Column[] columns = new[]
            {
                new TextLoader.Column("Value", valueType, 0, 0)
            };
            var dataView = mlContext.Data.LoadFromTextFile(filePath, columns);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Key", "Value");
            var model = pipeline.Fit(dataView);
            var mlnetResult = model.Transform(dataView);

            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);
            var onnxFileName = "ValueToKey.onnx";
            var onnxModelPath = GetOutputPath(onnxFileName);
            SaveOnnxModel(onnxModel, onnxModelPath, null);

            if (IsOnnxRuntimeSupported())
            {
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                var onnxTransformer = onnxEstimator.Fit(dataView);
                var onnxResult = onnxTransformer.Transform(dataView);

                CompareSelectedVectorColumns<UInt32>(model.ColumnPairs[0].outputColumnName, outputNames[1], mlnetResult, onnxResult);
            }

            Done();
        }

        private class TextData
        {
            public string Text { get; set; }
        }

        [Fact]
        public void WordTokenizerOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);

            var samples = new List<TextData>()
            {
                new TextData(){ Text = "cat sat on mat" },
                new TextData(){ Text = "mat not fit cat" },
                new TextData(){ Text = "cat think mat bad" },
            };

            var dataView = mlContext.Data.LoadFromEnumerable(samples);

            var pipe = mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text", new[] { ' ' });

            var model = pipe.Fit(dataView);
            var transformedData = model.Transform(dataView);

            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);
            var onnxFilename = "Tokenizer.onnx";
            var onnxFilePath = GetOutputPath(onnxFilename);
            SaveOnnxModel(onnxModel, onnxFilePath, null);
            if (IsOnnxRuntimeSupported())
            {
                // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxFilePath);
                var onnxTransformer = onnxEstimator.Fit(dataView);
                var onnxResult = onnxTransformer.Transform(dataView);
                CompareSelectedVectorColumns<ReadOnlyMemory<char>>(transformedData.Schema[1].Name, outputNames[1], transformedData, onnxResult);
            }

            Done();
        }

        [Theory]
        [CombinatorialData]
        public void NgramOnnxConnversionTest(
            [CombinatorialValues(1, 2, 3)] int ngramLength,
            bool useAllLength,
            NgramExtractingEstimator.WeightingCriteria weighting)
        {
            var mlContext = new MLContext(seed: 1);

            var samples = new List<TextData>()
            {
                new TextData(){ Text = "cat sat on mat" },
                new TextData(){ Text = "mat not fit cat" },
                new TextData(){ Text = "cat think mat bad" },
            };

            // Convert training data to IDataView.
            var dataView = mlContext.Data.LoadFromEnumerable(samples);

            IEstimator<ITransformer>[] pipelines =
            {
                mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text", new[] { ' ' })
                                .Append(mlContext.Transforms.Conversion.MapValueToKey("Tokens"))
                                .Append(mlContext.Transforms.Text.ProduceNgrams("NGrams", "Tokens",
                                            ngramLength: ngramLength,
                                            useAllLengths: useAllLength,
                                            weighting: weighting)),

                mlContext.Transforms.Text.ProduceWordBags("Tokens", "Text",
                                        ngramLength: ngramLength,
                                        useAllLengths: useAllLength,
                                        weighting: weighting)
            };

            for (int i = 0; i < pipelines.Length; i++)
            {
                var pipe = pipelines[i];
                var model = pipe.Fit(dataView);
                var transformedData = model.Transform(dataView);

                var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);
                var onnxFilename = $"Ngram-{i}-{ngramLength}-{useAllLength}-{weighting}.onnx";
                var onnxFilePath = GetOutputPath(onnxFilename);
                SaveOnnxModel(onnxModel, onnxFilePath, null);
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && Environment.Is64BitProcess)
                {
                    // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                    string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxFilePath);
                    var onnxTransformer = onnxEstimator.Fit(dataView);
                    var onnxResult = onnxTransformer.Transform(dataView);
                    CompareSelectedR4VectorColumns(transformedData.Schema[3].Name, outputNames[outputNames.Length-1], transformedData, onnxResult, 3);
                }
            }

            Done();
        }

        [Fact]
        public void OptionalColumnOnnxTest()
        {
            var mlContext = new MLContext(seed: 1);

            var samples = new List<BreastCancerCatFeatureExample>()
            {
                new BreastCancerCatFeatureExample() { Label = false, F1 = 0.0f, F2 = "F2"},
                new BreastCancerCatFeatureExample() { Label = true, F1 = 0.1f, F2 = "F2"},
            };
            IHostEnvironment env = mlContext as IHostEnvironment;
            var dataView = mlContext.Data.LoadFromEnumerable(samples);
            var args = new OptionalColumnTransform.Arguments { Columns = new[] { "F1" }, Data = dataView };
            var transform = OptionalColumnTransform.MakeOptional(env, args);

            var ctx = new OnnxContextImpl(mlContext, "model", "ML.NET", "0", 0, "machinelearning.dotnet", OnnxVersion.Stable);
            var outputData = transform.OutputData;
            LinkedList<ITransformCanSaveOnnx> transforms = null;
            ModelProto onnxModel;
            using (var ch = env.Start("ONNX conversion"))
            {
                SaveOnnxCommand.GetPipe(ctx, ch, outputData, out IDataView root, out IDataView sink, out transforms);
                onnxModel = SaveOnnxCommand.ConvertTransformListToOnnxModel(ctx, ch, root, sink, transforms, null, null);
            }

            var onnxFileName = "optionalcol.onnx";
            var onnxModelPath = GetOutputPath(onnxFileName);
            var onnxTextFileName = "optionalcol.txt";
            var onnxTextPath = GetOutputPath(onnxTextFileName);

            SaveOnnxModel(onnxModel, onnxModelPath, onnxTextPath);
            if (IsOnnxRuntimeSupported())
            {
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                var onnxTransformer = onnxEstimator.Fit(dataView);
                var onnxResult = onnxTransformer.Transform(dataView);
                CompareSelectedR4ScalarColumns(transform.Model.OutputSchema[2].Name, outputNames[1], outputData, onnxResult);
            }
            Done();
        }

        [Fact]
        public void KeyToValueOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath("breast-cancer.txt");
            var dataView = mlContext.Data.LoadFromTextFile<BreastCancerMulticlassExample>(dataPath,
                separatorChar: '\t',
                hasHeader: true);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("LabelKey", "Label").
                Append(mlContext.Transforms.Conversion.MapKeyToValue("LabelValue", "LabelKey"));

            var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);

            var onnxFileName = "KeyToValue.onnx";
            var onnxModelPath = GetOutputPath(onnxFileName);

            SaveOnnxModel(onnxModel, onnxModelPath, null);

            if (IsOnnxRuntimeSupported())
            {
                // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                var onnxTransformer = onnxEstimator.Fit(dataView);
                var onnxResult = onnxTransformer.Transform(dataView);
                CompareSelectedScalarColumns<ReadOnlyMemory<Char>>(transformedData.Schema[3].Name, outputNames[3], transformedData, onnxResult);
            }

            Done();
        }

        [Fact]
        public void MulticlassTrainersOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath("breast-cancer.txt");
            var dataView = mlContext.Data.LoadFromTextFile<BreastCancerMulticlassExample>(dataPath, separatorChar: '\t', hasHeader: true);

            List<IEstimator<ITransformer>> estimators = new List<IEstimator<ITransformer>>()
            {
                mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(),
                mlContext.MulticlassClassification.Trainers.NaiveBayes(),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(), useProbabilities:false),
                mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(),
                mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated()
            };

            if (Environment.Is64BitProcess)
            {
                estimators.Add(mlContext.MulticlassClassification.Trainers.LightGbm());
                estimators.Add(mlContext.MulticlassClassification.Trainers.LightGbm(
                    new LightGbmMulticlassTrainer.Options { UseSoftmax = true }));
            }

            var initialPipeline = mlContext.Transforms.ReplaceMissingValues("Features")
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"));

            foreach (var estimator in estimators)
            {
                var pipeline = initialPipeline.Append(estimator);
                var model = pipeline.Fit(dataView);
                var transformedData = model.Transform(dataView);

                var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);
                var onnxFileName = $"{estimator.ToString()}.onnx";
                var onnxModelPath = GetOutputPath(onnxFileName);

                SaveOnnxModel(onnxModel, onnxModelPath, null);

                // Compare results produced by ML.NET and ONNX's runtime.
                if (IsOnnxRuntimeSupported())
                {
                    // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                    string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                    var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                    var onnxTransformer = onnxEstimator.Fit(dataView);
                    var onnxResult = onnxTransformer.Transform(dataView);
                    CompareSelectedScalarColumns<uint>(transformedData.Schema[5].Name, outputNames[2], transformedData, onnxResult);
                }
            }
            Done();
        }

        [Fact]
        public void CopyColumnsOnnxTest()
        {
            var mlContext = new MLContext(seed: 1);

            var trainDataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataView = mlContext.Data.LoadFromTextFile<AdultData>(trainDataPath,
                separatorChar: ';',
                hasHeader: true);

            var pipeline = mlContext.Transforms.CopyColumns("Target1", "Target");
            var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);

            var onnxFileName = "copycolumns.onnx";
            var onnxModelPath = GetOutputPath(onnxFileName);

            SaveOnnxModel(onnxModel, onnxModelPath, null);

            if (IsOnnxRuntimeSupported())
            {
                // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                var onnxTransformer = onnxEstimator.Fit(dataView);
                var onnxResult = onnxTransformer.Transform(dataView);
                CompareSelectedR4ScalarColumns(model.ColumnPairs[0].outputColumnName, outputNames[2], transformedData, onnxResult);
            }
            Done();
        }

        [Fact]
        public void FeatureSelectionOnnxTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath("breast-cancer.txt");

            var dataView = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("ScalarFloat", DataKind.Single, 6),
                new TextLoader.Column("VectorFloat", DataKind.Single, 1, 4),
                new TextLoader.Column("VectorDouble", DataKind.Double, 4, 8),
                new TextLoader.Column("Label", DataKind.Boolean, 0)
            });

            var columns = new[] {
                new CountFeatureSelectingEstimator.ColumnOptions("FeatureSelectDouble", "VectorDouble", count: 1),
                new CountFeatureSelectingEstimator.ColumnOptions("ScalFeatureSelectMissing690", "ScalarFloat", count: 690),
                new CountFeatureSelectingEstimator.ColumnOptions("ScalFeatureSelectMissing100", "ScalarFloat", count: 100),
                new CountFeatureSelectingEstimator.ColumnOptions("VecFeatureSelectMissing690", "VectorDouble", count: 690),
                new CountFeatureSelectingEstimator.ColumnOptions("VecFeatureSelectMissing100", "VectorDouble", count: 100)
            };
            var pipeline = ML.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("FeatureSelect", "VectorFloat", count: 1)
                .Append(ML.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(columns))
                .Append(ML.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("FeatureSelectMIScalarFloat", "ScalarFloat"))
                .Append(ML.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("FeatureSelectMIVectorFloat", "VectorFloat"));

            var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);

            var onnxFileName = "countfeatures.onnx";
            var onnxModelPath = GetOutputPath(onnxFileName);

            SaveOnnxModel(onnxModel, onnxModelPath, null);

            if (IsOnnxRuntimeSupported())
            {
                // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                var onnxTransformer = onnxEstimator.Fit(dataView);
                var onnxResult = onnxTransformer.Transform(dataView);
                CompareSelectedR4ScalarColumns("FeatureSelectMIScalarFloat", "FeatureSelectMIScalarFloat0", transformedData, onnxResult);
                CompareSelectedR4VectorColumns("FeatureSelectMIVectorFloat", "FeatureSelectMIVectorFloat0", transformedData, onnxResult);
                CompareSelectedR4ScalarColumns("ScalFeatureSelectMissing690", "ScalFeatureSelectMissing6900", transformedData, onnxResult);
                CompareSelectedR8VectorColumns("VecFeatureSelectMissing690", "VecFeatureSelectMissing6900", transformedData, onnxResult);
            }
            Done();
        }



        [Fact]
        public void SelectColumnsOnnxTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath("breast-cancer.txt");

            var dataView = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("Label", DataKind.Boolean, 0),
                new TextLoader.Column("Thickness", DataKind.Int32, 1),
                new TextLoader.Column("Size", DataKind.Int32, 2),
                new TextLoader.Column("Shape", DataKind.Int32, 3),
                new TextLoader.Column("Adhesion", DataKind.Int32, 4),
                new TextLoader.Column("EpithelialSize", DataKind.Int32, 5),
                new TextLoader.Column("BlandChromatin", DataKind.Int32, 7),
                new TextLoader.Column("NormalNucleoli", DataKind.Int32, 8),
                new TextLoader.Column("Mitoses", DataKind.Int32, 9),
            });

            var pipeline = mlContext.Transforms.SelectColumns(new[] { "Size", "Shape", "Thickness", "Label" });

            var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);
            var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);

            var onnxFileName = "selectcolumns.onnx";
            var onnxModelPath = GetOutputPath(onnxFileName);

            SaveOnnxModel(onnxModel, onnxModelPath, null);

            if (IsOnnxRuntimeSupported())
            {
                // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath);
                var onnxTransformer = onnxEstimator.Fit(dataView);
                var onnxResult = onnxTransformer.Transform(dataView);

                // Verify that onnx output has only the four columns we selected from the input
                Assert.Equal(4, outputNames.Length);
                Assert.Equal("Size1", outputNames[0]);
                Assert.Equal("Shape1", outputNames[1]);
                Assert.Equal("Thickness1", outputNames[2]);
                Assert.Equal("Label1", outputNames[3]);

                CompareSelectedScalarColumns<int>("Size", "Size1", transformedData, onnxResult);
                CompareSelectedScalarColumns<int>("Shape", "Shape1", transformedData, onnxResult);
                CompareSelectedScalarColumns<int>("Thickness", "Thickness1", transformedData, onnxResult);
                CompareSelectedScalarColumns<bool>("Label", "Label1", transformedData, onnxResult);
            }

            onnxFileName = "SelectColumns.txt";
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "Transforms");
            var onnxTextModelPath = GetOutputPath(subDir, onnxFileName);
            SaveOnnxModel(onnxModel, null, onnxTextModelPath);
            CheckEquality(subDir, onnxFileName, digitsOfPrecision: 1);

            Done();
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
                        if (typeof(T) == typeof(ReadOnlyMemory<char>))
                            Assert.Equal(expected.GetItemOrDefault(i).ToString(), actual.GetItemOrDefault(i).ToString());
                        else
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
                    CompareNumbersWithTolerance(expected, actual.GetItemOrDefault(0), null, precision);
                }
            }
        }

        private void CompareSelectedScalarColumns<T>(string leftColumnName, string rightColumnName, IDataView left, IDataView right)
        {
            var leftColumn = left.Schema[leftColumnName];
            var rightColumn = right.Schema[rightColumnName];

            using (var expectedCursor = left.GetRowCursor(leftColumn))
            using (var actualCursor = right.GetRowCursor(rightColumn))
            {
                T expected = default;
                VBuffer<T> actual = default;
                var expectedGetter = expectedCursor.GetGetter<T>(leftColumn);
                var actualGetter = actualCursor.GetGetter<VBuffer<T>>(rightColumn);
                while (expectedCursor.MoveNext() && actualCursor.MoveNext())
                {
                    expectedGetter(ref expected);
                    actualGetter(ref actual);
                    var actualVal = actual.GetItemOrDefault(0);

                    Assert.Equal(1, actual.Length);

                    if (typeof(T) == typeof(ReadOnlyMemory<Char>))
                        Assert.Equal(expected.ToString(), actualVal.ToString());
                    else
                        Assert.Equal(expected, actualVal);
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
