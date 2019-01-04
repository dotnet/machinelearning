// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Google.Protobuf;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms;
using Microsoft.ML.UniversalModelFormat.Onnx;
using Newtonsoft.Json;
using Xunit;
using Xunit.Abstractions;

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

        /// <summary>
        /// In this test, we convert a trained <see cref="TransformerChain"/> into ONNX <see cref="UniversalModelFormat.Onnx.ModelProto"/> file and then
        /// call <see cref="OnnxScoringEstimator"/> to evaluate that file. The outputs of <see cref="OnnxScoringEstimator"/> are checked against the original
        /// ML.NET model's outputs.
        /// </summary>
        [Fact]
        public void SimpleEndToEndOnnxConversionTest()
        {
            // Step 1: Create and train a ML.NET pipeline.
            var trainDataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var mlContext = new MLContext(seed: 1, conc: 1);
            var data = mlContext.Data.ReadFromTextFile<AdultData>(trainDataPath,
                hasHeader: true,
                separatorChar: ';'
            );
            var cachedTrainData = mlContext.Data.Cache(data);
            var dynamicPipeline =
                mlContext.Transforms.Normalize("FeatureVector")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.StochasticDualCoordinateAscent(labelColumn: "Target", featureColumn: "FeatureVector"));
            var model = dynamicPipeline.Fit(data);
            var transformedData = model.Transform(data);

            // Step 2: Convert ML.NET model to ONNX format and save it as a file.
            var onnxModel = mlContext.Model.ConvertToOnnx(model, data);
            var onnxFileName = "model.onnx";
            var onnxModelPath = GetOutputPath(onnxFileName);
            SaveOnnxModel(onnxModel, onnxModelPath, null);

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && Environment.Is64BitProcess)
            {
                // Step 3: Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = new OnnxScoringEstimator(mlContext, onnxModelPath, inputNames, outputNames);
                var onnxTransformer = onnxEstimator.Fit(data);
                var onnxResult = onnxTransformer.Transform(data);

                // Step 4: Compare ONNX and ML.NET results.
                CompareSelectedR4ScalarColumns("Score", "Score0", transformedData, onnxResult, 2);
            }

            // Step 5: Check ONNX model's text format. This test will be not necessary if Step 3 and Step 4 can run on Linux and
            // Mac to support cross-platform tests.
            var subDir = Path.Combine("..", "..", "BaselineOutput", "Common", "Onnx", "Regression", "Adult");
            var onnxTextName = "SimplePipeline.txt";
            var onnxTextPath = GetOutputPath(subDir, onnxTextName);
            SaveOnnxModel(onnxModel, null, onnxTextPath);
            CheckEquality(subDir, onnxTextName);

            Done();
        }

        private class BreastCancerFeatureVector
        {
            [LoadColumn(1, 9), VectorType(9)]
            public float[] Features;
        }

        [Fact]
        public void KmeansOnnxConversionTest()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1, conc: 1);

            string dataPath = GetDataPath("breast-cancer.txt");
            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var data = mlContext.Data.ReadFromTextFile<BreastCancerFeatureVector>(dataPath,
                hasHeader: true,
                separatorChar: '\t');

            var pipeline = mlContext.Transforms.Normalize("Features").
                Append(mlContext.Clustering.Trainers.KMeans(features: "Features", advancedSettings: settings =>
                {
                    settings.MaxIterations = 1;
                    settings.K = 4;
                    settings.NumThreads = 1;
                    settings.InitAlgorithm = Trainers.KMeans.KMeansPlusPlusTrainer.InitAlgorithm.KMeansPlusPlus;
                }));

            var model = pipeline.Fit(data);
            var transformedData = model.Transform(data);

            var onnxModel = mlContext.Model.ConvertToOnnx(model, data);

            // Compare results produced by ML.NET and ONNX's runtime.
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && Environment.Is64BitProcess)
            {
                var onnxFileName = "model.onnx";
                var onnxModelPath = GetOutputPath(onnxFileName);
                SaveOnnxModel(onnxModel, onnxModelPath, null);

                // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = new OnnxScoringEstimator(mlContext, onnxModelPath, inputNames, outputNames);
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
            CheckEquality(subDir, onnxTextName);
            Done();
        }

        private void CreateDummyExamplesToMakeComplierHappy()
        {
            var dummyExample = new BreastCancerFeatureVector() { Features = null };
        }

        private void CompareSelectedR4VectorColumns(string leftColumnName, string rightColumnName, IDataView left, IDataView right, int precision = 6)
        {
            var leftColumnIndex = left.Schema[leftColumnName].Index;
            var rightColumnIndex = right.Schema[rightColumnName].Index;

            using (var expectedCursor = left.GetRowCursor(columnIndex => leftColumnIndex == columnIndex))
            using (var actualCursor = right.GetRowCursor(columnIndex => rightColumnIndex == columnIndex))
            {
                VBuffer<float> expected = default;
                VBuffer<float> actual = default;
                var expectedGetter = expectedCursor.GetGetter<VBuffer<float>>(leftColumnIndex);
                var actualGetter = actualCursor.GetGetter<VBuffer<float>>(rightColumnIndex);
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

        private void CompareSelectedR4ScalarColumns(string leftColumnName, string rightColumnName, IDataView left, IDataView right, int precision = 6)
        {
            var leftColumnIndex = left.Schema[leftColumnName].Index;
            var rightColumnIndex = right.Schema[rightColumnName].Index;

            using (var expectedCursor = left.GetRowCursor(columnIndex => leftColumnIndex == columnIndex))
            using (var actualCursor = right.GetRowCursor(columnIndex => rightColumnIndex == columnIndex))
            {
                float expected = default;
                VBuffer<float> actual = default;
                var expectedGetter = expectedCursor.GetGetter<float>(leftColumnIndex);
                var actualGetter = actualCursor.GetGetter<VBuffer<float>>(rightColumnIndex);
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
                fileText = Regex.Replace(fileText, "\"producerVersion\": \"([^\"]+)\"", "\"producerVersion\": \"##VERSION##\"");
                File.WriteAllText(textFormatPath, fileText);
            }
        }
    }
}
