// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
using Microsoft.ML.TestFrameworkCommon.Utility;
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
        // These two members are meant to be changed
        // Only when manually testing the Onnx GPU nuggets
        private const bool _fallbackToCpu = true;
        private static int? _gpuDeviceId = null;

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
                .Append(mlContext.Regression.Trainers.Sdca(new SdcaRegressionTrainer.Options()
                {
                    LabelColumnName = "Target",
                    FeatureColumnName = "FeatureVector",
                    NumberOfThreads = 1
                }));

            var onnxFileName = "model.onnx";
            var subDir = Path.Combine("Onnx", "Regression", "Adult");
            var onnxTextName = "SimplePipeline.txt";

            // Step 2: Convert ML.NET model to ONNX format and save it as a model file and a text file.
            TestPipeline(dynamicPipeline, cachedTrainData, onnxFileName, new ColumnComparison[] { new ColumnComparison("Score", 1) }, onnxTextName, subDir);

            // Step 3: Check ONNX model's text format. This test will be not necessary if Step 2 can run on Linux and
            // Mac to support cross-platform tests.

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

            [LoadColumn(3, 7), VectorType(6)]
            public string[] F3;
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

        [Fact]
        public void KmeansOnnxConversionTest()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var data = mlContext.Data.LoadFromTextFile<BreastCancerFeatureVector>(dataPath,
                separatorChar: '\t',
                hasHeader: false);

            var pipeline = mlContext.Transforms.NormalizeMinMax("Features").
                Append(mlContext.Clustering.Trainers.KMeans(new Trainers.KMeansTrainer.Options
                {
                    FeatureColumnName = DefaultColumnNames.Features,
                    MaximumNumberOfIterations = 1,
                    NumberOfClusters = 4,
                    NumberOfThreads = 1,
                    InitializationAlgorithm = Trainers.KMeansTrainer.InitializationAlgorithm.Random
                }));


            var onnxFileName = "model.onnx";
            var subDir = Path.Combine("Onnx", "Cluster", "BreastCancer");
            var onnxTextName = "Kmeans.txt";

            // Step 2: Convert ML.NET model to ONNX format and save it as a model file and a text file.
            TestPipeline(pipeline, data, onnxFileName, new ColumnComparison[] { new ColumnComparison("Score", 3) }, onnxTextName, subDir);

            CheckEquality(subDir, onnxTextName, digitsOfPrecision: 2);
            Done();
        }

        [Fact]
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
                // TODO TEST_STABILITY: Sdca has developed some instability with failures in comparison against baseline. Disabling it for now.
                //mlContext.Regression.Trainers.Sdca("Target","FeatureVector"),
                mlContext.Regression.Trainers.OnlineGradientDescent("Target","FeatureVector"),
                mlContext.Regression.Trainers.FastForest("Target", "FeatureVector"),
                mlContext.Regression.Trainers.FastTree("Target", "FeatureVector"),
                mlContext.Regression.Trainers.FastTreeTweedie("Target", "FeatureVector"),
                mlContext.Regression.Trainers.LbfgsPoissonRegression("Target", "FeatureVector"),
            };
            if (NativeLibrary.NativeLibraryExists("MklImports"))
            {
                estimators.Add(mlContext.Regression.Trainers.Ols("Target", "FeatureVector"));
            }
            if (Environment.Is64BitProcess && NativeLibrary.NativeLibraryExists("lib_lightgbm"))
            {
                estimators.Add(mlContext.Regression.Trainers.LightGbm("Target", "FeatureVector"));
            }
            foreach (var estimator in estimators)
            {
                var onnxModelFileName = $"{estimator}.onnx";
                var onnxTxtFileName = $"{estimator}.txt";
                var subDir = Path.Combine("Onnx", "Regression", "Adult");

                // Step 2: Convert ML.NET model to ONNX format and save it as a model file and a text file.
                TestPipeline(estimator, dataView, onnxModelFileName, new ColumnComparison[] { new ColumnComparison("Score", 3) }, onnxTxtFileName, subDir);
                CheckEquality(subDir, onnxTxtFileName, digitsOfPrecision: 1);
            }
            Done();
        }

        [Fact]
        public void BinaryClassificationTrainersOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var dataView = mlContext.Data.LoadFromTextFile<BreastCancerBinaryClassification>(dataPath, separatorChar: '\t', hasHeader: false);
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
            };
            if (NativeLibrary.NativeLibraryExists("MklImports"))
            {
                estimators.Add(mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression());
            }
            if (Environment.Is64BitProcess && NativeLibrary.NativeLibraryExists("lib_lightgbm"))
            {
                estimators.Add(mlContext.BinaryClassification.Trainers.LightGbm());
            }

            var initialPipeline = mlContext.Transforms.ReplaceMissingValues("Features").
                Append(mlContext.Transforms.NormalizeMinMax("Features"));
            foreach (var estimator in estimators)
            {
                var pipeline = initialPipeline.Append(estimator);
                var onnxFileName = $"{estimator}.onnx";

                TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("Score", 3), new ColumnComparison("PredictedLabel") });
            }
            Done();
        }

        [NativeDependencyFact("MklImports")]
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

            var onnxFileName = $"VectorWhitening.onnx";
            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("whitened1"), new ColumnComparison("whitened2") });

            Done();
        }

        private (IDataView, List<IEstimator<ITransformer>>, EstimatorChain<NormalizingTransformer>) GetEstimatorsForOnnxConversionTests()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var dataView = ML.Data.LoadFromTextFile<BreastCancerBinaryClassification>(dataPath, separatorChar: '\t', hasHeader: true);
            List<IEstimator<ITransformer>> estimators = new List<IEstimator<ITransformer>>()
            {
                ML.BinaryClassification.Trainers.AveragedPerceptron(),
                ML.BinaryClassification.Trainers.FastForest(),
                ML.BinaryClassification.Trainers.FastTree(),
                ML.BinaryClassification.Trainers.LbfgsLogisticRegression(),
                ML.BinaryClassification.Trainers.LinearSvm(),
                ML.BinaryClassification.Trainers.Prior(),
                ML.BinaryClassification.Trainers.SdcaLogisticRegression(),
                ML.BinaryClassification.Trainers.SdcaNonCalibrated(),
                ML.BinaryClassification.Trainers.SgdCalibrated(),
                ML.BinaryClassification.Trainers.SgdNonCalibrated(),
                ML.BinaryClassification.Trainers.SymbolicSgdLogisticRegression(),
            };
            if (Environment.Is64BitProcess)
            {
                estimators.Add(ML.BinaryClassification.Trainers.LightGbm());
            }

            var initialPipeline = ML.Transforms.ReplaceMissingValues("Features").
                Append(ML.Transforms.NormalizeMinMax("Features"));
            return (dataView, estimators, initialPipeline);
        }

        private void CommonCalibratorOnnxConversionTest(IEstimator<ITransformer> calibrator, IEstimator<ITransformer> calibratorNonStandard)
        {
            // Initialize variables needed for the ONNX conversion test
            var (dataView, estimators, initialPipeline) = GetEstimatorsForOnnxConversionTests();

            // Step 1: Test calibrator with binary prediction trainer
            foreach (var estimator in estimators)
            {
                var pipelineEstimators = initialPipeline.Append(estimator).Append(calibrator);
                var onnxFileName = $"{estimator}-With-{calibrator}.onnx";
                TestPipeline(pipelineEstimators, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("Score", 3), new ColumnComparison("PredictedLabel"), new ColumnComparison("Probability", 3) });
            }

            // Step 2: Test calibrator without any binary prediction trainer
            IDataView dataSoloCalibrator = ML.Data.LoadFromEnumerable(GetCalibratorTestData());
            var onnxFileNameSoloCalibrator = $"{calibrator}-SoloCalibrator.onnx";
            TestPipeline(calibrator, dataSoloCalibrator, onnxFileNameSoloCalibrator, new ColumnComparison[] { new ColumnComparison("Probability", 3) });

            // Step 3: Test calibrator with a non-default Score column name and without any binary prediction trainer
            IDataView dataSoloCalibratorNonStandard = ML.Data.LoadFromEnumerable(GetCalibratorTestDataNonStandard());
            var onnxFileNameSoloCalibratorNonStandard = $"{calibratorNonStandard}-SoloCalibrator-NonStandard.onnx";
            TestPipeline(calibratorNonStandard, dataSoloCalibratorNonStandard, onnxFileNameSoloCalibratorNonStandard, new ColumnComparison[] { new ColumnComparison("Probability", 3) });

            Done();
        }

        [NativeDependencyFact("MklImports")]
        public void PlattCalibratorOnnxConversionTest()
        {
            CommonCalibratorOnnxConversionTest(ML.BinaryClassification.Calibrators.Platt(),
                ML.BinaryClassification.Calibrators.Platt(scoreColumnName: "ScoreX"));
        }

        [NativeDependencyFact("MklImports")]
        public void FixedPlattCalibratorOnnxConversionTest()
        {
            // Below, FixedPlattCalibrator is utilized by defining slope and offset in Platt's constructor with sample values.
            CommonCalibratorOnnxConversionTest(ML.BinaryClassification.Calibrators.Platt(slope: -1f, offset: -0.05f),
                ML.BinaryClassification.Calibrators.Platt(slope: -1f, offset: -0.05f, scoreColumnName: "ScoreX"));
        }

        [NativeDependencyFact("MklImports")]
        public void NaiveCalibratorOnnxConversionTest()
        {
            CommonCalibratorOnnxConversionTest(ML.BinaryClassification.Calibrators.Naive(),
                ML.BinaryClassification.Calibrators.Naive(scoreColumnName: "ScoreX"));
        }

        class CalibratorInput
        {
            public bool Label { get; set; }
            public float Score { get; set; }
        }

        class CalibratorInputNonStandard
        {
            public bool Label { get; set; }
            public float ScoreX { get; set; }
        }

        static IEnumerable<CalibratorInput> GetCalibratorTestData()
        {
            for (int i = 0; i < 100; i++)
            {
                yield return new CalibratorInput { Score = i, Label = i % 2 == 0 };
            }
        }

        static IEnumerable<CalibratorInputNonStandard> GetCalibratorTestDataNonStandard()
        {
            for (int i = 0; i < 100; i++)
            {
                yield return new CalibratorInputNonStandard { ScoreX = i, Label = i % 2 == 0 };
            }
        }

        [Fact]
        public void TextNormalizingOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);
            var dataPath = GetDataPath("wikipedia-detox-250-line-test.tsv");
            var dataView = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("label", DataKind.Boolean, 0),
                new TextLoader.Column("text", DataKind.String, 1)
            }, hasHeader: true);
            var pipeline = new TextNormalizingEstimator(mlContext, keepDiacritics: true, columns: new[] { ("NormText", "text") }).Append(
                new TextNormalizingEstimator(mlContext, keepDiacritics: true, caseMode: TextNormalizingEstimator.CaseMode.Upper, columns: new[] { ("UpperText", "text") })).Append(
                new TextNormalizingEstimator(mlContext, keepDiacritics: true, caseMode: TextNormalizingEstimator.CaseMode.None, columns: new[] { ("OriginalText", "text") }));
            var onnxFileName = $"TextNormalizing.onnx";

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("NormText"), new ColumnComparison("UpperText"), new ColumnComparison("OriginalText") });

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
            // Compare vector columns.
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

            var pipeline = mlContext.Transforms.NormalizeLpNorm(nameof(DataPoint.Features), norm: norm, ensureZeroMean: ensureZeroMean);
            var onnxFileName = $"LpNorm-{norm}-{ensureZeroMean}.onnx";

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("Features", 3) });

            Done();
        }

        [Fact]
        public void CommandLineOnnxConversionTest()
        {
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            string modelPath = GetOutputPath("ModelWithLessIO.zip");
            var trainingPathArgs = $"data={dataPath} out={modelPath}";
            var trainingArgs = " loader=text{col=Label:BL:0 col=F1:R4:1-8 col=F2:TX:9} xf=Cat{col=F2} xf=Concat{col=Features:F1,F2} tr=ft{numberOfThreads=1 numberOfLeaves=8 numberOfTrees=3} seed=1";
            Assert.Equal(0, Maml.Main(new[] { "train " + trainingPathArgs + trainingArgs }));

            var subDir = Path.Combine("Onnx", "BinaryClassification", "BreastCancer");
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

        [Theory]
        [CombinatorialData]
        public void KeyToVectorTest([CombinatorialValues(DataKind.Single, DataKind.Int64, DataKind.Int32, DataKind.Int16, DataKind.UInt64,
            DataKind.UInt32, DataKind.UInt16, DataKind.Double, DataKind.String, DataKind.Boolean)] DataKind valueType,
            OneHotEncodingEstimator.OutputKind outputKind)
        {
            var mlContext = new MLContext(seed: 1);
            string filePath = (valueType == DataKind.Boolean) ? GetDataPath("type-conversion-boolean.txt") : GetDataPath("type-conversion.txt");

            TextLoader.Column[] columnsVector = new[]
            {
                new TextLoader.Column("Key", valueType, 0, 3)
            };
            TextLoader.Column[] columnsScalar = new[]
            {
                new TextLoader.Column("Key", valueType, 0)
            };
            IDataView[] dataViews =
            {
                mlContext.Data.LoadFromTextFile(filePath, columnsScalar, separatorChar: '\t'), //scalar
                mlContext.Data.LoadFromTextFile(filePath, columnsVector , separatorChar: '\t') //vector
            };

            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("Vector", "Key", outputKind);

            for (int j = 0; j < dataViews.Length; j++)
            {
                if (OneHotEncodingEstimator.OutputKind.Binary == outputKind) break;
                var onnxFileName = "KeyToVector.onnx";
                var onnxTextName = "KeyToVector.txt";

                TestPipeline(pipeline, dataViews[j], onnxFileName, new ColumnComparison[] { new ColumnComparison("Vector") }, onnxTextName);
            }
            Done();
        }

        [Fact]
        public void InitializerCreationTest()
        {
            var env = new MLContext(1);
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
            var pipeline =
                mlContext.Transforms.NormalizeMinMax("FeatureVector")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.Sdca(new SdcaRegressionTrainer.Options()
                {
                    LabelColumnName = "Target",
                    FeatureColumnName = "FeatureVector",
                    NumberOfThreads = 1
                }));

            var onnxFileName = "LogisticRegressionSaveModelToOnnxTest.onnx";
            var onnxTextName = "LogisticRegressionSaveModelToOnnxTest.txt";
            var subDir = Path.Combine("Onnx", "BinaryClassification", "BreastCancer");

            TestPipeline(pipeline, cachedTrainData, onnxFileName, null, onnxTextName, subDir);

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
            var pipeline =
                mlContext.Transforms.NormalizeMinMax("FeatureVector")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.LightGbm(labelColumnName: "Target", featureColumnName: "FeatureVector", numberOfIterations: 3, numberOfLeaves: 16, minimumExampleCountPerLeaf: 100));

            var onnxFileName = "LightGbmBinaryClassificationOnnxConversionTest.onnx";
            var onnxTextName = "LightGbmBinaryClassificationOnnxConversionTest.txt";
            var subDir = Path.Combine("Onnx", "BinaryClassification", "BreastCancer");

            TestPipeline(pipeline, cachedTrainData, onnxFileName, null, onnxTextName, subDir);

            CheckEquality(subDir, onnxTextName, digitsOfPrecision: 3);
            Done();
        }

        [Fact]
        public void MulticlassLogisticRegressionOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var data = mlContext.Data.LoadFromTextFile<BreastCancerMulticlassExample>(dataPath,
                separatorChar: '\t',
                hasHeader: false);

            var pipeline = mlContext.Transforms.ReplaceMissingValues("Features").
                Append(mlContext.Transforms.NormalizeMinMax("Features")).
                Append(mlContext.Transforms.Conversion.MapValueToKey("Label")).
                Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(new LbfgsMaximumEntropyMulticlassTrainer.Options() { NumberOfThreads = 1 }));

            var onnxFileName = "MultiClassificationLogisticRegressionSaveModelToOnnxTest.onnx";
            var onnxTextName = "MultiClassificationLogisticRegressionSaveModelToOnnxTest.txt";
            var subDir = Path.Combine("Onnx", "MultiClassClassification", "BreastCancer");

            TestPipeline(pipeline, data, onnxFileName, new ColumnComparison[] { new ColumnComparison("PredictedLabel"), new ColumnComparison("Score") }, onnxTextName, subDir);

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
        public void ConcatenateOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);

            var data = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("VectorDouble2", DataKind.Double, 1),
                new TextLoader.Column("VectorDouble1", DataKind.Double, 4, 8),
                new TextLoader.Column("Label", DataKind.Boolean, 0)
            });
            var pipeline = mlContext.Transforms.Concatenate("Features", "VectorDouble1", "VectorDouble2");
            var onnxFileName = "Concatenate.onnx";

            TestPipeline(pipeline, data, onnxFileName, new ColumnComparison[] { new ColumnComparison("Features") });

            Done();
        }

        [Fact]
        public void RemoveVariablesInPipelineTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var data = mlContext.Data.LoadFromTextFile<BreastCancerCatFeatureExample>(dataPath,
                separatorChar: '\t',
                hasHeader: false);

            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("F2", "F2", Transforms.OneHotEncodingEstimator.OutputKind.Bag)
            .Append(mlContext.Transforms.ReplaceMissingValues(new MissingValueReplacingEstimator.ColumnOptions("F2")))
            .Append(mlContext.Transforms.Concatenate("Features", "F1", "F2"))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features", numberOfLeaves: 2, numberOfTrees: 1, minimumExampleCountPerLeaf: 2));

            using var model = pipeline.Fit(data);
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
                var subDir = Path.Combine("Onnx", "BinaryClassification", "BreastCancer");
                var onnxTextName = "ExcludeVariablesInOnnxConversion.txt";
                var onnxFileName = "ExcludeVariablesInOnnxConversion.onnx";
                var onnxTextPath = GetOutputPath(subDir, onnxTextName);
                var onnxModelPath = GetOutputPath(subDir, onnxFileName);
                SaveOnnxModel(onnxModel, onnxModelPath, onnxTextPath);
                if (IsOnnxRuntimeSupported())
                {
                    // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                    var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(onnxModelPath, gpuDeviceId: _gpuDeviceId, fallbackToCpu: _fallbackToCpu);
                    var onnxTransformer = onnxEstimator.Fit(data);
                    var onnxResult = onnxTransformer.Transform(data);
                    CompareResults("Score", "Score", transformedData, onnxResult, isRightColumnOnnxScalar: true);
                    CompareResults("Probability", "Probability", transformedData, onnxResult, isRightColumnOnnxScalar: true);
                    CompareResults("PredictedLabel", "PredictedLabel", transformedData, onnxResult, isRightColumnOnnxScalar: true);
                    (onnxTransformer as IDisposable)?.Dispose();
                }
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
            var onnxFileName = "SmallWordEmbed.onnx";
            var onnxTextName = "SmallWordEmbed.txt";
            var subDir = Path.Combine("Onnx", "Transforms", "Sentiment");

            TestPipeline(pipeline, data, onnxFileName, null, onnxTextName, subDir);

            CheckEquality(subDir, onnxTextName, parseOption: NumberParseOption.UseSingle);
            Done();
        }

        [Theory]
        [CombinatorialData]
        public void TokenizingByCharactersOnnxConversionTest(bool useMarkerCharacters)
        {
            var mlContext = new MLContext(seed: 1);
            var dataPath = GetDataPath("wikipedia-detox-250-line-test.tsv");
            var dataView = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("label", DataKind.Boolean, 0),
                new TextLoader.Column("text", DataKind.String, 1)
            }, hasHeader: true);

            var pipeline = new TokenizingByCharactersEstimator(mlContext, useMarkerCharacters: useMarkerCharacters, columns: new[] { ("TokenizedText", "text") });
            var onnxFileName = $"TokenizingByCharacters.onnx";

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("TokenizedText") });

            Done();
        }

        [Theory]
        // These are the supported conversions
        // ML.NET does not allow any conversions between signed and unsigned numeric types
        // Onnx now allows casting a string to other types.
        [InlineData(DataKind.SByte, DataKind.SByte)]
        [InlineData(DataKind.SByte, DataKind.Int16)]
        [InlineData(DataKind.SByte, DataKind.Int32)]
        [InlineData(DataKind.SByte, DataKind.Int64)]
        [InlineData(DataKind.SByte, DataKind.Single)]
        [InlineData(DataKind.SByte, DataKind.Double)]
        [InlineData(DataKind.Byte, DataKind.Byte)]
        [InlineData(DataKind.Byte, DataKind.UInt16)]
        [InlineData(DataKind.Byte, DataKind.UInt32)]
        [InlineData(DataKind.Byte, DataKind.UInt64)]
        [InlineData(DataKind.Byte, DataKind.Single)]
        [InlineData(DataKind.Byte, DataKind.Double)]
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
        [InlineData(DataKind.String, DataKind.Double)]
        [InlineData(DataKind.String, DataKind.Single)]
        [InlineData(DataKind.String, DataKind.UInt64)]
        [InlineData(DataKind.String, DataKind.UInt32)]
        [InlineData(DataKind.String, DataKind.UInt16)]
        [InlineData(DataKind.String, DataKind.Byte)]
        [InlineData(DataKind.String, DataKind.Int64)]
        [InlineData(DataKind.String, DataKind.Int32)]
        [InlineData(DataKind.String, DataKind.Int16)]
        [InlineData(DataKind.String, DataKind.SByte)]
        public void OnnxTypeConversionTest(DataKind fromKind, DataKind toKind)
        {
            var mlContext = new MLContext(seed: 1);
            string filePath = GetDataPath("type-conversion.txt");

            TextLoader.Column[] columns = new[]
            {
                new TextLoader.Column("Value", fromKind, 0, 0)
            };
            var dataView = mlContext.Data.LoadFromTextFile(filePath, columns);

            var pipeline = mlContext.Transforms.Conversion.ConvertType("ValueConverted", "Value", outputKind: toKind);
            var onnxFileName = "typeconversion.onnx";

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("ValueConverted") });

            Done();
        }

        [Theory]
        [InlineData(9)]
        [InlineData(10)]
        [InlineData(11)]
        [InlineData(12)]
        public void PcaOnnxConversionTest(int customOpSetVersion)
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
                var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView, customOpSetVersion);

                var onnxFileName = "pca.onnx";
                var onnxModelPath = GetOutputPath(onnxFileName);

                SaveOnnxModel(onnxModel, onnxModelPath, null);

                if (IsOnnxRuntimeSupported())
                {
                    // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                    var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(onnxModelPath, gpuDeviceId: _gpuDeviceId, fallbackToCpu: _fallbackToCpu);
                    var onnxTransformer = onnxEstimator.Fit(dataView);
                    var onnxResult = onnxTransformer.Transform(dataView);
                    CompareResults("pca", "pca", transformedData, onnxResult);
                    (onnxTransformer as IDisposable)?.Dispose();
                }
            }
            Done();
        }

        [Fact]
        public void OneHotHashEncodingOnnxConversionTest()
        {
            var mlContext = new MLContext();
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);

            var dataView = mlContext.Data.LoadFromTextFile<BreastCancerCatFeatureExample>(dataPath);
            var pipeline = mlContext.Transforms.Categorical.OneHotHashEncoding(new[]{
                    new OneHotHashEncodingEstimator.ColumnOptions("Output", "F3", useOrderedHashing:false),
                });
            var onnxFileName = "OneHotHashEncoding.onnx";

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("Output") });

            Done();
        }

        private class HashData
        {
            public uint Value { get; set; }
        }

        [Theory]
        [CombinatorialData]
        public void MurmurHashKeyTest(
            [CombinatorialValues(DataKind.Byte, DataKind.UInt16, DataKind.UInt32, DataKind.UInt64)] DataKind keyType)
        {
            var dataFile = DeleteOutputPath("KeysToOnnx.txt");
            File.WriteAllLines(dataFile,
                new[]
                {
                    "2",
                    "5",
                    "19"
                });

            var data = ML.Data.LoadFromTextFile(dataFile, new[]
            {
                new TextLoader.Column("Value", keyType, new[]
                {
                    new TextLoader.Range(0)
                }, new KeyCount(10))
            });

            var pipeline = ML.Transforms.Conversion.Hash("ValueHashed", "Value");
            var onnxFileName = "MurmurHashV2.onnx";
            var onnxTextName = "MurmurHashV2.txt";

            TestPipeline(pipeline, data, onnxFileName, new ColumnComparison[] { new ColumnComparison("ValueHashed") }, onnxTextName);

            Done();
        }

        [Theory]
        [CombinatorialData]
        public void MurmurHashScalarTest(
            [CombinatorialValues(DataKind.SByte, DataKind.Int16, DataKind.Int32, DataKind.Int64, DataKind.Byte,
            DataKind.UInt16, DataKind.UInt32, DataKind.UInt64, DataKind.Single, DataKind.Double, DataKind.String, DataKind.Boolean)] DataKind type,
            [CombinatorialValues(1, 5, 31)] int numberOfBits, bool useOrderedHashing)
        {

            var mlContext = new MLContext();
            string dataPath = GetDataPath("type-samples.txt");

            var column = (type == DataKind.SByte) ? 0 :
                (type == DataKind.Byte) ? 2 :
                (type == DataKind.Int16) ? 4 :
                (type == DataKind.UInt16) ? 6 :
                (type == DataKind.Int32) ? 8 :
                (type == DataKind.UInt32) ? 10 :
                (type == DataKind.Int64) ? 12 :
                (type == DataKind.UInt64) ? 14 :
                (type == DataKind.Single) ? 16 :
                (type == DataKind.Double) ? 18 :
                (type == DataKind.String) ? 20 : 22;

            var dataView = mlContext.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("Value", type, column),
            }, separatorChar: '\t', hasHeader: true);

            var pipeline = new HashingEstimator(Env, "Value", useOrderedHashing: useOrderedHashing, numberOfBits: numberOfBits);
            var onnxFileName = "MurmurHashV2.onnx";
            var onnxTextName = "MurmurHashV2.txt";

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("Value") }, onnxTextName);

            Done();
        }

        [Theory]
        [CombinatorialData]
        // Due to lack of Onnxruntime support, OrderedHashing is not supported.
        // An InvalidOperationException stating that the onnx pipeline can't be fully converted is thrown
        // when users try to convert the items mentioned above.
        public void MurmurHashVectorTest(
            [CombinatorialValues(DataKind.SByte, DataKind.Int16, DataKind.Int32, DataKind.Int64, DataKind.Byte,
            DataKind.UInt16, DataKind.UInt32, DataKind.UInt64, DataKind.Single, DataKind.Double, DataKind.String, DataKind.Boolean)] DataKind type,
            [CombinatorialValues(1, 5, 31)] int numberOfBits)
        {

            var mlContext = new MLContext();
            string dataPath = GetDataPath("type-samples.txt");

            var columnStart = (type == DataKind.SByte) ? 0 :
                (type == DataKind.Byte) ? 2 :
                (type == DataKind.Int16) ? 4 :
                (type == DataKind.UInt16) ? 6 :
                (type == DataKind.Int32) ? 8 :
                (type == DataKind.UInt32) ? 10 :
                (type == DataKind.Int64) ? 12 :
                (type == DataKind.UInt64) ? 14 :
                (type == DataKind.Single) ? 16 :
                (type == DataKind.Double) ? 18 :
                (type == DataKind.String) ? 20 : 22;

            var columnEnd = (type == DataKind.SByte) ? 1 :
                (type == DataKind.Byte) ? 3 :
                (type == DataKind.Int16) ? 5 :
                (type == DataKind.UInt16) ? 7 :
                (type == DataKind.Int32) ? 9 :
                (type == DataKind.UInt32) ? 11 :
                (type == DataKind.Int64) ? 13 :
                (type == DataKind.UInt64) ? 15 :
                (type == DataKind.Single) ? 17 :
                (type == DataKind.Double) ? 19 :
                (type == DataKind.String) ? 21 : 23;

            var dataView = mlContext.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("Value", type, columnStart, columnEnd),
            }, separatorChar: '\t', hasHeader: true);

            var pipeline = new HashingEstimator(Env, "Value", useOrderedHashing: false, numberOfBits: numberOfBits);
            var onnxFileName = "MurmurHashV2.onnx";
            var onnxTextName = "MurmurHashV2.txt";

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("Value") }, onnxTextName);

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

            var onnxFileName = "IndicateMissingValues.onnx";
            var onnxTextName = "IndicateMissingValues.txt";
            var subDir = Path.Combine("Onnx", "Transforms");

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("MissingIndicator") }, onnxTextName, subDir);

            CheckEquality(subDir, onnxTextName, parseOption: NumberParseOption.UseSingle);
            Done();
        }

        [Theory]
        [CombinatorialData]
        public void ValueToKeyMappingOnnxConversionTest(
            [CombinatorialValues(DataKind.Single, DataKind.Int64, DataKind.Int32, DataKind.Int16, DataKind.UInt64,
            DataKind.UInt32, DataKind.UInt16, DataKind.Double, DataKind.String, DataKind.Boolean)] DataKind valueType,
            [CombinatorialValues(1, 2)] int maximumNumberOfKeys, ValueToKeyMappingEstimator.KeyOrdinality keyOrdinality,
            bool addKeyValueAnnotationsAsText)
        {
            var mlContext = new MLContext(seed: 1);
            string filePath = (valueType == DataKind.Boolean) ? GetDataPath("type-conversion-boolean.txt")
                : GetDataPath("type-conversion.txt");

            TextLoader.Column[] columnsVector = new[]
            {
                new TextLoader.Column("Value", valueType, 0, 3)
            };
            TextLoader.Column[] columnsScalar = new[]
            {
                new TextLoader.Column("Value", valueType, 0)
            };
            IDataView[] dataViews =
            {
                mlContext.Data.LoadFromTextFile(filePath, columnsScalar, separatorChar: '\t'), //scalar
                mlContext.Data.LoadFromTextFile(filePath, columnsVector , separatorChar: '\t') //vector
            };

            for (int j = 0; j < dataViews.Length; j++)
            {
                var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Key", "Value",
                    maximumNumberOfKeys: maximumNumberOfKeys, keyOrdinality: keyOrdinality,
                    addKeyValueAnnotationsAsText: addKeyValueAnnotationsAsText);

                var onnxFileName = "ValueToKey.onnx";

                TestPipeline(pipeline, dataViews[j], onnxFileName, new ColumnComparison[] { new ColumnComparison("Key") });
            }
            Done();
        }

        [Theory]
        [CombinatorialData]
        // Due to lack of support in OnnxRuntime, String => String mappings are not supported
        public void ValueMappingOnnxConversionTest([CombinatorialValues(DataKind.Int64, DataKind.Int32, DataKind.UInt32, DataKind.UInt64,
        DataKind.UInt16, DataKind.Int16, DataKind.Double, DataKind.String, DataKind.Boolean)]
        DataKind keyType, [CombinatorialValues(true, false)] bool treatValuesAsKeyType)
        {
            var mlContext = new MLContext(seed: 1);
            string filePath = (keyType == DataKind.Boolean) ? GetDataPath("type-conversion-boolean.txt")
                : GetDataPath("type-conversion.txt");

            TextLoader.Column[] columnsVector = new[]
            {
                new TextLoader.Column("Keys", keyType, 0, 2)
            };
            TextLoader.Column[] columnsScalar = new[]
            {
                new TextLoader.Column("Keys", keyType, 0)
            };
            IDataView[] dataViews =
            {
                mlContext.Data.LoadFromTextFile(filePath, columnsScalar, separatorChar: '\t'), //scalar
                mlContext.Data.LoadFromTextFile(filePath, columnsVector , separatorChar: '\t') //vector
            };
            List<IEstimator<ITransformer>> pipelines = new List<IEstimator<ITransformer>>();

            if (keyType == DataKind.Single)
            {
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<float, int> { { 3, 6 }, { 23, 46 } }, "Keys", treatValuesAsKeyType));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<float, long> { { 3, 6 }, { 23, 46 } }, "Keys", treatValuesAsKeyType));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<float, short> { { 3, 6 }, { 23, 46 } }, "Keys", treatValuesAsKeyType));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<float, uint> { { 3, 6 }, { 23, 46 } }, "Keys", treatValuesAsKeyType));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<float, ushort> { { 3, 6 }, { 23, 46 } }, "Keys", treatValuesAsKeyType));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<float, ulong> { { 3, 6 }, { 23, 46 } }, "Keys", treatValuesAsKeyType));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<float, string> { { 3, "True" }, { 23, "False" } }, "Keys", treatValuesAsKeyType));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<float, float> { { 3, 6 }, { 23, 46 } }, "Keys", treatValuesAsKeyType));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<float, double> { { 3, 698 }, { 23, 7908 } }, "Keys", treatValuesAsKeyType));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<float, bool> { { 3, false }, { 23, true } }, "Keys", treatValuesAsKeyType));
            }
            else if (keyType == DataKind.Double)
            {
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<double, int> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<double, uint> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<double, ushort> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<double, ulong> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<double, string> { { 3, "True" }, { 23, "False" } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<double, float> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<double, long> { { 3, 698 }, { 23, 7908 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<double, double> { { 3, 698 }, { 23, 7908 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<double, bool> { { 3, true }, { 23, false } }, "Keys"));
            }
            else if (keyType == DataKind.Boolean)
            {
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<bool, int> { { true, 6 }, { false, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<bool, short> { { true, 6 }, { false, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<bool, uint> { { true, 6 }, { false, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<bool, ushort> { { true, 6 }, { false, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<bool, ulong> { { true, 6 }, { false, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<bool, string> { { true, "True" }, { false, "False" } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<bool, float> { { true, 6 }, { false, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<bool, long> { { true, 698 }, { false, 7908 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<bool, double> { { true, 698 }, { false, 7908 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<bool, bool> { { false, true }, { true, false } }, "Keys"));
            }
            else if (keyType == DataKind.String)
            {
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<string, int> { { "3", 3 }, { "23", 23 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<string, short> { { "3", 3 }, { "23", 23 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<string, uint> { { "3", 6 }, { "23", 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<string, ushort> { { "3", 6 }, { "23", 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<string, ulong> { { "3", 6 }, { "23", 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<string, float> { { "3", 6 }, { "23", 23 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<string, double> { { "3", 6 }, { "23", 23 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<string, long> { { "3", 3 }, { "23", 23 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<string, bool> { { "3", true }, { "23", false } }, "Keys"));
            }
            else if (keyType == DataKind.Int32)
            {
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<int, short> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<int, int> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<int, long> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<int, ushort> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<int, uint> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<int, ulong> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<int, string> { { 3, "True" }, { 23, "False" } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<int, float> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<int, double> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
            }
            else if (keyType == DataKind.Int16)
            {
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<short, short> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<short, int> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<short, long> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<short, ushort> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<short, uint> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<short, ulong> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<short, string> { { 3, "True" }, { 23, "False" } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<short, float> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<short, double> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
            }
            else if (keyType == DataKind.Int64)
            {
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<long, short> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<long, int> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<long, long> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<long, ushort> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<long, uint> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<long, ulong> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<long, string> { { 3, "True" }, { 23, "False" } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<long, float> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<long, double> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
            }
            else if (keyType == DataKind.UInt32)
            {
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<uint, short> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<uint, int> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<uint, long> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<uint, ushort> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<uint, uint> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<uint, ulong> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<uint, string> { { 3, "True" }, { 23, "False" } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<uint, float> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<uint, double> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
            }
            else if (keyType == DataKind.UInt16)
            {
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ushort, short> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ushort, int> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ushort, long> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ushort, ushort> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ushort, uint> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ushort, ulong> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ushort, string> { { 3, "True" }, { 23, "False" } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ushort, float> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ushort, double> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
            }
            else if (keyType == DataKind.UInt64)
            {
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ulong, short> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ulong, int> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ulong, long> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ulong, ushort> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ulong, uint> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ulong, ulong> { { 3, 6 }, { 23, 46 } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ulong, string> { { 3, "True" }, { 23, "False" } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ulong, float> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
                pipelines.Add(mlContext.Transforms.Conversion.MapValue("Value", new Dictionary<ulong, double> { { 3, 6.435f }, { 23, 23.534f } }, "Keys"));
            }
            foreach (IEstimator<ITransformer> pipeline in pipelines)
            {
                for (int j = 0; j < dataViews.Length; j++)
                {
                    var onnxFileName = "ValueMapping.onnx";
                    TestPipeline(pipeline, dataViews[j], onnxFileName, new ColumnComparison[] { new ColumnComparison("Value") });
                }
            }
            Done();
        }

        [Theory]
        [InlineData(DataKind.Single)]
        [InlineData(DataKind.Int64)]
        [InlineData(DataKind.Int32)]
        [InlineData(DataKind.Int16)]
        [InlineData(DataKind.UInt64)]
        [InlineData(DataKind.UInt32)]
        [InlineData(DataKind.UInt16)]
        [InlineData(DataKind.Double)]
        [InlineData(DataKind.String)]
        [InlineData(DataKind.Boolean)]
        public void KeyToValueMappingOnnxConversionTest(DataKind valueType)
        {
            var mlContext = new MLContext(seed: 1);
            string filePath = (valueType == DataKind.Boolean) ? GetDataPath("type-conversion-boolean.txt") : GetDataPath("type-conversion.txt");

            TextLoader.Column[] columnsVector = new[]
            {
                new TextLoader.Column("Value", valueType, 0, 3)
            };
            TextLoader.Column[] columnsScalar = new[]
            {
                new TextLoader.Column("Value", valueType, 0)
            };
            IDataView[] dataViews =
            {
                mlContext.Data.LoadFromTextFile(filePath, columnsScalar, separatorChar: '\t'), //scalar
                mlContext.Data.LoadFromTextFile(filePath, columnsVector , separatorChar: '\t') //vector
            };

            IEstimator<ITransformer>[] pipelines =
            {
                mlContext.Transforms.Conversion.MapValueToKey("Key", "Value").
                Append(mlContext.Transforms.Conversion.MapKeyToValue("Value", "Key")),

                mlContext.Transforms.Conversion.MapValueToKey("Value").
                Append(mlContext.Transforms.Conversion.MapKeyToValue("Value"))
            };

            for (int i = 0; i < pipelines.Length; i++)
            {
                for (int j = 0; j < dataViews.Length; j++)
                {
                    var onnxFileName = "KeyToValue.onnx";

                    TestPipeline(pipelines[i], dataViews[j], onnxFileName, new ColumnComparison[] { new ColumnComparison("Value") });
                }
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

            var pipeline = mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text", new[] { ' ' });

            var onnxFileName = "Tokenizer.onnx";

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("Tokens") });

            Done();
        }

        [Theory]
        [CombinatorialData]
        public void NgramOnnxConversionTest(
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

                mlContext.Transforms.Text.TokenizeIntoCharactersAsKeys("Tokens", "Text")
                .Append(mlContext.Transforms.Text.ProduceNgrams("NGrams", "Tokens",
                            ngramLength: ngramLength,
                            useAllLengths: useAllLength,
                            weighting: weighting)),

                mlContext.Transforms.Text.ProduceWordBags("Tokens", "Text",
                            ngramLength: ngramLength,
                            useAllLengths: useAllLength,
                            weighting: weighting),

                mlContext.Transforms.Text.TokenizeIntoWords("Tokens0", "Text")
                .Append(mlContext.Transforms.Text.ProduceWordBags("Tokens", "Tokens0"))
            };

            for (int i = 0; i < pipelines.Length; i++)
            {
                var pipe = pipelines[i];
                var model = pipe.Fit(dataView);
                var transformedData = model.Transform(dataView);

                var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView);
                var onnxFilename = $"Ngram-{i}-{ngramLength}-{useAllLength}-{weighting}.onnx";
                var txtFilename = $"Ngram-{i}-{ngramLength}-{useAllLength}-{weighting}.txt";
                var onnxFilePath = GetOutputPath(onnxFilename);
                var txtFilePath = GetOutputPath(txtFilename);
                SaveOnnxModel(onnxModel, onnxFilePath, txtFilePath);

                if (IsOnnxRuntimeSupported())
                {
                    var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(onnxFilePath, gpuDeviceId: _gpuDeviceId, fallbackToCpu: _fallbackToCpu);
                    var onnxTransformer = onnxEstimator.Fit(dataView);
                    var onnxResult = onnxTransformer.Transform(dataView);
                    var columnName = i >= pipelines.Length - 2 ? "Tokens" : "NGrams";
                    CompareResults(columnName, columnName, transformedData, onnxResult, 3);

                    VBuffer<ReadOnlyMemory<char>> mlNetSlots = default;
                    VBuffer<ReadOnlyMemory<char>> onnxSlots = default;
                    transformedData.Schema[columnName].GetSlotNames(ref mlNetSlots);
                    onnxResult.Schema[columnName].GetSlotNames(ref onnxSlots);
                    Assert.Equal(mlNetSlots.Length, onnxSlots.Length);
                    var mlNetSlotNames = mlNetSlots.DenseValues().ToList();
                    var onnxSlotNames = onnxSlots.DenseValues().ToList();
                    for (int j = 0; j < mlNetSlots.Length; j++)
                        Assert.Equal(mlNetSlotNames[j].ToString(), onnxSlotNames[j].ToString());
                    (onnxTransformer as IDisposable)?.Dispose();
                }
            }
            Done();
        }

        [Fact]
        public void CustomStopWordsRemovingEstimatorOnnxTest()
        {
            var mlContext = new MLContext();

            var pipeline = mlContext.Transforms.Text.TokenizeIntoWords("Words", "Text")
                .Append(mlContext.Transforms.Text.RemoveStopWords(
                "WordsWithoutStopWords", "Words", stopwords:
                new[] { "cat", "sat", "on" }));

            var samples = new List<TextData>()
            {
                new TextData(){ Text = "cat sat on mat" },
                new TextData(){ Text = "mat not fit cat" },
                new TextData(){ Text = "a cat think mat bad" },
            };
            var dataView = mlContext.Data.LoadFromEnumerable(samples);
            var onnxFileName = $"CustomStopWordsRemovingEstimator.onnx";

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("WordsWithoutStopWords") });

            Done();
        }

        [Fact]
        public void StopWordsRemovingEstimatorOnnxTest()
        {
            var mlContext = new MLContext();

            var pipeline = mlContext.Transforms.Text.TokenizeIntoWords("Words", "Text")
                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords(
                "WordsWithoutStopWords", "Words", language:
                StopWordsRemovingEstimator.Language.English));

            var samples = new List<TextData>()
            {
                new TextData(){ Text = "a go cat sat on mat" },
                new TextData(){ Text = "a mat not fit go cat" },
                new TextData(){ Text = "cat think mat bad a" },
            };
            var dataView = mlContext.Data.LoadFromEnumerable(samples);
            var onnxFileName = $"StopWordsRemovingEstimator.onnx";

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("WordsWithoutStopWords") });

            Done();
        }

        [Theory]
        [InlineData(DataKind.Boolean)]
        [InlineData(DataKind.SByte)]
        [InlineData(DataKind.Byte)]
        [InlineData(DataKind.Int16)]
        [InlineData(DataKind.UInt16)]
        [InlineData(DataKind.Int32)]
        [InlineData(DataKind.UInt32)]
        [InlineData(DataKind.Int64)]
        [InlineData(DataKind.UInt64)]
        [InlineData(DataKind.Single)]
        [InlineData(DataKind.Double)]
        [InlineData(DataKind.String)]
        public void OptionalColumnOnnxTest(DataKind dataKind)
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);

            var dataView = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("Label", dataKind, 0),
                new TextLoader.Column("Thickness", DataKind.Single, 1),
            });

            IHostEnvironment env = mlContext as IHostEnvironment;
            var args = new OptionalColumnTransform.Arguments { Columns = new[] { "Label" }, Data = dataView };
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

            var onnxFileName = $"optionalcol-{dataKind}.onnx";
            var onnxModelPath = GetOutputPath(onnxFileName);
            var onnxTextFileName = "optionalcol.txt";
            var onnxTextPath = GetOutputPath(onnxTextFileName);

            SaveOnnxModel(onnxModel, onnxModelPath, onnxTextPath);
            if (IsOnnxRuntimeSupported())
            {
                string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath, gpuDeviceId: _gpuDeviceId, fallbackToCpu: _fallbackToCpu);
                var onnxTransformer = onnxEstimator.Fit(dataView);
                var onnxResult = onnxTransformer.Transform(dataView);
                CompareResults("Label", "Label", outputData, onnxResult, isRightColumnOnnxScalar: true);
                (onnxTransformer as IDisposable)?.Dispose();
            }
            Done();
        }

        [Fact]
        public void MulticlassTrainersOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataView = mlContext.Data.LoadFromTextFile<BreastCancerMulticlassExample>(dataPath, separatorChar: '\t', hasHeader: false);

            List<IEstimator<ITransformer>> estimators = new List<IEstimator<ITransformer>>()
            {
                mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(),
                mlContext.MulticlassClassification.Trainers.NaiveBayes(),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.AveragedPerceptron()),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.AveragedPerceptron(), useProbabilities:false),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression()),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(), useProbabilities:false),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.LinearSvm()),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.LinearSvm(), useProbabilities:false),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.FastForest()),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.FastForest(), useProbabilities:false),
                mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(),
                mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated()
            };

            if (Environment.Is64BitProcess && NativeLibrary.NativeLibraryExists("lib_lightgbm"))
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

                var onnxFileName = $"{estimator}.onnx";

                TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("PredictedLabel"), new ColumnComparison("Score", 4) });
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
            var onnxFileName = "copycolumns.onnx";

            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("Target") });

            Done();
        }

        [NativeDependencyFact("onnxruntime")]
        public void SelectiveExportOnnxTest()
        {
            var mlContext = new MLContext(seed: 1);

            var trainDataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataView = mlContext.Data.LoadFromTextFile<AdultData>(trainDataPath,
                separatorChar: ';',
                hasHeader: true);

            var mlpipeline = mlContext.Transforms.CopyColumns("Target1", "Target");
            var onnxFileName = "copycolumns.onnx";

            var mlmodel = mlpipeline.Fit(dataView);

            var onnxModelPath = GetOutputPath(onnxFileName);
            using (var stream = File.Create(onnxModelPath))
            {
                mlContext.Model.ConvertToOnnx(mlmodel, dataView, stream, "Target1");
            }

            var model = new OnnxCSharpToProtoWrapper.ModelProto();
            using (var modelStream = File.OpenRead(onnxModelPath))
            using (var codedStream = Google.Protobuf.CodedInputStream.CreateWithLimits(modelStream, Int32.MaxValue, 10))
                model = OnnxCSharpToProtoWrapper.ModelProto.Parser.ParseFrom(codedStream);

            Assert.True(model.Graph.Output.Count == 1);
            Assert.Equal("Target1.output", model.Graph.Output[0].Name);

            // Make sure that even though the column wasn't passed to ONNX, that it can still be used directly from ML.Net
            var pipeline = mlContext.Transforms.ApplyOnnxModel(onnxModelPath);
            var loadedModel = pipeline.Fit(dataView);

            // Getting the preview will cause an issue if there is an error since ONNX is no longer exporting that column.
            var loadedData = loadedModel.Transform(dataView).Preview(1);
            Assert.Equal((Single)140.66, loadedData.ColumnView[1].Values[0]);

            Done();
        }

        [Fact]
        public void UseKeyDataViewTypeAsUInt32InOnnxInput()
        {
            // In this test an onnx model which expect a uin32 input column is applied to a KeyDataViewType input column
            // This, is done as needed by NimbusML. For more context see: https://github.com/microsoft/NimbusML/issues/426

            // Step 1: Load the Iris Dataset and apply a Value To Key Mapping to it.
            // Save the resulting dataview in .idv format eliminating all hidden columns
            var mlContext = new MLContext();
            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("Label", DataKind.String, 0),
                        new TextLoader.Column("SepalLength", DataKind.Single, 1),
                        new TextLoader.Column("SepalWidth", DataKind.Single, 2),
                        new TextLoader.Column("PetalLength", DataKind.Single, 3),
                        new TextLoader.Column("PetalWidth", DataKind.Single, 4)
                    },
                hasHeader: false
            );

            string dataPath = GetDataPath("iris.txt");
            var originalData = loader.Load(dataPath);
            var pipeline1 = mlContext.Transforms.Conversion.MapValueToKey("Label");
            var mappedData = pipeline1.Fit(originalData).Transform(originalData);

            string mappedDataPath = GetOutputPath("kdvt-as-uint32-mapped-data.idv");
            using (FileStream stream = new FileStream(mappedDataPath, FileMode.Create))
                mlContext.Data.SaveAsBinary(mappedData, stream, keepHidden: false);

            // Step 2: Load back the saved .idv
            // This IDataView will have a Label column of type KeyDataViewType
            // It's necessary to do this, because if I were to use mappedData directly inside the next
            // steps, then when saving the ONNX model, it would actually also save the ValueToKeyTransformer part
            // and that wouldn't reproduce the scenario.
            IDataView reloadedData = mlContext.Data.LoadFromBinary(mappedDataPath);

            // Step 3: Create ONNX model which simply applies Identity to Label column
            var pipeline2 = mlContext.Transforms.CopyColumns("Label", "Label");
            var model = pipeline2.Fit(reloadedData);

            var onnxModelPath = GetOutputPath("onnxmodel1-kdvt-as-uint32.onnx");
            using (FileStream stream = new FileStream(onnxModelPath, FileMode.Create))
                mlContext.Model.ConvertToOnnx(model, reloadedData, stream);

            // Step 4: Get input and output names of model
            var onnxProtoBufModel = mlContext.Model.ConvertToOnnxProtobuf(model, reloadedData);
            string[] inputNames = onnxProtoBufModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
            string[] outputNames = onnxProtoBufModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();

            if (IsOnnxRuntimeSupported())
            {
                // Step 5: Apply Onnx Model
                var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath, gpuDeviceId: _gpuDeviceId, fallbackToCpu: _fallbackToCpu);
                var onnxTransformer = onnxEstimator.Fit(reloadedData);
                var onnxResult = onnxTransformer.Transform(reloadedData);

                // Step 6: Compare results to an onnx model created using the mappedData IDataView
                // Notice that this ONNX model would actually include the steps to do the ValueToKeyTransformer mapping,
                // because mappedData actually includes the information to do the mapping, and so ONNX would that automatically.
                // And because of this, it can only be applied to originalData dataview, despite mappedData was used to create the model.
                // If it's tried to apply this model to mappedData or reloadedData, it will throw an exception, since the ONNX model
                // will expect a Label input of type string (which only originalData provides).
                string onnxModelPath2 = GetOutputPath("onnxmodel2-kdvt-as-uint32.onnx");
                using (FileStream stream = new FileStream(onnxModelPath2, FileMode.Create))
                    mlContext.Model.ConvertToOnnx(model, mappedData, stream);
                var onnxEstimator2 = mlContext.Transforms.ApplyOnnxModel(outputNames, inputNames, onnxModelPath2, gpuDeviceId: _gpuDeviceId, fallbackToCpu: _fallbackToCpu);
                var onnxTransformer2 = onnxEstimator2.Fit(originalData);
                var onnxResult2 = onnxTransformer2.Transform(originalData);

                var stdSuffix = ".output";
                foreach (var name in outputNames)
                {
                    Assert.EndsWith(stdSuffix, name);
                    var colName = name.Replace(stdSuffix, "");
                    CompareResults(colName, colName, onnxResult, onnxResult2);
                }
                (onnxTransformer as IDisposable)?.Dispose();
                (onnxTransformer2 as IDisposable)?.Dispose();
            }

            Done();
        }

        [Theory]
        [InlineData(DataKind.String)]
        [InlineData(DataKind.Single)]
        [InlineData(DataKind.Double)]
        public void FeatureSelectionOnnxTest(DataKind dataKind)
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);

            var dataView = mlContext.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("Scalar", dataKind, 6),
                new TextLoader.Column("Vector", dataKind, 1, 6),
                new TextLoader.Column("Label", DataKind.Boolean, 0)
            });

            IEstimator<ITransformer>[] pipelines =
            {
                // one or more features selected
                mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("VectorOutput", "Vector", count: 690).
                    Append(mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("ScalarOutput", "Scalar", count: 100)),

                // no feature selected => column suppressed
                mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("VectorOutput", "Vector", count: 800).
                    Append(mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("ScalarOutput", "Scalar", count: 800)),

                mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("VectorOutput", "Vector").
                    Append(mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("ScalarOutput", "Scalar"))
            };
            for (int i = 0; i < pipelines.Length; i++)
            {
                //There's currently no support for suppressed string columns, since onnx string variable initiation is not supported
                if (dataKind == DataKind.String && i > 0)
                    break;

                var onnxFileName = "countfeatures.onnx";

                TestPipeline(pipelines[i], dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("VectorOutput"), new ColumnComparison("ScalarOutput") });
            }
            Done();
        }

        [Fact]
        public void SelectColumnsOnnxTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);

            var dataView = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("Label", DataKind.Boolean, 0),
                new TextLoader.Column("Thickness", DataKind.Double, 1),
                new TextLoader.Column("Size", DataKind.Single, 2),
                new TextLoader.Column("Shape", DataKind.Int32, 3),
                new TextLoader.Column("Adhesion", DataKind.Int32, 4),
                new TextLoader.Column("EpithelialSize", DataKind.Int32, 5),
                new TextLoader.Column("BlandChromatin", DataKind.Int32, 7),
                new TextLoader.Column("NormalNucleoli", DataKind.Int32, 8),
                new TextLoader.Column("Mitoses", DataKind.Int32, 9),
            });

            var pipeline = mlContext.Transforms.ReplaceMissingValues("Size").Append(mlContext.Transforms.SelectColumns(new[] { "Size", "Shape", "Thickness", "Label" }));
            var onnxFileName = "selectcolumns.onnx";
            var onnxTxtName = "SelectColumns.txt";
            var subDir = Path.Combine("Onnx", "Transforms");
            TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("Size"), new ColumnComparison("Shape"), new ColumnComparison("Thickness"), new ColumnComparison("Label") }, onnxTxtName, subDir);

            CheckEquality(subDir, onnxTxtName, digitsOfPrecision: 1);

            Done();
        }

        private class BreastCancerMulticlassExampleNonDefaultColNames
        {
            [LoadColumn(1)]
            public string Label;

            [LoadColumn(2, 9), VectorType(8)]
            public float[] MyFeatureVector;
        }

        private class BreastCancerBinaryClassificationNonDefaultColNames
        {
            [LoadColumn(0)]
            public bool Label;

            [LoadColumn(2, 9), VectorType(8)]
            public float[] MyFeatureVector;
        }

        [Fact]
        public void NonDefaultColNamesBinaryClassificationOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var dataView = mlContext.Data.LoadFromTextFile<BreastCancerBinaryClassificationNonDefaultColNames>(dataPath, separatorChar: '\t', hasHeader: false);
            List<IEstimator<ITransformer>> estimators = new List<IEstimator<ITransformer>>()
            {
                mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "MyFeatureVector"),
                mlContext.BinaryClassification.Trainers.FastForest("Label", "MyFeatureVector"),
                mlContext.BinaryClassification.Trainers.FastTree("Label", "MyFeatureVector"),
                mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression("Label", "MyFeatureVector"),
                mlContext.BinaryClassification.Trainers.LinearSvm("Label", "MyFeatureVector"),
                mlContext.BinaryClassification.Trainers.Prior(),
                mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "MyFeatureVector"),
                mlContext.BinaryClassification.Trainers.SdcaNonCalibrated("Label", "MyFeatureVector"),
                mlContext.BinaryClassification.Trainers.SgdCalibrated("Label", "MyFeatureVector"),
                mlContext.BinaryClassification.Trainers.SgdNonCalibrated("Label", "MyFeatureVector"),
            };
            if (NativeLibrary.NativeLibraryExists("MklImports"))
            {
                estimators.Add(mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression("Label", "MyFeatureVector"));
            }
            if (Environment.Is64BitProcess && NativeLibrary.NativeLibraryExists("lib_lightgbm"))
            {
                estimators.Add(mlContext.BinaryClassification.Trainers.LightGbm("Label", "MyFeatureVector"));
            }

            var initialPipeline = mlContext.Transforms.ReplaceMissingValues("MyFeatureVector").
                Append(mlContext.Transforms.NormalizeMinMax("MyFeatureVector"));
            foreach (var estimator in estimators)
            {
                var pipeline = initialPipeline.Append(estimator);
                var onnxFileName = $"{estimator}.onnx";

                TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("Score", 3), new ColumnComparison("PredictedLabel") });
            }
            Done();
        }

        [Fact]
        public void NonDefaultColNamesMultiClassificationOnnxConversionTest()
        {
            var mlContext = new MLContext(seed: 1);

            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataView = mlContext.Data.LoadFromTextFile<BreastCancerMulticlassExampleNonDefaultColNames>(dataPath, separatorChar: '\t', hasHeader: false);

            List<IEstimator<ITransformer>> estimators = new List<IEstimator<ITransformer>>()
            {
                mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy("Label", "MyFeatureVector"),
                mlContext.MulticlassClassification.Trainers.NaiveBayes("Label", "MyFeatureVector"),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "MyFeatureVector")),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "MyFeatureVector"), useProbabilities:false),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression("Label", "MyFeatureVector")),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression("Label", "MyFeatureVector"), useProbabilities:false),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.LinearSvm("Label", "MyFeatureVector")),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.LinearSvm("Label", "MyFeatureVector"), useProbabilities:false),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.FastForest("Label", "MyFeatureVector")),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.FastForest("Label", "MyFeatureVector"), useProbabilities:false),
                mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "MyFeatureVector"),
                mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated("Label", "MyFeatureVector")
            };

            if (Environment.Is64BitProcess && NativeLibrary.NativeLibraryExists("lib_lightgbm"))
            {
                estimators.Add(mlContext.MulticlassClassification.Trainers.LightGbm("Label", "MyFeatureVector"));
            }

            var initialPipeline = mlContext.Transforms.ReplaceMissingValues("MyFeatureVector")
                .Append(mlContext.Transforms.NormalizeMinMax("MyFeatureVector"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"));

            foreach (var estimator in estimators)
            {
                var pipeline = initialPipeline.Append(estimator);
                var onnxFileName = $"{estimator}.onnx";

                TestPipeline(pipeline, dataView, onnxFileName, new ColumnComparison[] { new ColumnComparison("Score", 4), new ColumnComparison("PredictedLabel") });
            }
            Done();
        }

        [Fact]
        public void OneHotHashEncodingOnnxConversionWithCustomOpSetVersionTest()
        {
            var mlContext = new MLContext();
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);

            var dataView = ML.Data.LoadFromTextFile<BreastCancerCatFeatureExample>(dataPath);
            var pipe = ML.Transforms.Categorical.OneHotHashEncoding(new[]{
                    new OneHotHashEncodingEstimator.ColumnOptions("Output", "F3", useOrderedHashing:false),
                });
            var model = pipe.Fit(dataView);
            var transformedData = model.Transform(dataView);

            try
            {
                var onnxModelPath = GetOutputPath("onnxmodel_custom_opset_version_test.onnx");
                using (FileStream stream = new FileStream(onnxModelPath, FileMode.Create))
                    mlContext.Model.ConvertToOnnx(model, dataView, 9, stream);
                Assert.True(false);
            }
            catch (System.Exception ex)
            {
                Assert.Contains("Requested OpSet version 9 is lower than HashTransform's minimum OpSet version requirement: 11", ex.Message);
                return;
            }

            try
            {
                var onnxModelPath = GetOutputPath("onnxmodel_custom_opset_version_test.onnx");
                using (FileStream stream = new FileStream(onnxModelPath, FileMode.Create))
                    mlContext.Model.ConvertToOnnx(model, dataView, 13, stream);
                Assert.True(false);
            }
            catch (System.Exception ex)
            {
                Assert.Contains("Requested OpSet version 13 is higher than the current most updated OpSet version 12", ex.Message);
                return;
            }

            try
            {
                var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView, 9);
                Assert.True(false);
            }
            catch (System.Exception ex)
            {
                Assert.Contains("Requested OpSet version 9 is lower than HashTransform's minimum OpSet version requirement: 11", ex.Message);
                return;
            }

            try
            {
                var onnxModel = mlContext.Model.ConvertToOnnxProtobuf(model, dataView, 13);
                Assert.True(false);
            }
            catch (System.Exception ex)
            {
                Assert.Contains("Requested OpSet version 13 is higher than the current most updated OpSet version 12", ex.Message);
                return;
            }

            Done();
        }

        [Theory]
        [CombinatorialData]
        public void NormalizingEstimatorConversionTests(
            bool fixZero)
        {
            // Shared variables.
            var columnsToCompare = new ColumnComparison[] { new ColumnComparison("Features") };
            IEstimator<ITransformer> pipe;
            string onnxFileName;

            // Data for vector inputs.
            var vecSamples = new DataPoint[]
            {
                new DataPoint { Features = new float[3] {0.01f, 0.02f, 0.03f} },
                new DataPoint { Features = new float[3] {0.04f, 0.05f, 0.06f} },
                new DataPoint { Features = new float[3] {0.07f, 0.08f, 0.09f} },
                new DataPoint { Features = new float[3] {0.10f, 0.11f, 0.12f} },
                new DataPoint { Features = new float[3] {0.13f, 0.14f, 0.15f} }
            };

            // Data for scalar inputs.
            var scalarSamples = new[]
            {
                new { Features = 0.03f },
                new { Features = 0.06f },
                new { Features = 0.09f },
                new { Features = 0.12f },
                new { Features = 0.15f }
            };

            // Test vector input NormalizeMinMax.
            pipe = ML.Transforms.NormalizeMinMax(nameof(DataPoint.Features), fixZero: fixZero);
            onnxFileName = $"NormMinMaxVec-{fixZero}.onnx";
            TestPipeline(pipe, vecSamples, onnxFileName, columnsToCompare);

            // Test scalar input NormalizeMinMax.
            pipe = ML.Transforms.NormalizeMinMax(nameof(DataPoint.Features), fixZero: fixZero);
            onnxFileName = $"NormMinMaxScalar-{fixZero}.onnx";
            TestPipeline(pipe, scalarSamples, onnxFileName, columnsToCompare);

            // Test vector input NormalizeMeanVariance.
            pipe = ML.Transforms.NormalizeMeanVariance(nameof(DataPoint.Features), fixZero: fixZero);
            onnxFileName = $"NormMeanVarVec-{fixZero}.onnx";
            TestPipeline(pipe, vecSamples, onnxFileName, columnsToCompare);

            // Test scalar input NormalizeMeanVariance.
            pipe = ML.Transforms.NormalizeMeanVariance(nameof(DataPoint.Features), fixZero: fixZero);
            onnxFileName = $"NormMeanVarScalar-{fixZero}.onnx";
            TestPipeline(pipe, scalarSamples, onnxFileName, columnsToCompare);

            // Test vector input NormalizeLogMeanVariance.
            pipe = ML.Transforms.NormalizeLogMeanVariance(nameof(DataPoint.Features), fixZero: fixZero, useCdf: false);
            onnxFileName = $"NormLogMeanVarVec-{fixZero}.onnx";
            TestPipeline(pipe, vecSamples, onnxFileName, columnsToCompare);

            // Test scalar input NormalizeLogMeanVariance.
            pipe = ML.Transforms.NormalizeLogMeanVariance(nameof(DataPoint.Features), fixZero: fixZero, useCdf: false);
            onnxFileName = $"NormLogMeanVarScalar-{fixZero}.onnx";
            TestPipeline(pipe, scalarSamples, onnxFileName, columnsToCompare);

            // Test vector input NormalizeRobustScaling.
            pipe = ML.Transforms.NormalizeRobustScaling(nameof(DataPoint.Features), centerData: fixZero);
            onnxFileName = $"NormRobScalVec-{fixZero}.onnx";
            TestPipeline(pipe, vecSamples, onnxFileName, columnsToCompare);

            // Test scalar input NormalizeRobustScaling.
            pipe = ML.Transforms.NormalizeRobustScaling(nameof(DataPoint.Features), centerData: fixZero);
            onnxFileName = $"NormRobScalScalar-{fixZero}.onnx";
            TestPipeline(pipe, scalarSamples, onnxFileName, columnsToCompare);

            Done();
        }

        /// <summary>
        /// Class used to denote which comlumns are being compared. Precision defaults to 6 when not specified.
        /// </summary>
        private class ColumnComparison
        {
            public string Name;
            public int Precision = 6;

            public ColumnComparison(string name, int precision = 6)
            {
                Name = name;
                Precision = precision;
            }
        }

        /// <summary>
        /// Helper method that takes a single IEstimator{ITransformer} and an IEnumerable{TRow} and converts the IEstimator{ITransformer} to an
        /// EstimatorChain{ITransformer} and the IEnumerable{TRow} into an IDataView and then calls the actual testing method.
        /// </summary>
        /// <typeparam name="TRow">The type of the IEnumerable, will be auto detected.</typeparam>
        /// <param name="pipeline">A single IEstimator{ITransformer} to be tested.</param>
        /// <param name="data">The test data as a IEnumerable{TRow}.</param>
        /// <param name="onnxFileName">Name to save the ONNX model file.</param>
        /// <param name="columnsToCompare">Columns you want to compare. This assumes that the column name in ONNX is the same as ML.Net, so only 1 name per column is provided. The second value is the precision</param>
        /// <param name="schemaDefinition">Optional schema definition for the IEnumerable{TRow}.</param>
        /// <param name="onnxTxtName">Optional file path to write the text version of the onnx model.</param>
        /// <param name="onnxTxtSubDir">Optional subdirectory for the onnxTxtName.</param>
        private void TestPipeline<TRow>(IEstimator<ITransformer> pipeline, IEnumerable<TRow> data, string onnxFileName, ColumnComparison[] columnsToCompare, SchemaDefinition schemaDefinition = null, string onnxTxtName = null, string onnxTxtSubDir = null)
            where TRow : class
        {
            var dataView = ML.Data.LoadFromEnumerable(data, schemaDefinition);
            TestPipeline(pipeline, dataView, onnxFileName, columnsToCompare, onnxTxtName, onnxTxtSubDir);
        }

        /// <summary>
        /// Helper method that takes a single IEstimator{ITransformer} and an IDataView and converts the IEstimator{ITransformer} to an
        /// EstimatorChain{ITransformer} and then calls the actual testing method.
        /// </summary>
        /// <param name="pipeline">A single IEstimator{ITransformer} to be tested.</param>
        /// <param name="dataView">The test data.</param>
        /// <param name="onnxFileName">Name to save the ONNX model file.</param>
        /// <param name="columnsToCompare">Columns you want to compare. This assumes that the column name in ONNX is the same as ML.Net, so only 1 name per column is provided. The second value is the precision</param>
        /// <param name="onnxTxtName">Optional file path to write the text version of the onnx model.</param>
        /// <param name="onnxTxtSubDir">Optional subdirectory for the onnxTxtName.</param>
        private void TestPipeline(IEstimator<ITransformer> pipeline, IDataView dataView, string onnxFileName, ColumnComparison[] columnsToCompare, string onnxTxtName = null, string onnxTxtSubDir = null)
        {
            var chain = new EstimatorChain<ITransformer>().Append(pipeline);
            TestPipeline(chain, dataView, onnxFileName, columnsToCompare, onnxTxtName, onnxTxtSubDir);
        }

        /// <summary>
        /// Helper method that takes an EstimatorChain{TLastTransformer} and an IEnumerable{TRow} and converts the IEnumerable{TRow} into an IDataView and then
        /// calls the actual testing method.
        /// </summary>
        /// <typeparam name="TLastTransformer">The type of for the EstimatorChain. Will be auto detected.</typeparam>
        /// <typeparam name="TRow">The type of the IEnumerable. Will be auto detected.</typeparam>
        /// <param name="pipeline">A single IEstimator{ITransformer} to be tested.</param>
        /// <param name="data">The test data as a IEnumerable{TRow}.</param>
        /// <param name="onnxFileName">Name to save the ONNX model file.</param>
        /// <param name="columnsToCompare">Columns you want to compare. This assumes that the column name in ONNX is the same as ML.Net, so only 1 name per column is provided. The second value is the precision</param>
        /// <param name="schemaDefinition">Optional schema definition for the IEnumerable{TRow}.</param>
        /// <param name="onnxTxtName">Optional file path to write the text version of the onnx model.</param>
        /// <param name="onnxTxtSubDir">Optional subdirectory for the onnxTxtName.</param>
        private void TestPipeline<TLastTransformer, TRow>(EstimatorChain<TLastTransformer> pipeline, IEnumerable<TRow> data, string onnxFileName, ColumnComparison[] columnsToCompare, SchemaDefinition schemaDefinition = null, string onnxTxtName = null, string onnxTxtSubDir = null)
            where TLastTransformer : class, ITransformer
            where TRow : class
        {
            var dataView = ML.Data.LoadFromEnumerable(data, schemaDefinition);
            TestPipeline(pipeline, dataView, onnxFileName, columnsToCompare, onnxTxtName, onnxTxtSubDir);
        }

        /// <summary>
        /// Testing method that takes an EstimatorChain{TLastTransformer} and an IDataView and converst the IEnumerable{TRow} into an IDataView and then
        /// converts the chain to an ONNX model and compares the results.
        /// </summary>
        /// <typeparam name="TLastTransformer">The type of for the EstimatorChain. Will be auto detected.</typeparam>
        /// <param name="pipeline">A single IEstimator{ITransformer} to be tested.</param>
        /// <param name="dataView">The test data.</param>
        /// <param name="onnxFileName">Name to save the ONNX model file.</param>
        /// <param name="columnsToCompare">Columns you want to compare. This assumes that the column name in ONNX is the same as ML.Net, so only 1 name per column is provided. The second value is the precision</param>
        /// <param name="onnxTxtName">Optional file path to write the text version of the onnx model.</param>
        /// <param name="onnxTxtSubDir">Optional subdirectory for the onnxTxtName.</param>
        private void TestPipeline<TLastTransformer>(EstimatorChain<TLastTransformer> pipeline, IDataView dataView, string onnxFileName, ColumnComparison[] columnsToCompare, string onnxTxtName = null, string onnxTxtSubDir = null)
            where TLastTransformer : class, ITransformer
        {
            using var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);
            var onnxModel = ML.Model.ConvertToOnnxProtobuf(model, dataView);

            var onnxModelPath = GetOutputPath(onnxFileName);
            var onnxTextFullPath = GetOutputPath(onnxTxtSubDir, onnxTxtName);

            SaveOnnxModel(onnxModel, onnxModelPath, onnxTextFullPath);

            // Compare results produced by ML.NET and ONNX's runtime.
            if (IsOnnxRuntimeSupported() && columnsToCompare != null)
            {
                // Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
                var onnxEstimator = ML.Transforms.ApplyOnnxModel(onnxModelPath, gpuDeviceId: _gpuDeviceId, fallbackToCpu: _fallbackToCpu);
                var onnxTransformer = onnxEstimator.Fit(dataView);
                var onnxResult = onnxTransformer.Transform(dataView);

                // Compare all the columns between ML.Net and ONNX.
                foreach (var column in columnsToCompare)
                {
                    CompareResults(column.Name, column.Name, transformedData, onnxResult, column.Precision, true);
                }
                (onnxTransformer as IDisposable)?.Dispose();
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
