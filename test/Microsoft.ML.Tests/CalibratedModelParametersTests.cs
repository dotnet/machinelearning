// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class CalibratedModelParametersTests : TestDataPipeBase
    {
        public CalibratedModelParametersTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestParameterMixingCalibratedModelParametersLoading()
        {
            var data = GetDenseDataset();
            var model = ML.BinaryClassification.Trainers.LbfgsLogisticRegression(
                new LbfgsLogisticRegressionBinaryTrainer.Options { NumberOfThreads = 1 }).Fit(data);

            var modelAndSchemaPath = GetOutputPath("TestParameterMixingCalibratedModelParametersLoading.zip");
            ML.Model.Save(model, data.Schema, modelAndSchemaPath);

            var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
            var castedModel = loadedModel as BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>;

            Assert.NotNull(castedModel);

            Type expectedInternalType = typeof(ParameterMixingCalibratedModelParameters<LinearBinaryModelParameters, PlattCalibrator>);
            Assert.Equal(expectedInternalType, castedModel.Model.GetType());
            Assert.Equal(model.Model.GetType(), castedModel.Model.GetType());
            Done();
        }

        [Fact]
        public void TestValueMapperCalibratedModelParametersLoading()
        {
            var data = GetDenseDataset();

            var model = ML.BinaryClassification.Trainers.Gam(
                new GamBinaryTrainer.Options { NumberOfThreads = 1 }).Fit(data);

            var modelAndSchemaPath = GetOutputPath("TestValueMapperCalibratedModelParametersLoading.zip");
            ML.Model.Save(model, data.Schema, modelAndSchemaPath);

            var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
            var castedModel = loadedModel as BinaryPredictionTransformer<CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>;

            Assert.NotNull(castedModel);

            Type expectedInternalType = typeof(ValueMapperCalibratedModelParameters<GamBinaryModelParameters, PlattCalibrator>);
            Assert.Equal(expectedInternalType, castedModel.Model.GetType());
            Assert.Equal(model.Model.GetType(), castedModel.Model.GetType());
            Done();
        }


        [Fact]
        public void TestFeatureWeightsCalibratedModelParametersLoading()
        {
            var data = GetDenseDataset();

            var model = ML.BinaryClassification.Trainers.FastTree(
                new FastTreeBinaryTrainer.Options { NumberOfThreads = 1 }).Fit(data);

            var modelAndSchemaPath = GetOutputPath("TestFeatureWeightsCalibratedModelParametersLoading.zip");
            ML.Model.Save(model, data.Schema, modelAndSchemaPath);

            var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
            var castedModel = loadedModel as BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>;

            Assert.NotNull(castedModel);

            Type expectedInternalType = typeof(FeatureWeightsCalibratedModelParameters<FastTreeBinaryModelParameters, PlattCalibrator>);
            Assert.Equal(expectedInternalType, castedModel.Model.GetType());
            Assert.Equal(model.Model.GetType(), castedModel.Model.GetType());
            Done();
        }

        [Theory]
        [InlineData(typeof(LinearBinaryModelParameters))]
        [InlineData(typeof(GamBinaryModelParameters))]
        [InlineData(typeof(FastTreeBinaryModelParameters))]
        [InlineData(typeof(FastForestBinaryModelParameters))]
        [InlineData(typeof(LightGbmBinaryModelParameters))]
        public void BinaryModelWithoutEmbeddedCalibratorLoadsUnwrapped(Type modelType)
        {
            var loaded = RoundTripBinaryModel(CreateBinaryModel(modelType));

            Assert.Equal(modelType, loaded.GetType());
            Assert.False(loaded is CalibratedModelParametersBase);
            AssertBinaryScores(loaded);
            Done();
        }

        [Theory]
        [InlineData(typeof(LinearBinaryModelParameters), false)]
        [InlineData(typeof(LinearBinaryModelParameters), true)]
        [InlineData(typeof(GamBinaryModelParameters), false)]
        [InlineData(typeof(FastTreeBinaryModelParameters), false)]
        [InlineData(typeof(FastForestBinaryModelParameters), false)]
        [InlineData(typeof(LightGbmBinaryModelParameters), false)]
        public void EmbeddedCalibratorLoadsAndResavesInWrapperFormat(Type modelType, bool useNaiveCalibrator)
        {
            ICalibrator calibrator = useNaiveCalibrator
                ? new NaiveCalibrator(Env, min: -4, binSize: 4, binProbs: new[] { 0.125f, 0.875f })
                : new PlattCalibrator(Env, slope: -0.75, offset: 0.25);
            Type wrapperType = modelType == typeof(LinearBinaryModelParameters) && !useNaiveCalibrator
                ? typeof(ParameterMixingCalibratedModelParameters<,>)
                : modelType == typeof(LightGbmBinaryModelParameters)
                    ? typeof(ValueMapperCalibratedModelParameters<,>)
                    : typeof(SchemaBindableCalibratedModelParameters<,>);

            // These synthetic archives exercise the supported layout, not a particular historical release.
            var loaded = RoundTripBinaryModel(CreateBinaryModel(modelType), calibrator);

            Assert.Equal(wrapperType.MakeGenericType(modelType, typeof(ICalibrator)), loaded.GetType());
            AssertCalibratedBinaryModel(loaded, modelType, calibrator, wrapperType);
            AssertBinaryScores(loaded, calibrated: true, useNaiveCalibrator: useNaiveCalibrator);

            var reloaded = RoundTripBinaryModel(loaded);

            AssertCalibratedBinaryModel(reloaded, modelType, calibrator, wrapperType);
            AssertBinaryScores(reloaded, calibrated: true, useNaiveCalibrator: useNaiveCalibrator);
            Done();
        }

        [Theory]
        [InlineData(typeof(LinearBinaryModelParameters))]
        [InlineData(typeof(GamBinaryModelParameters))]
        [InlineData(typeof(FastTreeBinaryModelParameters))]
        [InlineData(typeof(FastForestBinaryModelParameters))]
        [InlineData(typeof(LightGbmBinaryModelParameters))]
        public void CalibratedBinaryModelRoundTripPreservesWrapperAndPredictions(Type modelType)
        {
            var calibrator = new PlattCalibrator(Env, slope: -0.75, offset: 0.25);
            var model = CreateCalibratedBinaryModel(CreateBinaryModel(modelType), calibrator);

            var loaded = RoundTripBinaryModel(model);

            Assert.Equal(model.GetType(), loaded.GetType());
            AssertCalibratedBinaryModel(loaded, modelType, calibrator, model.GetType().GetGenericTypeDefinition());
            AssertBinaryScores(loaded, calibrated: true);
            Done();
        }

        [Theory]
        [InlineData(typeof(LinearBinaryModelParameters))]
        [InlineData(typeof(GamBinaryModelParameters))]
        [InlineData(typeof(FastTreeBinaryModelParameters))]
        [InlineData(typeof(FastForestBinaryModelParameters))]
        [InlineData(typeof(LightGbmBinaryModelParameters))]
        public void InvalidEmbeddedCalibratorFailsToLoad(Type modelType)
        {
            var model = CreateBinaryModel(modelType);
            var calibrator = new PlattCalibrator(Env, slope: double.NaN, offset: 0.25);

            var exception = Assert.Throws<InvalidOperationException>(() => RoundTripBinaryModel(model, calibrator));

            // The component catalogue wraps the calibrator's decoding failure.
            Assert.IsType<FormatException>(exception.GetBaseException());
            Done();
        }

        private IPredictorProducing<float> CreateBinaryModel(Type modelType)
        {
            // All five models score -2.5 for [-1] and 1.5 for [1], without training or native libraries.
            if (modelType == typeof(LinearBinaryModelParameters))
                return new LinearBinaryModelParameters(Env, new VBuffer<float>(1, new[] { 2f }), bias: -0.5f);
            if (modelType == typeof(GamBinaryModelParameters))
            {
                return new GamBinaryModelParameters(Env,
                    new[] { new[] { 0d, double.PositiveInfinity } }, new[] { new[] { -2d, 2d } },
                    intercept: -0.5, inputLength: 1, featureToInputMap: null);
            }

            InternalRegressionTree tree = modelType == typeof(FastForestBinaryModelParameters)
                ? new InternalQuantileRegressionTree(
                    splitFeatures: new[] { 0 }, splitGain: new[] { 1d }, gainPValue: null,
                    rawThresholds: new[] { 0f }, defaultValueForMissing: null,
                    lteChild: new[] { -1 }, gtChild: new[] { -2 }, leafValues: new[] { -2.5, 1.5 },
                    categoricalSplitFeatures: new int[1][], categoricalSplit: new bool[1])
                : new InternalRegressionTree(
                    splitFeatures: new[] { 0 }, splitGain: new[] { 1d }, gainPValue: null,
                    rawThresholds: new[] { 0f }, defaultValueForMissing: null,
                    lteChild: new[] { -1 }, gtChild: new[] { -2 }, leafValues: new[] { -2.5, 1.5 },
                    categoricalSplitFeatures: new int[1][], categoricalSplit: new bool[1]);
            var ensemble = new InternalTreeEnsemble();
            ensemble.AddTree(tree);

            if (modelType == typeof(FastTreeBinaryModelParameters))
                return new FastTreeBinaryModelParameters(Env, ensemble, featureCount: 1, innerArgs: null);
            if (modelType == typeof(FastForestBinaryModelParameters))
                return new FastForestBinaryModelParameters(Env, ensemble, featureCount: 1, innerArgs: null);
            if (modelType == typeof(LightGbmBinaryModelParameters))
                return new LightGbmBinaryModelParameters(Env, ensemble, featureCount: 1, innerArgs: null);
            throw new ArgumentOutOfRangeException(nameof(modelType));
        }

        private IPredictorProducing<float> CreateCalibratedBinaryModel(IPredictorProducing<float> model, PlattCalibrator calibrator)
        {
            return model switch
            {
                LinearBinaryModelParameters linear => new ParameterMixingCalibratedModelParameters<LinearBinaryModelParameters, PlattCalibrator>(Env, linear, calibrator),
                GamBinaryModelParameters gam => new ValueMapperCalibratedModelParameters<GamBinaryModelParameters, PlattCalibrator>(Env, gam, calibrator),
                FastTreeBinaryModelParameters fastTree => new FeatureWeightsCalibratedModelParameters<FastTreeBinaryModelParameters, PlattCalibrator>(Env, fastTree, calibrator),
                FastForestBinaryModelParameters fastForest => new FeatureWeightsCalibratedModelParameters<FastForestBinaryModelParameters, PlattCalibrator>(Env, fastForest, calibrator),
                LightGbmBinaryModelParameters lightGbm => new FeatureWeightsCalibratedModelParameters<LightGbmBinaryModelParameters, PlattCalibrator>(Env, lightGbm, calibrator),
                _ => throw new ArgumentOutOfRangeException(nameof(model))
            };
        }

        private IPredictorProducing<float> RoundTripBinaryModel(IPredictorProducing<float> model, ICalibrator embeddedCalibrator = null)
        {
            using var stream = new MemoryStream();
            using (var writer = RepositoryWriter.CreateNew(stream, Env, useFileSystem: false))
            {
                ModelSaveContext.SaveModel(writer, model, "Predictor");
                if (embeddedCalibrator != null)
                    ModelSaveContext.SaveModel(writer, embeddedCalibrator, Path.Combine("Predictor", "Calibrator"));
                writer.Commit();
            }

            stream.Position = 0;
            using var reader = RepositoryReader.Open(stream, Env, useFileSystem: false);
            if (model is CalibratedModelParametersBase calibrated)
            {
                // Current-format archives have a bare predictor and its calibrator as siblings.
                ModelLoadContext.LoadModel<IPredictorProducing<float>, SignatureLoadModel>(
                    Env, out var subModel, reader, Path.Combine("Predictor", "Predictor"));
                Assert.Equal(calibrated.SubModel.GetType(), subModel.GetType());
                ModelLoadContext.LoadModel<ICalibrator, SignatureLoadModel>(
                    Env, out var calibrator, reader, Path.Combine("Predictor", "Calibrator"));
                Assert.Equal(calibrated.Calibrator.GetType(), calibrator.GetType());
            }

            ModelLoadContext.LoadModel<IPredictorProducing<float>, SignatureLoadModel>(Env, out var loaded, reader, "Predictor");
            return loaded;
        }

        private static void AssertCalibratedBinaryModel(IPredictorProducing<float> model, Type modelType,
            ICalibrator expectedCalibrator, Type wrapperType)
        {
            var calibrated = Assert.IsAssignableFrom<CalibratedModelParametersBase>(model);
            Assert.Equal(modelType, calibrated.SubModel.GetType());
            Assert.Equal(wrapperType, model.GetType().GetGenericTypeDefinition());
            Assert.IsAssignableFrom<IDistPredictorProducing<float, float>>(model);
            Assert.Equal(wrapperType == typeof(ParameterMixingCalibratedModelParameters<,>), model is IParameterMixer<float>);
            Assert.Equal(wrapperType != typeof(SchemaBindableCalibratedModelParameters<,>), model is IValueMapperDist);
            if (wrapperType == typeof(SchemaBindableCalibratedModelParameters<,>))
                Assert.IsAssignableFrom<ISchemaBindableMapper>(model);

            if (expectedCalibrator is PlattCalibrator platt)
            {
                var actual = Assert.IsType<PlattCalibrator>(calibrated.Calibrator);
                Assert.Equal(platt.Slope, actual.Slope);
                Assert.Equal(platt.Offset, actual.Offset);
            }
            else
            {
                var naive = Assert.IsType<NaiveCalibrator>(expectedCalibrator);
                var actual = Assert.IsType<NaiveCalibrator>(calibrated.Calibrator);
                Assert.Equal(naive.Min, actual.Min);
                Assert.Equal(naive.BinSize, actual.BinSize);
                Assert.Equal(naive.BinProbs, actual.BinProbs);
            }
        }

        private void AssertBinaryScores(IPredictorProducing<float> model, bool calibrated = false, bool useNaiveCalibrator = false)
        {
            var builder = new ArrayDataViewBuilder(Env);
            builder.AddColumn<float>("Features", NumberDataViewType.Single, new[] { -1f }, new[] { 1f });
            var data = builder.GetDataView();
            var bindable = ScoreUtils.GetSchemaBindableMapper(Env, model);
            var mapper = Assert.IsAssignableFrom<ISchemaBoundRowMapper>(
                bindable.Bind(Env, new RoleMappedSchema(data.Schema, label: null, feature: "Features")));
            Assert.Equal(calibrated, mapper.OutputSchema.TryGetColumnIndex("Probability", out int probabilityColumn));

            using var cursor = data.GetRowCursor(data.Schema);
            using var row = mapper.GetRow(cursor, mapper.OutputSchema);
            var scoreGetter = row.GetGetter<float>(row.Schema["Score"]);
            var probabilityGetter = calibrated ? row.GetGetter<float>(row.Schema[probabilityColumn]) : null;
            var expectedScores = new[] { -2.5f, 1.5f };
            var expectedProbabilities = useNaiveCalibrator
                ? new[] { 0.125f, 0.875f }
                : new[] { (float)(1 / (1 + Math.Exp(2.125))), (float)(1 / (1 + Math.Exp(-0.875))) };

            for (int i = 0; i < expectedScores.Length; i++)
            {
                Assert.True(cursor.MoveNext());
                float score = 0;
                scoreGetter(ref score);
                Assert.Equal(expectedScores[i], score);
                if (calibrated)
                {
                    float probability = 0;
                    probabilityGetter(ref probability);
                    Assert.Equal(expectedProbabilities[i], probability, precision: 6);
                }
            }
            Assert.False(cursor.MoveNext());
        }

        /// <summary>
        /// Features: x1, x2, x3, xRand; y = 10*x1 + 20x2 + 5.5x3 + e, xRand- random and Label y is to dependant on xRand.
        /// xRand has the least importance: Evaluation metrics do not change a lot when xRand is permuted.
        /// x2 has the biggest importance.
        /// </summary>
        private IDataView GetDenseDataset()
        {
            // Setup synthetic dataset.
            const int numberOfInstances = 1000;
            var rand = new Random(10);
            float[] yArray = new float[numberOfInstances];
            float[] x1Array = new float[numberOfInstances];
            float[] x2Array = new float[numberOfInstances];
            float[] x3Array = new float[numberOfInstances];
            float[] x4RandArray = new float[numberOfInstances];

            for (var i = 0; i < numberOfInstances; i++)
            {
                var x1 = rand.Next(1000);
                x1Array[i] = x1;
                var x2Important = rand.Next(10000);
                x2Array[i] = x2Important;
                var x3 = rand.Next(5000);
                x3Array[i] = x3;
                var x4Rand = rand.Next(1000);
                x4RandArray[i] = x4Rand;

                var noise = rand.Next(50);

                yArray[i] = (float)(10 * x1 + 20 * x2Important + 5.5 * x3 + noise);
            }

            GetBinaryClassificationLabels(yArray);

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("X1", NumberDataViewType.Single, x1Array);
            bldr.AddColumn("X2Important", NumberDataViewType.Single, x2Array);
            bldr.AddColumn("X3", NumberDataViewType.Single, x3Array);
            bldr.AddColumn("X4Rand", NumberDataViewType.Single, x4RandArray);
            bldr.AddColumn("Label", NumberDataViewType.Single, yArray);

            var srcDV = bldr.GetDataView();
            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2Important", "X3", "X4Rand")
                .Append(ML.Transforms.NormalizeMinMax("Features"));

            return pipeline.Append(ML.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Boolean))
                    .Fit(srcDV).Transform(srcDV);
        }

        private void GetBinaryClassificationLabels(float[] rawScores)
        {
            float averageScore = GetArrayAverage(rawScores);

            // Center the response and then take the sigmoid to generate the classes
            for (int i = 0; i < rawScores.Length; i++)
                rawScores[i] = MathUtils.Sigmoid(rawScores[i] - averageScore) > 0.5 ? 1 : 0;
        }

        private float GetArrayAverage(float[] scores)
        {
            // Compute the average so we can center the response
            float averageScore = 0.0f;
            for (int i = 0; i < scores.Length; i++)
                averageScore += scores[i];
            averageScore /= scores.Length;

            return averageScore;
        }
    }
}
