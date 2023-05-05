// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
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

        #region Helpers
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
        #endregion
    }
}
