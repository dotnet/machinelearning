// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void TestEstimatorLogisticRegression()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();
            var trainer = ML.BinaryClassification.Trainers.LogisticRegression();
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView);
            trainer.Fit(transformedDataView, model.Model.SubModel);
            Done();
        }

        [Fact]
        public void TestEstimatorMulticlassLogisticRegression()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetMulticlassPipeline();
            var trainer = ML.MulticlassClassification.Trainers.LogisticRegression();
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView);
            trainer.Fit(transformedDataView, model.Model);
            Done();
        }

        [Fact]
        public void TestEstimatorPoissonRegression()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.PoissonRegression();
            TestEstimatorCore(trainer, dataView);

            var model = trainer.Fit(dataView);
            trainer.Fit(dataView, model.Model);
            Done();
        }

        [Fact]
        public void TestLogisticRegressionNoStats()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();

            pipe = pipe.Append(ML.BinaryClassification.Trainers.LogisticRegression(new LogisticRegressionBinaryTrainer.Options { ShowTrainingStatistics = true }));
            var transformerChain = pipe.Fit(dataView) as TransformerChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>>;

            var linearModel = transformerChain.LastTransformer.Model.SubModel as LinearBinaryModelParameters;
            var stats = linearModel.Statistics as ModelStatistics;

            Assert.NotNull(stats);

            var stats2 = linearModel.Statistics as LinearModelParameterStatistics;

            Assert.Null(stats2);
        }

        [Fact]
        public void TestLogisticRegressionWithStats()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();

            pipe = pipe.Append(ML.BinaryClassification.Trainers.LogisticRegression(
                new LogisticRegressionBinaryTrainer.Options
                {
                    ShowTrainingStatistics = true,
                    ComputeStandardDeviation = new ComputeLRTrainingStdThroughMkl(),
                }));

            var transformer = pipe.Fit(dataView) as TransformerChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>>;

            var linearModel = transformer.LastTransformer.Model.SubModel as LinearBinaryModelParameters;
            var stats = linearModel.Statistics as LinearModelParameterStatistics;

            var biasStats = stats?.GetBiasStatistics();
            Assert.NotNull(biasStats);

            biasStats = stats.GetBiasStatisticsForValue(2);

            Assert.NotNull(biasStats);

            CompareNumbersWithTolerance(biasStats.StandardError, 0.250672936);
            CompareNumbersWithTolerance(biasStats.ZScore, 7.97852373);

            var scoredData = transformer.Transform(dataView);

            var coefficients  = stats.GetCoefficientStatistics(scoredData.Schema["Features"], 100) ;

            Assert.Equal(19, coefficients.Length);

            foreach(var coefficient in coefficients)
                Assert.True(coefficient.StandardError < 1.0);

        }

        [Fact]
        public void TestLRWithStatsBackCompatibility()
        {
            string dropModelPath = GetDataPath("backcompat/LrWithStats.zip");
            string trainData = GetDataPath("adult.tiny.with-schema.txt");

            using (FileStream fs = File.OpenRead(dropModelPath))
            {
                var result = ModelFileUtils.LoadPredictorOrNull(Env, fs) as ParameterMixingCalibratedPredictor;

                var subPredictor = result?.SubPredictor as LinearBinaryModelParameters;
                var stats = subPredictor?.Statistics;

                Assert.Equal(458.970917f, stats.Deviance);
                Assert.Equal(539.276367f, stats.NullDeviance);
                Assert.Equal(7, stats.ParametersCount);
                Assert.Equal(500, stats.TrainingExampleCount);

            }
        }

        [Fact]
        public void TestMLRWithStatsBackCompatibility()
        {
            string dropModelPath = GetDataPath("backcompat/MlrWithStats.zip");
            string trainData = GetDataPath("iris.data");

            using (FileStream fs = File.OpenRead(dropModelPath))
            {
                var result = ModelFileUtils.LoadPredictorOrNull(Env, fs) as MulticlassLogisticRegressionModelParameters;
                var stats = result?.Statistics;

                Assert.Equal(132.012238f, stats.Deviance);
                Assert.Equal(329.583679f, stats.NullDeviance);
                Assert.Equal(11, stats.ParametersCount);
                Assert.Equal(150, stats.TrainingExampleCount);
            }
        }
    }
}
