// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Xunit;
using System.IO;
using Microsoft.ML.Model;
using System;

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
        public void TestLRNoStats()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();

            pipe = pipe.Append(ML.BinaryClassification.Trainers.LogisticRegression(new LogisticRegressionBinaryTrainer.Options { ShowTrainingStatistics = true }));
            var transformerChain = pipe.Fit(dataView) as TransformerChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>>;
            var linearModel = transformerChain.LastTransformer.Model.SubModel as LinearBinaryModelParameters;
            var stats = linearModel.Statistics as ModelStatisticsBase;

            Assert.NotNull(stats);

            var stats2 = linearModel.Statistics as LinearModelParameterStatistics;

            Assert.Null(stats2);

            Done();
        }


        [Fact]
        public void TestLRWithStats()
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

            Action<LinearBinaryModelParameters> validateStats = (modelParameters) =>
            {
                var stats = linearModel.Statistics as LinearModelParameterStatistics;
                var biasStats = stats?.GetBiasStatistics();
                Assert.NotNull(biasStats);

                biasStats = stats.GetBiasStatisticsForValue(2);

                Assert.NotNull(biasStats);

                CompareNumbersWithTolerance(biasStats.StandardError, 0.250672936);
                CompareNumbersWithTolerance(biasStats.ZScore, 7.97852373);

                var scoredData = transformer.Transform(dataView);

                var coefficients = stats.GetWeightsCoefficientStatistics(100);

                Assert.Equal(18, coefficients.Length);

                foreach (var coefficient in coefficients)
                    Assert.True(coefficient.StandardError < 1.0);
            };

            validateStats(linearModel);

            var modelPath = GetOutputPath("TestLRWithStats.zip");
            // Save model. 
            using (var file = File.Create(modelPath))
                transformer.SaveTo(ML, file);

            // Load model.
            TransformerChain<ITransformer> transformerChain;
            using (var file = File.OpenRead(modelPath))
                transformerChain = TransformerChain.LoadFrom(ML, file);

            // why can't this load like a BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
            var lastTransformer = transformerChain.LastTransformer as BinaryPredictionTransformer<IPredictorProducing<float>>;
            var model = lastTransformer.Model as ParameterMixingCalibratedModelParameters<IPredictorWithFeatureWeights<float>, ICalibrator>;

            linearModel = model.SubModel as LinearBinaryModelParameters;

            validateStats(linearModel);

            Done();

        }


        [Fact]
        public void TestLRWithStatsBackCompatibility()
        {
            string dropModelPath = GetDataPath("backcompat/LrWithStats.zip");
            string trainData = GetDataPath("adult.tiny.with-schema.txt");

            using (FileStream fs = File.OpenRead(dropModelPath))
            {
                var result = ModelFileUtils.LoadPredictorOrNull(Env, fs) as ParameterMixingCalibratedModelParameters<IPredictorWithFeatureWeights<float>, ICalibrator>;
                var subPredictor = result?.SubModel as LinearBinaryModelParameters;
                var stats = subPredictor?.Statistics;

                CompareNumbersWithTolerance(stats.Deviance, 458.970917);
                CompareNumbersWithTolerance(stats.NullDeviance, 539.276367);
                Assert.Equal(7, stats.ParametersCount);
                Assert.Equal(500, stats.TrainingExampleCount);

            }

            Done();
        }

        [Fact]
        public void TestMLRNoStats()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetMultiClassPipeline();
            var trainer = ML.MulticlassClassification.Trainers.LogisticRegression();
            var pipeWithTrainer = pipe.Append(trainer);

            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformer = pipeWithTrainer.Fit(dataView);
            var model = transformer.LastTransformer.Model as MulticlassLogisticRegressionModelParameters;
            var stats = model.Statistics;

            Assert.Null(stats);

            Done();
        }

        [Fact]
        public void TestMLRWithStats()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetMultiClassPipeline();

            var trainer = ML.MulticlassClassification.Trainers.LogisticRegression(new LogisticRegressionMulticlassClassificationTrainer.Options
            {
                ShowTrainingStatistics = true
            });
            var pipeWithTrainer = pipe.Append(trainer);

            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformer = pipeWithTrainer.Fit(dataView);
            var model = transformer.LastTransformer.Model as MulticlassLogisticRegressionModelParameters;

            Action<MulticlassLogisticRegressionModelParameters> validateStats = (modelParams) =>
            {
                var stats = modelParams.Statistics;
                Assert.NotNull(stats);

                CompareNumbersWithTolerance(stats.Deviance, 45.3556442);
                CompareNumbersWithTolerance(stats.NullDeviance, 329.583679199219);
                Assert.Equal(14, stats.ParametersCount);
                Assert.Equal(150, stats.TrainingExampleCount);
            };

            validateStats(model);

            var modelPath = GetOutputPath("TestMLRWithStats.zip");
            // Save model. 
            using (var file = File.Create(modelPath))
                transformer.SaveTo(ML, file);

            // Load model.
            TransformerChain<ITransformer> transformerChain;
            using (var file = File.OpenRead(modelPath))
                transformerChain = TransformerChain.LoadFrom(ML, file);

            var lastTransformer = transformerChain.LastTransformer as MulticlassPredictionTransformer<IPredictorProducing<VBuffer<float>>>;
            model = lastTransformer.Model as MulticlassLogisticRegressionModelParameters ;

            validateStats(model);

            Done();
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

            Done();
        }
    }
}
