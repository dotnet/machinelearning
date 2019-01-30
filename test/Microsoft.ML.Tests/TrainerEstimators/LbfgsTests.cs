// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Learners;
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
            trainer.Train(transformedDataView, model.Model);
            Done();
        }

        [Fact]
        public void TestEstimatorMulticlassLogisticRegression()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetMultiClassPipeline();
            var trainer = ML.MulticlassClassification.Trainers.LogisticRegression();
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView);
            trainer.Train(transformedDataView, model.Model);
            Done();
        }

        [Fact]
        public void TestEstimatorPoissonRegression()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.PoissonRegression();
            TestEstimatorCore(trainer, dataView);

            var model = trainer.Fit(dataView);
            trainer.Train(dataView, model.Model);
            Done();
        }

        [Fact]
        public void TestLogisticRegressionNoStats()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();

            pipe = pipe.Append(ML.BinaryClassification.Trainers.LogisticRegression(new LogisticRegression.Options { ShowTrainingStats = true }));
            var transformerChain = pipe.Fit(dataView) as TransformerChain<BinaryPredictionTransformer<ParameterMixingCalibratedPredictor>>;

            var linearModel = transformerChain.LastTransformer.Model.SubPredictor as LinearBinaryModelParameters;
            var stats = linearModel.Statistics;
            LinearModelStatistics.TryGetBiasStatistics(stats, 2, out float stdError, out float zScore, out float pValue);

            Assert.Equal(0f, stdError);
            Assert.Equal(0f, zScore);
        }

        [Fact]
        public void TestLogisticRegressionWithStats()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();

            pipe = pipe.Append(ML.BinaryClassification.Trainers.LogisticRegression(
                new LogisticRegression.Options { 
                    ShowTrainingStats = true,
                    StdComputer = new ComputeLRTrainingStdThroughHal(),
            }));

            var transformer = pipe.Fit(dataView) as TransformerChain<BinaryPredictionTransformer<ParameterMixingCalibratedPredictor>>;

            var linearModel = transformer.LastTransformer.Model.SubPredictor as LinearBinaryModelParameters;
            var stats = linearModel.Statistics;
            LinearModelStatistics.TryGetBiasStatistics(stats, 2, out float stdError, out float zScore, out float pValue);

            CompareNumbersWithTolerance(stdError, 0.250672936);
            CompareNumbersWithTolerance(zScore, 7.97852373);

            var scoredData = transformer.Transform(dataView);

            var coeffcients  = stats.GetCoefficientStatistics(linearModel, scoredData.Schema["Features"], 100);

            Assert.Equal(19, coeffcients.Length);

            foreach(var coefficient in coeffcients)
                Assert.True(coefficient.StandardError < 1.0);

        }
    }
}
