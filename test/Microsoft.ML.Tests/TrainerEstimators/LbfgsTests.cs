// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
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
            pipe = pipe.Append(new LogisticRegression(Env, "Features", "Label"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestEstimatorMulticlassLogisticRegression()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetMultiClassPipeline();
            pipe = pipe.Append(new MulticlassLogisticRegression(Env, "Features", "Label"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestEstimatorPoissonRegression()
        {
            var dataView = GetRegressionPipeline();
            var pipe = new PoissonRegression(Env, "Features", "Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestLogisticRegressionStats()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();

            pipe = pipe.Append(new LogisticRegression(Env, "Features", "Label", advancedSettings: s => { s.ShowTrainingStats = true; }));
            var transformerChain = pipe.Fit(dataView) as TransformerChain<BinaryPredictionTransformer<ParameterMixingCalibratedPredictor>>;

            var linearModel = transformerChain.LastTransformer.Model.SubPredictor as LinearBinaryPredictor;
            var stats = linearModel.Statistics;
            LinearModelStatistics.TryGetBiasStatistics(stats, 2, out float stdError, out float zScore, out float pValue);

            CompareNumbersWithTolerance(stdError, 0.250672936);
            CompareNumbersWithTolerance(zScore, 7.97852373);
        }

        [Fact]
        public void TestLogisticRegressionStats_MKL()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();

            pipe = pipe.Append(new LogisticRegression(Env, "Features", "Label", advancedSettings: s => {
                s.ShowTrainingStats = true;
                s.StdComputer = new ComputeLRTrainingStdThroughHal();
            }));

            var transformerChain = pipe.Fit(dataView) as TransformerChain<BinaryPredictionTransformer<ParameterMixingCalibratedPredictor>>;

            var linearModel = transformerChain.LastTransformer.Model.SubPredictor as LinearBinaryPredictor;
            var stats = linearModel.Statistics;
            LinearModelStatistics.TryGetBiasStatistics(stats, 2, out float stdError, out float zScore, out float pValue);

            CompareNumbersWithTolerance(stdError, 0.250672936);
            CompareNumbersWithTolerance(zScore, 7.97852373);
        }
    }
}
