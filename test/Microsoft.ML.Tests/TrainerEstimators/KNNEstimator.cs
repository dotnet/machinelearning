// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void TestEstimatorMultiClassKnn()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetMultiClassPipeline();
            var trainer = new MultiClassNearestNeighborTrainer(Env, "Label", "Features");
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView);
            trainer.Train(transformedDataView, model.Model);
            Done();
        }

        [Fact]
        public void TestEstimatorRegressionKnn()
        {
            var dataView = GetRegressionPipeline();
            var trainer = new RegressionNearestNeighborTrainer(Env, "Label", "Features");
            TestEstimatorCore(trainer, dataView);

            var model = trainer.Fit(dataView);
            trainer.Train(dataView, model.Model);
            Done();
        }


        [Fact]
        public void TestEstimatorBinaryKnn()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();
            var trainer = new BinaryNearestNeighborTrainer(Env, "Label", "Features");
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView);
            trainer.Train(transformedDataView, model.Model);
            Done();
        }
    }
}