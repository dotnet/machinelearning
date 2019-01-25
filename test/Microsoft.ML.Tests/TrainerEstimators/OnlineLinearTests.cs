// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers.Online;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void OnlineLinearWorkout()
        {
            var dataPath = GetDataPath("breast-cancer.txt");

            var regressionData = TextLoaderStatic.CreateReader(ML, ctx => (Label: ctx.LoadFloat(0), Features: ctx.LoadFloat(1, 10)))
                .Read(dataPath);

            var regressionPipe = regressionData.MakeNewEstimator()
                .Append(r => (r.Label, Features: r.Features.Normalize()));

            var regressionTrainData = regressionPipe.Fit(regressionData).Transform(regressionData).AsDynamic;

            var ogdTrainer = ML.Regression.Trainers.OnlineGradientDescent();
            TestEstimatorCore(ogdTrainer, regressionTrainData);
            var ogdModel = ogdTrainer.Fit(regressionTrainData);
            ogdTrainer.Train(regressionTrainData, ogdModel.Model);

            var binaryData = TextLoaderStatic.CreateReader(ML, ctx => (Label: ctx.LoadBool(0), Features: ctx.LoadFloat(1, 10)))
               .Read(dataPath);

            var binaryPipe = binaryData.MakeNewEstimator()
                .Append(r => (r.Label, Features: r.Features.Normalize()));

            var binaryTrainData = binaryPipe.Fit(binaryData).Transform(binaryData).AsDynamic;
            var apTrainer = ML.BinaryClassification.Trainers.AveragedPerceptron(
                new AveragedPerceptronTrainer.Options{ LearningRate = 0.5f });
            TestEstimatorCore(apTrainer, binaryTrainData);

            var apModel = apTrainer.Fit(binaryTrainData);
            apTrainer.Train(binaryTrainData, apModel.Model);

            var svmTrainer = ML.BinaryClassification.Trainers.LinearSupportVectorMachines();
            TestEstimatorCore(svmTrainer, binaryTrainData);

            var svmModel = svmTrainer.Fit(binaryTrainData);
            svmTrainer.Train(binaryTrainData, apModel.Model);

            Done();

        }
    }
}
