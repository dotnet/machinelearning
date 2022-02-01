// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void OnlineLinearWorkout()
        {
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);

            var regressionData = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("Label", DataKind.Single, 0),
                new TextLoader.Column("Features", DataKind.Single, 1, 10)
            });

            var regressionPipe = ML.Transforms.NormalizeMinMax("Features");

            var regressionTrainData = regressionPipe.Fit(regressionData).Transform(regressionData);

            var ogdTrainer = ML.Regression.Trainers.OnlineGradientDescent();
            TestEstimatorCore(ogdTrainer, regressionTrainData);
            var ogdModel = ogdTrainer.Fit(regressionTrainData);
            ogdTrainer.Fit(regressionTrainData, ogdModel.Model);

            var binaryData = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("Label", DataKind.Boolean, 0),
                new TextLoader.Column("Features", DataKind.Single, 1, 10)
            });

            var binaryPipe = ML.Transforms.NormalizeMinMax("Features");

            var binaryTrainData = binaryPipe.Fit(binaryData).Transform(binaryData);
            var apTrainer = ML.BinaryClassification.Trainers.AveragedPerceptron(
                new AveragedPerceptronTrainer.Options { LearningRate = 0.5f });
            TestEstimatorCore(apTrainer, binaryTrainData);

            var apModel = apTrainer.Fit(binaryTrainData);
            apTrainer.Fit(binaryTrainData, apModel.Model);

            var svmTrainer = ML.BinaryClassification.Trainers.LinearSvm();
            TestEstimatorCore(svmTrainer, binaryTrainData);

            var svmModel = svmTrainer.Fit(binaryTrainData);
            svmTrainer.Fit(binaryTrainData, apModel.Model);

            Done();

        }
    }
}
