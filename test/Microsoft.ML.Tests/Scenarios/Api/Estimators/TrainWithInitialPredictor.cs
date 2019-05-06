// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Train with initial predictor: Similar to the simple train scenario, but also accept a pre-trained initial model.
        /// The scenario might be one of the online linear learners that can take advantage of this, for example, averaged perceptron.
        /// </summary>
        [Fact]
        public void TrainWithInitialPredictor()
        {

            var ml = new MLContext(seed: 1);

            var data = ml.Data.LoadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.trainFilename), hasHeader: true);

            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("Features", "SentimentText");

            // Train the pipeline, prepare train set. Since it will be scanned multiple times in the subsequent trainer, we cache the 
            // transformed data in memory.
            var trainData = ml.Data.Cache(pipeline.Fit(data).Transform(data));

            // Train the first predictor.
            var trainer = ml.BinaryClassification.Trainers.SdcaNonCalibrated(
                new SdcaNonCalibratedBinaryTrainer.Options { NumberOfThreads = 1 });

            var firstModel = trainer.Fit(trainData);

            // Train the second predictor on the same data.
            var secondTrainer = ml.BinaryClassification.Trainers.AveragedPerceptron("Label","Features");

            var trainRoles = new RoleMappedData(trainData, label: "Label", feature: "Features");
            var finalModel = ((ITrainer)secondTrainer).Train(new TrainContext(trainRoles, initialPredictor: firstModel.Model));

        }
    }
}
