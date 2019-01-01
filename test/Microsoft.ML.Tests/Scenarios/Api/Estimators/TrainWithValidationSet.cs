// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.RunTests;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Train with validation set: Similar to the simple train scenario, but also support a validation set.
        /// The learner might be trees with early stopping.
        /// </summary>
        [Fact]
        public void New_TrainWithValidationSet()
        {
            var ml = new MLContext(seed: 1, conc: 1);
            // Pipeline.
            var data = ml.Data.ReadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.trainFilename), hasHeader: true);
            var pipeline = ml.Transforms.Text.FeaturizeText("SentimentText", "Features");

            // Train the pipeline, prepare train and validation set.
            var preprocess = pipeline.Fit(data);
            var trainData = preprocess.Transform(data);
            var validDataSource = ml.Data.ReadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.testFilename), hasHeader: true);
            var validData = preprocess.Transform(validDataSource);

            // Train model with validation set.
            var trainer = ml.BinaryClassification.Trainers.FastTree("Label","Features");
            var model = trainer.Train(ml.Data.Cache(trainData), ml.Data.Cache(validData));
        }
    }
}
