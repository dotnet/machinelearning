// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
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
            var dataPath = GetDataPath(SentimentDataPath);
            var validationDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline.
                var reader = new TextLoader(env, MakeSentimentTextLoaderArgs());
                var pipeline = new TextTransform(env, "SentimentText", "Features");

                // Train the pipeline, prepare train and validation set.
                var data = reader.Read(new MultiFileSource(dataPath));
                var preprocess = pipeline.Fit(data);
                var trainData = preprocess.Transform(data);
                var validData = preprocess.Transform(reader.Read(new MultiFileSource(validationDataPath)));

                // Train model with validation set.
                var trainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments(), "Features", "Label");
                var model = trainer.Train(trainData, validData);
            }
        }
    }
}
