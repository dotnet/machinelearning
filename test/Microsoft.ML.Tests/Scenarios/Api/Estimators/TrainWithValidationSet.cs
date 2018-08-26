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
                var pipeline = new MyTextLoader(env, MakeSentimentTextLoaderArgs())
                    .Append(new MyTextTransform(env, MakeSentimentTextTransformArgs()));

                // Train the pipeline, prepare train and validation set.
                var reader = pipeline.Fit(new MultiFileSource(dataPath));
                var trainData = reader.Read(new MultiFileSource(dataPath));
                var validData = reader.Read(new MultiFileSource(validationDataPath));

                // Train model with validation set.
                var trainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments(), "Features", "Label");
                var model = trainer.Train(trainData, validData);
            }
        }
    }
}
