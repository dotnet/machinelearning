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
        /// Train with initial predictor: Similar to the simple train scenario, but also accept a pre-trained initial model.
        /// The scenario might be one of the online linear learners that can take advantage of this, e.g., averaged perceptron.
        /// </summary>
        [Fact]
        public void New_TrainWithInitialPredictor()
        {
            var dataPath = GetDataPath(SentimentDataPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                var data = new TextLoader(env, MakeSentimentTextLoaderArgs()).Read(new MultiFileSource(dataPath));

                // Pipeline.
                var pipeline = new MyTextTransform(env, MakeSentimentTextTransformArgs());

                // Train the pipeline, prepare train set.
                var trainData = pipeline.FitAndTransform(data);

                // Train the first predictor.
                var trainer = new MySdca(env, new LinearClassificationTrainer.Arguments
                {
                    NumThreads = 1
                }, "Features", "Label");
                var firstModel = trainer.Fit(trainData);

                // Train the second predictor on the same data.
                var secondTrainer = new MyAveragedPerceptron(env, new AveragedPerceptronTrainer.Arguments(), "Features", "Label");
                var finalModel = secondTrainer.Train(trainData, firstModel.InnerModel);
            }
        }
    }
}
