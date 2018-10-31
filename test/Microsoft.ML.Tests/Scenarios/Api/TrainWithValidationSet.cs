﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms.Text;
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
        public void TrainWithValidationSet()
        {

            using (var env = new LocalEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = TextLoader.ReadFile(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename)));

                var trans = TextFeaturizingEstimator.Create(env, MakeSentimentTextTransformArgs(), loader);
                var trainData = trans;

                // Apply the same transformations on the validation set.
                // Sadly, there is no way to easily apply the same loader to different data, so we either have
                // to create another loader, or to save the loader to model file and then reload.

                // A new one is not always feasible, but this time it is.
                var validLoader = TextLoader.ReadFile(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(GetDataPath(TestDatasets.Sentiment.testFilename)));
                var validData = ApplyTransformUtils.ApplyAllTransformsToData(env, trainData, validLoader);

                // Cache both datasets.
                var cachedTrain = new CacheDataView(env, trainData, prefetch: null);
                var cachedValid = new CacheDataView(env, validData, prefetch: null);

                // Train.
                var trainer = new FastTreeBinaryClassificationTrainer(env, DefaultColumnNames.Label, DefaultColumnNames.Features, numTrees: 3);
                var trainRoles = new RoleMappedData(cachedTrain, label: "Label", feature: "Features");
                var validRoles = new RoleMappedData(cachedValid, label: "Label", feature: "Features");
                trainer.Train(new Runtime.TrainContext(trainRoles, validRoles));
            }
        }
    }
}
