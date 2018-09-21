// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.TextAnalytics;
using System.Collections.Generic;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{

    public partial class ApiScenariosTests
    {
        private TOut GetValue<TOut>(Dictionary<string, object> keyValues, string key)
        {
            if (keyValues.ContainsKey(key))
                return (TOut)keyValues[key];

            return default(TOut);
        }

        /// <summary>
        /// Introspective training: Models that produce outputs and are otherwise black boxes are of limited use;
        /// it is also necessary often to understand at least to some degree what was learnt. To outline critical
        /// scenarios that have come up multiple times:
        ///  *) When I train a linear model, I should be able to inspect coefficients.
        ///  *) The tree ensemble learners, I should be able to inspect the trees.
        ///  *) The LDA transform, I should be able to inspect the topics.
        ///  I view it as essential from a usability perspective that this be discoverable to someone without 
        ///  having to read documentation.E.g.: if I have var lda = new LdaTransform().Fit(data)(I don't insist on that
        ///  exact signature, just giving the idea), then if I were to type lda.
        ///  In Visual Studio, one of the auto-complete targets should be something like GetTopics.
        /// </summary>

        [Fact]
        public void IntrospectiveTraining()
        {

            using (var env = new LocalEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = TextLoader.ReadFile(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename)));

                var words = WordBagTransform.Create(env, new WordBagTransform.Arguments()
                {
                    NgramLength = 1,
                    Column = new[] { new WordBagTransform.Column() { Name = "Tokenize", Source = new[] { "SentimentText" } } }
                }, loader);

                var lda = new LdaTransform(env, new LdaTransform.Arguments()
                {
                    NumTopic = 10,
                    NumIterations = 3,
                    NumThreads = 1,
                    Column = new[] { new LdaTransform.Column { Source = "Tokenize", Name = "Features"}
                    }
                }, words);
                var trainData = lda;

                var cachedTrain = new CacheDataView(env, trainData, prefetch: null);
                // Train the first predictor.
                var linearTrainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments
                {
                    NumThreads = 1
                });
                var trainRoles = new RoleMappedData(cachedTrain, label: "Label", feature: "Features");
                var linearPredictor = linearTrainer.Train(new Runtime.TrainContext(trainRoles));
                VBuffer<float> weights = default;
                linearPredictor.GetFeatureWeights(ref weights);

                var topicSummary = lda.GetTopicSummary();
                var treeTrainer = new FastTreeBinaryClassificationTrainer(env, DefaultColumnNames.Label, DefaultColumnNames.Features,
                    advancedSettings: s =>{ s.NumTrees = 2; });
                var ftPredictor = treeTrainer.Train(new Runtime.TrainContext(trainRoles));
                FastTreeBinaryPredictor treePredictor;
                if (ftPredictor is CalibratedPredictorBase calibrator)
                    treePredictor = (FastTreeBinaryPredictor)calibrator.SubPredictor;
                else
                    treePredictor = (FastTreeBinaryPredictor)ftPredictor;
                var featureNameCollection = FeatureNameCollection.Create(trainRoles.Schema);
                foreach (var tree in treePredictor.GetTrees())
                {
                    var lteChild = tree.LteChild;
                    var gteChild = tree.GtChild;
                    // Get nodes.
                    for (var i = 0; i < tree.NumNodes; i++)
                    {
                        var node = tree.GetNode(i, false, featureNameCollection);
                        var gainValue = GetValue<double>(node.KeyValues, "GainValue");
                        var splitGain = GetValue<double>(node.KeyValues, "SplitGain");
                        var featureName = GetValue<string>(node.KeyValues, "SplitName");
                        var previousLeafValue = GetValue<double>(node.KeyValues, "PreviousLeafValue");
                        var threshold = GetValue<string>(node.KeyValues, "Threshold").Split(new[] { ' ' }, 2)[1];
                        var nodeIndex = i;
                    }
                    // Get leaves.
                    for (var i = 0; i < tree.NumLeaves; i++)
                    {
                        var node = tree.GetNode(i, true, featureNameCollection);
                        var leafValue = GetValue<double>(node.KeyValues, "LeafValue");
                        var extras = GetValue<string>(node.KeyValues, "Extras");
                        var nodeIndex = ~i;
                    }
                }
            }
        }
    }
}
