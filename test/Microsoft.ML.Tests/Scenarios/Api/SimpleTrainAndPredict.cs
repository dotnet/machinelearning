// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Xunit;
using System.Linq;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Start with a dataset in a text file. Run text featurization on text values. 
        /// Train a linear model over that. (I am thinking sentiment classification.) 
        /// Out of the result, produce some structure over which you can get predictions programmatically 
        /// (e.g., the prediction does not happen over a file as it did during training).
        /// </summary>
        [Fact]
        public void SimpleTrainAndPredict()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = new TextLoader(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(dataPath));

                var trans = TextTransform.Create(env, MakeSentimentTextTransformArgs(), loader);

                // Train
                var trainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments
                {
                    NumThreads = 1
                });

                var cached = new CacheDataView(env, trans, prefetch: null);
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));

                var scoreRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
                IDataScorerTransform scorer = ScoreUtils.GetScorer(predictor, scoreRoles, env, trainRoles.Schema);

                // Create prediction engine and test predictions.
                var model = env.CreatePredictionEngine<SentimentData, SentimentPrediction>(scorer);

                // Take a couple examples out of the test data and run predictions on top.
                var testLoader = new TextLoader(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(GetDataPath(SentimentTestPath)));
                var testData = testLoader.AsEnumerable<SentimentData>(env, false);
                foreach (var input in testData.Take(5))
                {
                    var prediction = model.Predict(input);
                    // Verify that predictions match and scores are separated from zero.
                    Assert.Equal(input.Sentiment, prediction.Sentiment);
                    Assert.True(input.Sentiment.IsTrue && prediction.Score > 1 || input.Sentiment.IsFalse && prediction.Score < -1);
                }
            }
        }

        private static TextTransform.Arguments MakeSentimentTextTransformArgs(bool normalize = true)
        {
            return new TextTransform.Arguments()
            {
                Column = new TextTransform.Column
                {
                    Name = "Features",
                    Source = new[] { "SentimentText" }
                },
                KeepDiacritics = false,
                KeepPunctuations = false,
                TextCase = Runtime.TextAnalytics.TextNormalizerTransform.CaseNormalizationMode.Lower,
                OutputTokens = true,
                StopWordsRemover = new Runtime.TextAnalytics.PredefinedStopWordsRemoverFactory(),
                VectorNormalizer = normalize ? TextTransform.TextNormKind.L2 : TextTransform.TextNormKind.None,
                CharFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 2, AllLengths = true },
            };
        }

        private static TextLoader.Arguments MakeIrisTextLoaderArgs()
        {

            return new TextLoader.Arguments()
            {
                Separator = "comma",
                HasHeader = true,
                Column = new[]
                    {
                        new TextLoader.Column()
                        {
                            Name = "SepalLength",
                            Source = new [] { new TextLoader.Range() { Min=0, Max=0} },
                            Type = DataKind.R4
                        },
                        new TextLoader.Column()
                        {
                            Name = "SepalWidth",
                            Source = new [] { new TextLoader.Range() { Min=1, Max=1} },
                            Type = DataKind.R4
                        },
                        new TextLoader.Column()
                        {
                            Name = "PetalLength",
                            Source = new [] { new TextLoader.Range() { Min=2, Max=2} },
                            Type = DataKind.R4
                        },
                        new TextLoader.Column()
                        {
                            Name = "PetalWidth",
                            Source = new [] { new TextLoader.Range() { Min=3, Max=3} },
                            Type = DataKind.R4
                        },
                        new TextLoader.Column()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoader.Range() { Min=4, Max=4} },
                            Type = DataKind.Text
                        }
                    }
            };
        }
        private static TextLoader.Arguments MakeSentimentTextLoaderArgs()
        {
            return new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                    {
                        new TextLoader.Column()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoader.Range() { Min=0, Max=0} },
                            Type = DataKind.BL
                        },

                        new TextLoader.Column()
                        {
                            Name = "SentimentText",
                            Source = new [] { new TextLoader.Range() { Min=1, Max=1} },
                            Type = DataKind.Text
                        }
                    }
            };
        }
    }
}
