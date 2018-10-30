// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.PipelineApi
{
    public partial class PipelineApiScenarioTests
    {
        /// <summary>
        /// Start with a dataset in a text file. Run text featurization on text values. 
        /// Train a linear model over that. (I am thinking sentiment classification.) 
        /// Out of the result, produce some structure over which you can get predictions programmatically 
        /// (for example, the prediction does not happen over a file as it did during training).
        /// </summary>
        [Fact]
        void SimpleTrainAndPredict()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentDataPath);
            var pipeline = new Legacy.LearningPipeline();

            pipeline.Add(new TextLoader(dataPath).CreateFrom<SentimentData>());

            pipeline.Add(MakeSentimentTextTransform());

            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            var singlePrediction = model.Predict(new SentimentData() { SentimentText = "Not big fan of this." });
            Assert.True(singlePrediction.Sentiment);
        }

        private static TextFeaturizer MakeSentimentTextTransform()
        {
            return new TextFeaturizer("Features", "SentimentText")
            {
                KeepPunctuations = false,
                OutputTokens = true,
                StopWordsRemover = new PredefinedStopWordsRemover(),
                VectorNormalizer = TextFeaturizingEstimatorTextNormKind.L2,
                CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 2, AllLengths = true }
            };
        }
    }
}
