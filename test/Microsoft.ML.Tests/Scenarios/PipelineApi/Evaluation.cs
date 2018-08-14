using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.PipelineApi
{
    public partial class PipelineApiScenarioTests
    {
        /// <summary>
        /// Evaluation: Similar to the simple train scenario, except instead of having some 
        /// predictive structure, be able to score another "test" data file, run the result 
        /// through an evaluator and get metrics like AUC, accuracy, PR curves, and whatnot. 
        /// Getting metrics out of this shoudl be as straightforward and unannoying as possible.
        /// </summary>
        [Fact]
        public void Evaluation()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentDataPath);
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(dataPath).CreateFrom<SentimentData>());
            pipeline.Add(MakeSentimentTextTransform());
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            var testLearningPipelineItem = new TextLoader(testDataPath).CreateFrom<SentimentData>();
            var evaluator = new BinaryClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testLearningPipelineItem);

            var singlePrediction = model.Predict(new SentimentData() { SentimentText = "Not big fan of this." });
            Assert.True(singlePrediction.Sentiment);
        }
    }
}
