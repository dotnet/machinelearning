// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
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
            var pipeline = new Legacy.LearningPipeline();

            pipeline.Add(new TextLoader(dataPath).CreateFrom<SentimentData>());
            pipeline.Add(MakeSentimentTextTransform());
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            var testLearningPipelineItem = new TextLoader(testDataPath).CreateFrom<SentimentData>();
            var evaluator = new BinaryClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testLearningPipelineItem);
        }
    }
}
