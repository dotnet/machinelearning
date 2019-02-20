// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class GetNextPipelineTests
    {
        [TestMethod]
        public void GetNextPipeline()
        {
            var context = new MLContext();
            var uciAdult = DatasetUtil.GetUciAdultDataView();
            var columns = AutoMlUtils.GetColumnInfoTuples(context, uciAdult, new ColumnInformation() { LabelColumn = DatasetUtil.UciAdultLabel });

            // get next pipeline
            var pipeline = PipelineSuggester.GetNextPipeline(new List<PipelineScore>(), columns, TaskKind.BinaryClassification);

            // serialize & deserialize pipeline
            var serialized = JsonConvert.SerializeObject(pipeline);
            Console.WriteLine(serialized);
            var deserialized = JsonConvert.DeserializeObject<Pipeline>(serialized);

            // run pipeline
            var estimator = deserialized.ToEstimator();
            var scoredData = estimator.Fit(uciAdult).Transform(uciAdult);
            var score = context.BinaryClassification.EvaluateNonCalibrated(scoredData).Accuracy;
            var result = new PipelineScore(deserialized, score, true);

            Assert.IsNotNull(result);
        }

        [TestMethod]
        public void GetNextPipelineMock()
        {
            var context = new MLContext();
            var uciAdult = DatasetUtil.GetUciAdultDataView();
            var columns = AutoMlUtils.GetColumnInfoTuples(context, uciAdult, new ColumnInformation() { LabelColumn = DatasetUtil.UciAdultLabel });

            // Get next pipeline loop
            var history = new List<PipelineScore>();
            var task = TaskKind.BinaryClassification;
            var maxIterations = 60;
            for (var i = 0; i < maxIterations; i++)
            {
                // Get next pipeline
                var pipeline = PipelineSuggester.GetNextPipeline(history, columns, task);
                if (pipeline == null)
                {
                    break;
                }

                var result = new PipelineScore(pipeline, AutoMlUtils.Random.NextDouble(), true);
                history.Add(result);
            }
            
            Assert.AreEqual(maxIterations, history.Count);

            // Get all 'Stage 1' and 'Stage 2' runs from Pipeline Suggester
            var allAvailableTrainers = RecipeInference.AllowedTrainers(context, task, null);
            var stage1Runs = history.Take(allAvailableTrainers.Count());
            var stage2Runs = history.Skip(allAvailableTrainers.Count());

            // Get the trainer names from top 3 Stage 1 runs
            var topStage1Runs = stage1Runs.OrderByDescending(r => r.Score).Take(3);
            var topStage1TrainerNames = topStage1Runs.Select(r => r.Pipeline.Nodes.Last().Name);

            // Get unique trainer names from Stage 2 runs
            var stage2TrainerNames = stage2Runs.Select(r => r.Pipeline.Nodes.Last().Name).Distinct();

            // Assert that are only 3 unique trainers used in stage 2
            Assert.AreEqual(3, stage2TrainerNames.Count());
            // Assert that all trainers in stage 2 were the top trainers from stage 1
            Assert.IsFalse(topStage1TrainerNames.Except(stage2TrainerNames).Any());
        }
    }
}
