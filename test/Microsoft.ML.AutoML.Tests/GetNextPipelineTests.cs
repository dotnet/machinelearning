// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Newtonsoft.Json;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{

    public class GetNextPipelineTests : BaseTestClass
    {
        public GetNextPipelineTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void GetNextPipeline()
        {
            var context = new MLContext(1);
            var uciAdult = DatasetUtil.GetUciAdultDataView();
            var columns = DatasetColumnInfoUtil.GetDatasetColumnInfo(context, uciAdult, new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel });

            // get next pipeline
            var pipeline = PipelineSuggester.GetNextPipeline(context, new List<PipelineScore>(), columns,
                TaskKind.BinaryClassification, ((IChannelProvider)context).Start("AutoMLTest"));

            // serialize & deserialize pipeline
            var serialized = JsonConvert.SerializeObject(pipeline);
            Console.WriteLine(serialized);
            var deserialized = JsonConvert.DeserializeObject<Pipeline>(serialized);

            // run pipeline
            var estimator = deserialized.ToEstimator(context);
            var scoredData = estimator.Fit(uciAdult).Transform(uciAdult);
            var score = context.BinaryClassification.EvaluateNonCalibrated(scoredData).Accuracy;
            var result = new PipelineScore(deserialized, score, true);

            Assert.NotNull(result);
        }

        [Fact]
        public void GetNextPipelineMock()
        {
            var context = new MLContext(1);
            var uciAdult = DatasetUtil.GetUciAdultDataView();
            var columns = DatasetColumnInfoUtil.GetDatasetColumnInfo(context, uciAdult, new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel });

            // Get next pipeline loop
            var history = new List<PipelineScore>();
            var task = TaskKind.BinaryClassification;
            var maxIterations = 60;
            for (var i = 0; i < maxIterations; i++)
            {
                // Get next pipeline
                var pipeline = PipelineSuggester.GetNextPipeline(context, history, columns, task, ((IChannelProvider)context).Start("AutoMLTest"));
                if (pipeline == null)
                {
                    break;
                }

                var result = new PipelineScore(pipeline, AutoMlUtils.Random.Value.NextDouble(), true);
                history.Add(result);
            }

            Assert.Equal(maxIterations, history.Count);

            // Get all 'Stage 1' and 'Stage 2' runs from Pipeline Suggester
            var allAvailableTrainers = RecipeInference.AllowedTrainers(context, task, new ColumnInformation(), null);
            var stage1Runs = history.Take(allAvailableTrainers.Count());
            var stage2Runs = history.Skip(allAvailableTrainers.Count());

            // Get the trainer names from top 3 Stage 1 runs
            var topStage1Runs = stage1Runs.OrderByDescending(r => r.Score).Take(3);
            var topStage1TrainerNames = topStage1Runs.Select(r => r.Pipeline.Nodes.Last().Name);

            // Get unique trainer names from Stage 2 runs
            var stage2TrainerNames = stage2Runs.Select(r => r.Pipeline.Nodes.Last().Name).Distinct();

            // Assert that are only 3 unique trainers used in stage 2
            Assert.Equal(3, stage2TrainerNames.Count());
            // Assert that all trainers in stage 2 were the top trainers from stage 1
            Assert.False(topStage1TrainerNames.Except(stage2TrainerNames).Any());
        }
    }
}
