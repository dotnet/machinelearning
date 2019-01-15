using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Auto;

namespace InternalClient
{
    internal static class GetNextPipeline
    {
        private const string Label = "Label";
        private const string TrainDataPath = @"C:\data\sample_train2.csv";

        public static void Run()
        {
            // load data
            var context = new MLContext();
            var columnInference = context.Data.InferColumns(TrainDataPath, Label, true);
            var textLoader = context.Data.CreateTextReader(columnInference);
            var data = textLoader.Read(TrainDataPath);
            
            // get trainers & transforms
            var transforms = TransformInferenceApi.InferTransforms(context, data, Label);
            var availableTrainers = RecipeInference.AllowedTrainers(context, TaskKind.BinaryClassification, 4);

            // get next pipeline loop
            var history = new List<PipelineRunResult>();
            for(var i = 0; i < 100; i++)
            {
                // get next pipeline
                var pipeline = PipelineSuggester.GetNextPipeline(history, transforms, availableTrainers);
                if(pipeline == null)
                {
                    break;
                }
                Console.WriteLine($"{i}\t{pipeline}");

                // mock pipeline run
                var pipelineScore = AutoMlUtils.Random.NextDouble();
                var result = new PipelineRunResult(null, null, pipeline, pipelineScore, null);

                history.Add(result);
            }

            Console.ReadLine();
        }
    }
}
