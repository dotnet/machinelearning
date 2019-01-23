using System;
using System.Collections.Generic;
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
            var columns = AutoMlUtils.GetColumnInfoTuples(context, uciAdult, DatasetUtil.UciAdultLabel, null);

            // get next pipeline
            var pipeline = PipelineSuggester.GetNextPipeline(new List<PipelineRunResult>(), columns, TaskKind.BinaryClassification, 5);

            // serialize & deserialize pipeline
            var serialized = JsonConvert.SerializeObject(pipeline);
            Console.WriteLine(serialized);
            var deserialized = JsonConvert.DeserializeObject<Pipeline>(serialized);

            // run pipeline
            var estimator = deserialized.ToEstimator();
            var scoredData = estimator.Fit(uciAdult).Transform(uciAdult);
            var score = context.BinaryClassification.EvaluateNonCalibrated(scoredData).Accuracy;
            var result = new PipelineRunResult(deserialized, score, true);

            Assert.IsNotNull(result);
        }

        [TestMethod]
        public void GetNextPipelineMock()
        {
            var context = new MLContext();
            var uciAdult = DatasetUtil.GetUciAdultDataView();
            var columns = AutoMlUtils.GetColumnInfoTuples(context, uciAdult, DatasetUtil.UciAdultLabel, null);

            // get next pipeline loop
            var history = new List<PipelineRunResult>();
            var maxIterations = 10;
            for (var i = 0; i < maxIterations; i++)
            {
                // get next pipeline
                var pipeline = PipelineSuggester.GetNextPipeline(history, columns, TaskKind.BinaryClassification, maxIterations - i);

                var result = new PipelineRunResult(pipeline, AutoMlUtils.Random.NextDouble(), true);
                history.Add(result);
            }

            Assert.AreEqual(maxIterations, history.Count);
        }
    }
}
