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

            // get next pipeline loop
            var history = new List<PipelineRunResult>();
            var maxIterations = 2;
            for (var i = 0; i < maxIterations; i++)
            {
                // get next pipeline
                var pipeline = PipelineSuggester.GetNextPipeline(history, columns, TaskKind.BinaryClassification, maxIterations - i);
                var serialized = JsonConvert.SerializeObject(pipeline);
                Console.WriteLine(serialized);
                var deserialized = JsonConvert.DeserializeObject<Pipeline>(serialized);

                // run pipeline
                var estimator = deserialized.ToEstimator();
                var scoredData = estimator.Fit(uciAdult).Transform(uciAdult);
                var score = context.BinaryClassification.EvaluateNonCalibrated(scoredData).Accuracy;
                var result = new PipelineRunResult(deserialized, score, true);

                history.Add(result);
            }

            Assert.AreEqual(2, history.Count);
        }
    }
}
