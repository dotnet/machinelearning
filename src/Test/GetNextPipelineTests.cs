using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class GetNextPipelineTests
    {
        [Ignore]
        [TestMethod]
        public void GetNextPipeline()
        {
            var context = new MLContext();

            var uciAdult = DatasetUtil.GetUciAdultDataView();

            // get trainers & transforms
            var transforms = TransformInferenceApi.InferTransforms(context, uciAdult, DatasetUtil.UciAdultLabel);
            var availableTrainers = RecipeInference.AllowedTrainers(context, TaskKind.BinaryClassification, 4);

            // get next pipeline loop
            var history = new List<PipelineRunResult>();
            for (var i = 0; i < 2; i++)
            {
                // get next pipeline
                var pipeline = PipelineSuggester.GetNextPipeline(history, transforms, availableTrainers);
                var serialized = JsonConvert.SerializeObject(pipeline);
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
