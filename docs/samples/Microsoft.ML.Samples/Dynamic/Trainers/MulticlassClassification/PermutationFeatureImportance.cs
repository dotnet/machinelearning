using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic.Trainers.MulticlassClassification
{
    public static class PermutationFeatureImportance
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed:1);

            // Create sample data.
            var samples = Data.GenerateData();

            // Load the sample data as an IDataView.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Define a training pipeline that concatenates features into a vector, normalizes them, and then
            // trains a linear model.
            var pipeline = mlContext.Transforms.Concatenate("Features", Data.FeatureColumns)
                    .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy());

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Compute the permutation metrics for the linear model using the normalized data.
            var transformedData = model.Transform(data);
            var linearPredictor = model.LastTransformer;
            var permutationMetrics = mlContext.MulticlassClassification.PermutationFeatureImportance(
                linearPredictor, transformedData, permutationCount: 30);

            // Now let's look at which features are most important to the model overall.
            // Get the feature indices sorted by their impact on microaccuracy.
            var sortedIndices = permutationMetrics.Select((metrics, index) => new { index, metrics.MicroAccuracy})
                .OrderByDescending(feature => Math.Abs(feature.MicroAccuracy.Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature\tChange in MicroAccuracy\t95% Confidence in the Mean Change in MicroAccuracy");
            var microAccuracy = permutationMetrics.Select(x => x.MicroAccuracy).ToArray();
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}\t{1:G4}\t{2:G4}",
                    Data.FeatureColumns[i],
                    microAccuracy[i].Mean,
                    1.96 * microAccuracy[i].StandardError);
            }

            // Expected output:
            //Feature     Change in MicroAccuracy  95% Confidence in the Mean Change in MicroAccuracy
            //Feature2     -0.1395                 0.0006567
            //Feature1     -0.05367                0.0006908
        }

        private class Data
        {
            public float Label { get; set; }

            public float Feature1 { get; set; }

            public float Feature2 { get; set; }

            public static readonly string[] FeatureColumns = new string[] { nameof(Feature1), nameof(Feature2) };

            public static IEnumerable<Data> GenerateData(int nExamples = 10000,
                double bias = 0, double weight1 = 1, double weight2 = 2, int seed = 1)
            {
                var rng = new Random(seed);
                var max = bias + 4.5*weight1 + 4.5*weight2 + 0.5;
                for (int i = 0; i < nExamples; i++)
                {
                    var data = new Data
                    {
                        Feature1 = (float)(rng.Next(10) * (rng.NextDouble() - 0.5)),
                        Feature2 = (float)(rng.Next(10) * (rng.NextDouble() - 0.5)),
                    };

                    // Create a noisy label.
                    var value = (float)(bias + weight1 * data.Feature1 + weight2 * data.Feature2 + rng.NextDouble() - 0.5);
                    if (value < max / 3)
                        data.Label = 0;
                    else if (value < 2 * max / 3)
                        data.Label = 1;
                    else
                        data.Label = 2;
                    yield return data;
                }
            }
        }
    }
}
