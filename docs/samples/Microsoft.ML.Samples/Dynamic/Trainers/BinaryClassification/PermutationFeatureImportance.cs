using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic.Trainers.BinaryClassification
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
                    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Compute the permutation metrics for the linear model using the normalized data.
            var transformedData = model.Transform(data);
            var linearPredictor = model.LastTransformer;
            var permutationMetrics = mlContext.BinaryClassification.PermutationFeatureImportance(
                linearPredictor, transformedData, permutationCount: 30);

            // Now let's look at which features are most important to the model overall.
            // Get the feature indices sorted by their impact on AUC.
            var sortedIndices = permutationMetrics.Select((metrics, index) => new { index, metrics.AreaUnderRocCurve})
                .OrderByDescending(feature => Math.Abs(feature.AreaUnderRocCurve.Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature\tModel Weight\tChange in AUC\t95% Confidence in the Mean Change in AUC");
            var auc = permutationMetrics.Select(x => x.AreaUnderRocCurve).ToArray();
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}\t{1:0.00}\t{2:G4}\t{3:G4}",
                    Data.FeatureColumns[i],
                    linearPredictor.Model.SubModel.Weights[i],
                    auc[i].Mean,
                    1.96 * auc[i].StandardError);
            }

            // Expected output:
            //  Feature     Model Weight Change in AUC  95% Confidence in the Mean Change in AUC
            //  Feature2        35.15     -0.387        0.002015
            //  Feature1        17.94     -0.1514       0.0008963
        }

        private class Data
        {
            public bool Label { get; set; }

            public float Feature1 { get; set; }

            public float Feature2 { get; set; }

            public static readonly string[] FeatureColumns = new string[] { nameof(Feature1), nameof(Feature2) };

            public static IEnumerable<Data> GenerateData(int nExamples = 10000,
                double bias = 0, double weight1 = 1, double weight2 = 2, int seed = 1)
            {
                var rng = new Random(seed);
                for (int i = 0; i < nExamples; i++)
                {
                    var data = new Data
                    {
                        Feature1 = (float)(rng.Next(10) * (rng.NextDouble() - 0.5)),
                        Feature2 = (float)(rng.Next(10) * (rng.NextDouble() - 0.5)),
                    };

                    // Create a noisy label.
                    var value = (float)(bias + weight1 * data.Feature1 + weight2 * data.Feature2 + rng.NextDouble() - 0.5);
                    data.Label = Sigmoid(value) > 0.5;
                    yield return data;
                }
            }

            private static double Sigmoid(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-1 * x));
            }
        }
    }
}
