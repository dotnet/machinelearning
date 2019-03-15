using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic.Trainers.Regression
{
    public static class FastTree
    {
        // This example requires installation of additional NuGet package
        // <a href="https://www.nuget.org/packages/Microsoft.ML.FastTree/">Microsoft.ML.FastTree</a>.
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training examples.
            var examples = GenerateRandomDataPoints(1000);

            // Convert the examples list to an IDataView object, which is consumable by ML.NET API.
            var data = mlContext.Data.LoadFromEnumerable(examples);

            // Define the trainer.
            var pipeline = mlContext.BinaryClassification.Trainers.FastTree();

            // Train the model.
            var model = pipeline.Fit(data);
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count)
        {
            var random = new Random(0);
            float randomFloat() => (float)random.NextDouble();
            for (int i = 0; i < count; i++)
            {
                var label = randomFloat();
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with label.
                    Features = Enumerable.Repeat(label, 50).Select(x => x + randomFloat()).ToArray()
                };
            }
        }

        private class DataPoint
        {
            public float Label { get; set; }
            [VectorType(50)]
            public float[] Features { get; set; }
        }
    }
}