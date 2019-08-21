using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;

namespace Samples.Dynamic
{
    public static class RobustScalerWithScale
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<InputData>()
            {
                new InputData(){ Feature1 = 1f },

                new InputData(){ Feature1 = 3f },

                new InputData(){ Feature1 = 5f },

                new InputData(){ Feature1 = 7f },

                new InputData(){ Feature1 = 9f },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for scaling the feature1 column
            var pipeline = mlContext.Transforms.RobustScalerTransformer("Feature1", center: false);

            // The transformed data.
            var transformedData = pipeline.Fit(dataview).Transform(dataview);

            // Now let's take a look at what this did. The values should be scaled by the range * ratio.
            // We can extract the newly created columns as an IEnumerable of TransformedData.
            var featuresColumn = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false);

            // And we can write out a few rows
            Console.WriteLine($"Features column obtained post-transformation.");
            foreach (var featureRow in featuresColumn)
                Console.WriteLine(featureRow.Feature1);

            // Expected output:
            //  Features column obtained post-transformation.
            //  0.25
            //  .75
            //  1.25
            //  1.75
            //  2.25
        }

        private class InputData
        {
            public float Feature1;
        }

        private sealed class TransformedData
        {
            public float Feature1 { get; set; }
        }
    }
}
