using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;

namespace Samples.Dynamic
{
    public static class ToStringSingleColumn
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<InputData>()
            {
                new InputData(){ Feature1 = 0.1f, Feature2 = 1.1, Feature3 = 1 },

                new InputData(){ Feature1 = 0.2f, Feature2 =1.2, Feature3 = 2 },

                new InputData(){ Feature1 = 0.3f, Feature2 = 1.3, Feature3 = 3 },

                new InputData(){ Feature1 = 0.4f, Feature2 = 1.4, Feature3 = 4 },

                new InputData(){ Feature1 = 0.5f, Feature2 = 1.5, Feature3 = 5 },

                new InputData(){ Feature1 = 0.6f, Feature2 = 1.6, Feature3 = 6 },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for converting the "Feature1" column into its string representations
            //
            var pipeline = mlContext.Transforms.ToStringTransformer("Feature1Output", "Feature1");

            // The transformed data.
            var transformedData = pipeline.Fit(dataview).Transform(dataview);

            // Now let's take a look at what this did.
            // We can extract the newly created columns as an IEnumerable of
            // TransformedData.
            var featuresColumn = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false);

            // And we can write out a few rows
            Console.WriteLine($"Features column obtained post-transformation.");
            foreach (var featureRow in featuresColumn)
                Console.WriteLine(featureRow.Feature1Output);

            // Expected output:
            //  Features column obtained post-transformation.
            //  0.100000
            //  0.200000
            //  0.300000
            //  0.400000
            //  0.500000
            //  0.600000
        }

        private class InputData
        {
            public float Feature1;
            public double Feature2;
            public int Feature3;
        }

        private sealed class TransformedData
        {
            public float Feature1 { get; set; }
            public double Feature2 { get; set; }
            public int Feature3 { get; set; }
            public string Feature1Output { get; set; }
        }
    }
}
