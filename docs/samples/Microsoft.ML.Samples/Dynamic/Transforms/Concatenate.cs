using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class Concatenate
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<InputData>()
            {
                new InputData(){ Feature1 = 0.1f, Feature2 = new[]{ 1.1f, 2.1f,
                    3.1f }, Feature3 = 1 },

                new InputData(){ Feature1 = 0.2f, Feature2 = new[]{ 1.2f, 2.2f,
                    3.2f }, Feature3 = 2 },

                new InputData(){ Feature1 = 0.3f, Feature2 = new[]{ 1.3f, 2.3f,
                    3.3f }, Feature3 = 3 },

                new InputData(){ Feature1 = 0.4f, Feature2 = new[]{ 1.4f, 2.4f,
                    3.4f }, Feature3 = 4 },

                new InputData(){ Feature1 = 0.5f, Feature2 = new[]{ 1.5f, 2.5f,
                    3.5f }, Feature3 = 5 },

                new InputData(){ Feature1 = 0.6f, Feature2 = new[]{ 1.6f, 2.6f,
                    3.6f }, Feature3 = 6 },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for concatenating the "Feature1", "Feature2" and
            // "Feature3" columns together into a vector that will be the Features
            // column. Concatenation is necessary because trainers take feature
            // vectors as inputs.
            //
            // Please note that the "Feature3" column is converted from int32 to
            // float using the ConvertType. The Concatenate requires all columns to
            // be of same type.
            var pipeline = mlContext.Transforms.Conversion.ConvertType("Feature3",
                outputKind: DataKind.Single)
                .Append(mlContext.Transforms.Concatenate("Features", new[]
                    { "Feature1", "Feature2", "Feature3" }));

            // The transformed data.
            var transformedData = pipeline.Fit(dataview).Transform(dataview);

            // Now let's take a look at what this concatenation did.
            // We can extract the newly created column as an IEnumerable of
            // TransformedData.
            var featuresColumn = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false);

            // And we can write out a few rows
            Console.WriteLine($"Features column obtained post-transformation.");
            foreach (var featureRow in featuresColumn)
                Console.WriteLine(string.Join(" ", featureRow.Features));

            // Expected output:
            //  Features column obtained post-transformation.
            //  0.1 1.1 2.1 3.1 1
            //  0.2 1.2 2.2 3.2 2
            //  0.3 1.3 2.3 3.3 3
            //  0.4 1.4 2.4 3.4 4
            //  0.5 1.5 2.5 3.5 5
            //  0.6 1.6 2.6 3.6 6
        }

        private class InputData
        {
            public float Feature1;
            [VectorType(3)]
            public float[] Feature2;
            public int Feature3;
        }

        private sealed class TransformedData
        {
            public float[] Features { get; set; }
        }
    }
}
