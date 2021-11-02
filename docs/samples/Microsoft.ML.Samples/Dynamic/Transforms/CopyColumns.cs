using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class CopyColumns
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<InputData>()
            {
                new InputData(){ ImageId = 1, Features = new [] { 1.0f, 1.0f,
                    1.0f } },

                new InputData(){ ImageId = 2, Features = new [] { 2.0f, 2.0f,
                    2.0f } },

                new InputData(){ ImageId = 3, Features = new [] { 3.0f, 3.0f,
                    3.0f } },

                new InputData(){ ImageId = 4, Features = new [] { 4.0f, 4.0f,
                    4.0f } },

                new InputData(){ ImageId = 5, Features = new [] { 5.0f, 5.0f,
                    5.0f } },

                new InputData(){ ImageId = 6, Features = new [] { 6.0f, 6.0f,
                    6.0f } },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // CopyColumns is commonly used to rename columns.
            // For example, if you want to train towards ImageId, and your trainer
            // expects a "Label" column, you can use CopyColumns to rename ImageId
            // to Label. Technically, the ImageId column still exists, but it won't
            // be materialized unless you actually need it somewhere (e.g. if you
            // were to save the transformed data without explicitly dropping the
            // column). This is a general property of IDataView's lazy evaluation.
            var pipeline = mlContext.Transforms.CopyColumns("Label", "ImageId");

            // Now we can transform the data and look at the output to confirm the
            // behavior of CopyColumns. Don't forget that this operation doesn't
            // actually evaluate data until we read the data below.
            var transformedData = pipeline.Fit(dataview).Transform(dataview);

            // We can extract the newly created column as an IEnumerable of
            // SampleInfertDataTransformed, the class we define below.
            var rowEnumerable = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false);

            // And finally, we can write out the rows of the dataset, looking at the
            // columns of interest.
            Console.WriteLine($"Label and ImageId columns obtained " +
                $"post-transformation.");

            foreach (var row in rowEnumerable)
                Console.WriteLine($"Label: {row.Label} ImageId: {row.ImageId}");

            // Expected output:
            // ImageId and Label columns obtained post-transformation.
            //  Label: 1 ImageId: 1
            //  Label: 2 ImageId: 2
            //  Label: 3 ImageId: 3
            //  Label: 4 ImageId: 4
            //  Label: 5 ImageId: 5
            //  Label: 6 ImageId: 6
        }

        private class InputData
        {
            public int ImageId { get; set; }
            public float[] Features { get; set; }
        }

        private class TransformedData : InputData
        {
            public int Label { get; set; }
        }
    }
}
