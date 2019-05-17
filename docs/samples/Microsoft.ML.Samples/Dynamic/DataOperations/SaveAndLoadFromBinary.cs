using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class SaveAndLoadFromBinary
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            IEnumerable<DataPoint> dataPoints = GenerateRandomDataPoints(10);

            // Convert the list of data points to an IDataView object, which is consumable by ML.NET API.
            IDataView data = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Inspect the data before saving to a binary file.
            PrintPreviewRows(dataPoints);

            // The rows in the data.
            // 0, 0.7262433
            // 1, 0.8173254
            // 0, 0.7680227
            // 1, 0.5581612
            // 0, 0.2060332
            // 1, 0.5588848
            // 0, 0.9060271
            // 1, 0.4421779
            // 0, 0.9775497
            // 1, 0.2737045

            // Create a FileStream object and write the IDataView to it as a binary IDV file. 
            using (FileStream stream = new FileStream("data.idv", FileMode.Create))
            {
                mlContext.Data.SaveAsBinary(data, stream);
            }

            // Create an IDataView object by loading the binary IDV file.
            IDataView loadedData = mlContext.Data.LoadFromBinary("data.idv");

            // Inspect the data that is loaded from the previously saved binary file.
            var loadedDataEnumerable = mlContext.Data.CreateEnumerable<DataPoint>(loadedData, reuseRowObject: false);
            PrintPreviewRows(loadedDataEnumerable);

            // The rows in the data.
            // 0, 0.7262433
            // 1, 0.8173254
            // 0, 0.7680227
            // 1, 0.5581612
            // 0, 0.2060332
            // 1, 0.5588848
            // 0, 0.9060271
            // 1, 0.4421779
            // 0, 0.9775497
            // 1, 0.2737045

            File.Delete("data.idv");
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count, int seed = 0)
        {
            var random = new Random(seed);
            for (int i = 0; i < count; i++)
            {
                yield return new DataPoint
                {
                    Label = i % 2,

                    // Create random features that are correlated with label.
                    Features = (float)random.NextDouble()
                };
            }
        }

        // Example with label and feature values. A data set is a collection of such examples.
        private class DataPoint
        {
            public float Label { get; set; }

            public float Features { get; set; }
        }

        // Print helper.
        private static void PrintPreviewRows(IEnumerable<DataPoint> data)
        {
            Console.WriteLine($"The rows in the data.");
            foreach (var row in data)
                Console.WriteLine($"{row.Label}, {row.Features}");
        }
    }
}
