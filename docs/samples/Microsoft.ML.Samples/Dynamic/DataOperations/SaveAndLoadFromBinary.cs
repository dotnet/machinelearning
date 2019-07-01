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
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            var dataPoints = new List<DataPoint>()
            {
                new DataPoint(){ Label = 0, Features = 4},
                new DataPoint(){ Label = 0, Features = 5},
                new DataPoint(){ Label = 0, Features = 6},
                new DataPoint(){ Label = 1, Features = 8},
                new DataPoint(){ Label = 1, Features = 9},
            };

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            IDataView data = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Create a FileStream object and write the IDataView to it as a binary
            // IDV file. 
            using (FileStream stream = new FileStream("data.idv", FileMode.Create))
                mlContext.Data.SaveAsBinary(data, stream);

            // Create an IDataView object by loading the binary IDV file.
            IDataView loadedData = mlContext.Data.LoadFromBinary("data.idv");

            // Inspect the data that is loaded from the previously saved binary file
            var loadedDataEnumerable = mlContext.Data
                .CreateEnumerable<DataPoint>(loadedData, reuseRowObject: false);

            foreach (DataPoint row in loadedDataEnumerable)
                Console.WriteLine($"{row.Label}, {row.Features}");

            // Preview of the loaded data.
            // 0, 4
            // 0, 5
            // 0, 6
            // 1, 8
            // 1, 9
        }

        // Example with label and feature values. A data set is a collection of such
        // examples.
        private class DataPoint
        {
            public float Label { get; set; }

            public float Features { get; set; }
        }
    }
}
