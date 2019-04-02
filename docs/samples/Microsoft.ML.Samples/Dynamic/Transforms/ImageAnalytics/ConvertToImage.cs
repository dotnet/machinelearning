using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class ConvertToImage
    {
        private const int imageHeight = 224;
        private const int imageWidth = 224;
        private const int numberOfChannels = 3;
        private const int inputSize = imageHeight * imageWidth * numberOfChannels;

        // Sample that shows how an input array (of doubles) can be used to interop with image related estimators in ML.NET.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(4);

            // Convert the list of data points to an IDataView object, which is consumable by ML.NET API.
            var data = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Image loading pipeline. 
            var pipeline = mlContext.Transforms.ConvertToImage(imageHeight, imageWidth, "Image", "Features")
                          .Append(mlContext.Transforms.ExtractPixels("Pixels", "Image"));

            var transformedData = pipeline.Fit(data).Transform(data);

            // Preview the transformedData. 
            var transformedDataPreview = transformedData.Preview();
            PrintPreview(transformedDataPreview);
            // Features                 Image                    Pixels
            // 185,209,196,142,52       System.Drawing.Bitmap    185,209,196,142,52
            // 182,235,84,23,87         System.Drawing.Bitmap    182,235,84,23,87
            // 192,214,247,22,38        System.Drawing.Bitmap    192,214,247,22,38
            // 242,161,141,223,192      System.Drawing.Bitmap    242,161,141,223,192
        }

        private static void PrintPreview(DataDebuggerPreview data)
        {
            foreach (var colInfo in data.ColumnView)
                Console.Write("{0,-25}", colInfo.Column.Name);

            Console.WriteLine();
            foreach (var row in data.RowView)
            {
                foreach (var kvPair in row.Values)
                {
                    if (kvPair.Key == "Pixels" || kvPair.Key == "Features")
                    {
                        var rawValues = ((VBuffer<float>)kvPair.Value).DenseValues().Take(5);
                        Console.Write("{0,-25}", string.Join(",", rawValues));
                    }
                    else
                        Console.Write("{0,-25}", kvPair.Value);
                }
                Console.WriteLine();
            }
        }

        private class DataPoint
        {
            [VectorType(inputSize)]
            public float[] Features { get; set; }
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count, int seed = 0)
        {
            var random = new Random(seed);

            for (int i = 0; i < count; i++)
                yield return new DataPoint { Features = Enumerable.Repeat(0, inputSize).Select(x => (float)random.Next(0, 256)).ToArray() };
        }
    }
}
