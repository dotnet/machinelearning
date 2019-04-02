using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class ConvertToImage
    {
        private const int inputSize = 3 * 224 * 224;

        private class DataPoint
        {
            [VectorType(inputSize)]
            public double[] Features { get; set; }
        }

        private sealed class TransformedData
        {
            public float[] Pixels { get; set; }
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count, int seed = 0)
        {
            var random = new Random(seed);

            for (int i = 0; i < count; i++)
            {
                yield return new DataPoint
                {
                    Features = Enumerable.Repeat(0, inputSize).Select(x => random.NextDouble() * 100).ToArray()
                };
            }
        }

        // Sample that shows how an input array (of doubles) can be used to interop with image related estimators in ML.NET.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(10);

            // Convert the list of data points to an IDataView object, which is consumable by ML.NET API.
            var data = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Image loading pipeline. 
            var pipeline = mlContext.Transforms.ConvertToImage(224, 224, "Image", "Features")
                          .Append(mlContext.Transforms.ResizeImages("ImageResized", inputColumnName: "Image", imageWidth: 100, imageHeight: 100))
                          .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageResized"));


            var transformedData = pipeline.Fit(data).Transform(data);

            // The transformedData IDataView contains the loaded and resized raw values

            // Preview 1 row of the transformedData. 
            var transformedDataPreview = transformedData.Preview(1);
            foreach (var kvPair in transformedDataPreview.RowView[0].Values)
            {
                Console.WriteLine("{0} : {1}", kvPair.Key, kvPair.Value);
            }

            // Features: Dense vector of size 150528
            // Image: System.Drawing.Bitmap
            // ImageResized : System.Drawing.Bitmap
            // Pixels : Dense vector of size 30000

            Console.WriteLine("--------------------------------------------------");

            // Using schema comprehension to display raw pixels for each row.
            // Display extracted pixels in column 'Pixels'.
            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, true);
            foreach (var item in convertedData)
            {
                var pixels = item.Pixels.Take(5);
                Console.WriteLine("pixels:{0}...", string.Join(",", pixels));
            }

            // pixels:72,71,37,57,30...
            // pixels:71,26,61,40,71...
            // pixels:75,75,35,27,93...
            // pixels:94,63,39,63,63...
            // pixels:78,83,11,46,94...
            // pixels:15,29,25,67,35...
            // pixels:44,74,59,42,26...
            // pixels:62,85,46,46,16...
            // pixels:71,64,57,45,91...
            // pixels:52,19,6,35,81...
        }
    }
}
