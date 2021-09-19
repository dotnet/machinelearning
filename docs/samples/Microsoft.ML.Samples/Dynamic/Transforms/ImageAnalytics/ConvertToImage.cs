using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class ConvertToImage
    {
        private const int imageHeight = 224;
        private const int imageWidth = 224;
        private const int numberOfChannels = 3;
        private const int inputSize = imageHeight * imageWidth * numberOfChannels;

        // Sample that shows how an input array (of doubles) can be used to interop
        // with image related estimators in ML.NET.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(4);

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var data = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Image loading pipeline. 
            var pipeline = mlContext.Transforms.ConvertToImage(imageHeight,
                imageWidth, "Image", "Features")
                .Append(mlContext.Transforms.ExtractPixels("Pixels", "Image"));

            var transformedData = pipeline.Fit(data).Transform(data);

            // Preview the transformedData.
            PrintColumns(transformedData);

            // Features                 Image                    Pixels
            // 185,209,196,142,52...    {Width=224, Height=224}  185,209,196,142,52...
            // 182,235,84,23,87...      {Width=224, Height=224}  182,235,84,23,87...
            // 192,214,247,22,38...     {Width=224, Height=224}  192,214,247,22,38...
            // 242,161,141,223,192...   {Width=224, Height=224}  242,161,141,223,192...
        }

        private static void PrintColumns(IDataView transformedData)
        {
            Console.WriteLine("{0, -25} {1, -25} {2, -25}", "Features", "Image",
                "Pixels");

            using (var cursor = transformedData.GetRowCursor(transformedData
                .Schema))
            {
                // Note that it is best to get the getters and values *before*
                // iteration, so as to faciliate buffer sharing (if applicable), and
                // column -type validation once, rather than many times.
                VBuffer<float> features = default;
                VBuffer<float> pixels = default;
                Bitmap imageObject = null;

                var featuresGetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[
                    "Features"]);

                var pixelsGetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[
                    "Pixels"]);

                var imageGetter = cursor.GetGetter<Bitmap>(cursor.Schema["Image"]);
                while (cursor.MoveNext())
                {

                    featuresGetter(ref features);
                    pixelsGetter(ref pixels);
                    imageGetter(ref imageObject);

                    Console.WriteLine("{0, -25} {1, -25} {2, -25}", string.Join(",",
                        features.DenseValues().Take(5)) + "...", imageObject
                        .PhysicalDimension, string.Join(",", pixels.DenseValues()
                        .Take(5)) + "...");
                }

                // Dispose the image.
                imageObject.Dispose();
            }
        }

        private class DataPoint
        {
            [VectorType(inputSize)]
            public float[] Features { get; set; }
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)
        {
            var random = new Random(seed);

            for (int i = 0; i < count; i++)
                yield return new DataPoint
                {
                    Features = Enumerable.Repeat(0,
                    inputSize).Select(x => (float)random.Next(0, 256)).ToArray()
                };
        }
    }
}
