using System;
using System.Drawing;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace Samples.Dynamic
{
    public static class ApplyOnnxModelWithInMemoryImages
    {
        // Example of applying ONNX transform on in-memory images.
        public static void Example()
        {
            // Download the squeeznet image model from ONNX model zoo, version 1.2
            // https://github.com/onnx/models/tree/master/vision/classification/squeezenet or use
            // Microsoft.ML.Onnx.TestModels nuget.
            // It's a multiclass classifier. It consumes an input "data_0" and
            // produces an output "softmaxout_1".
            var modelPath = @"squeezenet\00000001\model.onnx";

            // Create ML pipeline to score the data using OnnxScoringEstimator
            var mlContext = new MLContext();

            // Create in-memory data points. Its Image/Scores field is the
            // input /output of the used ONNX model.
            var dataPoints = new ImageDataPoint[]
            {
                new ImageDataPoint(Color.Red),
                new ImageDataPoint(Color.Green)
            };

            // Convert training data to IDataView, the general data type used in
            // ML.NET.
            var dataView = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Create a ML.NET pipeline which contains two steps. First,
            // ExtractPixle is used to convert the 224x224 image to a 3x224x224
            // float tensor. Then the float tensor is fed into a ONNX model with an
            // input called "data_0" and an output called "softmaxout_1". Note that
            // "data_0" and "softmaxout_1" are model input and output names stored
            // in the used ONNX model file. Users may need to inspect their own
            // models to get the right input and output column names.
            // Map column "Image" to column "data_0"
            // Map column "data_0" to column "softmaxout_1"
            var pipeline = mlContext.Transforms.ExtractPixels("data_0", "Image")
                .Append(mlContext.Transforms.ApplyOnnxModel("softmaxout_1",
                "data_0", modelPath));

            var model = pipeline.Fit(dataView);
            var onnx = model.Transform(dataView);

            // Convert IDataView back to IEnumerable<ImageDataPoint> so that user
            // can inspect the output, column "softmaxout_1", of the ONNX transform.
            // Note that Column "softmaxout_1" would be stored in ImageDataPont
            //.Scores because the added attributed [ColumnName("softmaxout_1")]
            // tells that ImageDataPont.Scores is equivalent to column
            // "softmaxout_1".
            var transformedDataPoints = mlContext.Data.CreateEnumerable<
                ImageDataPoint>(onnx, false).ToList();

            // The scores are probabilities of all possible classes, so they should
            // all be positive.
            foreach (var dataPoint in transformedDataPoints)
            {
                var firstClassProb = dataPoint.Scores.First();
                var lastClassProb = dataPoint.Scores.Last();
                Console.WriteLine("The probability of being the first class is " +
                    (firstClassProb * 100) + "%.");

                Console.WriteLine($"The probability of being the last class is " +
                    (lastClassProb * 100) + "%.");
            }

            // Expected output:
            //  The probability of being the first class is 0.002542659%.
            //  The probability of being the last class is 0.0292684%.
            //  The probability of being the first class is 0.02258059%.
            //  The probability of being the last class is 0.394428%.
        }

        // This class is used in Example() to describe data points which will be
        // consumed by ML.NET pipeline.
        private class ImageDataPoint
        {
            // Height of Image.
            private const int height = 224;

            // Width of Image.
            private const int width = 224;

            // Image will be consumed by ONNX image multiclass classification model.
            [ImageType(height, width)]
            public Bitmap Image { get; set; }

            // Expected output of ONNX model. It contains probabilities of all
            // classes. Note that the ColumnName below should match the output name
            // in the used ONNX model file.
            [ColumnName("softmaxout_1")]
            public float[] Scores { get; set; }

            public ImageDataPoint()
            {
                Image = null;
            }

            public ImageDataPoint(Color color)
            {
                Image = new Bitmap(width, height);
                for (int i = 0; i < width; ++i)
                    for (int j = 0; j < height; ++j)
                        Image.SetPixel(i, j, color);
            }
        }
    }
}
