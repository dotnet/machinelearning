using System;
using System.Drawing;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;

namespace Samples.Dynamic
{
    class ConvertToGrayScaleInMemory
    {
        public static void Example()
        {
            var mlContext = new MLContext();
            // Create an image list.
            var images = new[] { new ImageDataPoint(2, 3, Color.Blue), new
                ImageDataPoint(2, 3, Color.Red) };

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var data = mlContext.Data.LoadFromEnumerable(images);

            // Convert image to gray scale.
            var pipeline = mlContext.Transforms.ConvertToGrayscale("GrayImage",
                "Image");

            // Fit the model.
            var model = pipeline.Fit(data);

            // Test path: image files -> IDataView -> Enumerable of Bitmaps.
            var transformedData = model.Transform(data);

            // Load images in DataView back to Enumerable.
            var transformedDataPoints = mlContext.Data.CreateEnumerable<
                ImageDataPoint>(transformedData, false);

            // Print out input and output pixels.
            foreach (var dataPoint in transformedDataPoints)
            {
                var image = dataPoint.Image;
                var grayImage = dataPoint.GrayImage;
                for (int x = 0; x < grayImage.Width; ++x)
                {
                    for (int y = 0; y < grayImage.Height; ++y)
                    {
                        var pixel = image.GetPixel(x, y);
                        var grayPixel = grayImage.GetPixel(x, y);
                        Console.WriteLine($"The original pixel is {pixel} and its" +
                            $"pixel in gray is {grayPixel}");
                    }
                }
            }

            // Expected output:
            //   The original pixel is Color[A = 255, R = 0, G = 0, B = 255] and its pixel in gray is Color[A = 255, R = 28, G = 28, B = 28]
            //   The original pixel is Color[A = 255, R = 0, G = 0, B = 255] and its pixel in gray is Color[A = 255, R = 28, G = 28, B = 28]
            //   The original pixel is Color[A = 255, R = 0, G = 0, B = 255] and its pixel in gray is Color[A = 255, R = 28, G = 28, B = 28]
            //   The original pixel is Color[A = 255, R = 0, G = 0, B = 255] and its pixel in gray is Color[A = 255, R = 28, G = 28, B = 28]
            //   The original pixel is Color[A = 255, R = 0, G = 0, B = 255] and its pixel in gray is Color[A = 255, R = 28, G = 28, B = 28]
            //   The original pixel is Color[A = 255, R = 0, G = 0, B = 255] and its pixel in gray is Color[A = 255, R = 28, G = 28, B = 28]
            //   The original pixel is Color[A = 255, R = 255, G = 0, B = 0] and its pixel in gray is Color[A = 255, R = 77, G = 77, B = 77]
            //   The original pixel is Color[A = 255, R = 255, G = 0, B = 0] and its pixel in gray is Color[A = 255, R = 77, G = 77, B = 77]
            //   The original pixel is Color[A = 255, R = 255, G = 0, B = 0] and its pixel in gray is Color[A = 255, R = 77, G = 77, B = 77]
            //   The original pixel is Color[A = 255, R = 255, G = 0, B = 0] and its pixel in gray is Color[A = 255, R = 77, G = 77, B = 77]
            //   The original pixel is Color[A = 255, R = 255, G = 0, B = 0] and its pixel in gray is Color[A = 255, R = 77, G = 77, B = 77]
            //   The original pixel is Color[A = 255, R = 255, G = 0, B = 0] and its pixel in gray is Color[A = 255, R = 77, G = 77, B = 77]
        }

        private class ImageDataPoint
        {
            [ImageType(3, 4)]
            public Bitmap Image { get; set; }

            [ImageType(3, 4)]
            public Bitmap GrayImage { get; set; }

            public ImageDataPoint()
            {
                Image = null;
                GrayImage = null;
            }

            public ImageDataPoint(int width, int height, Color color)
            {
                Image = new Bitmap(width, height);
                for (int i = 0; i < width; ++i)
                    for (int j = 0; j < height; ++j)
                        Image.SetPixel(i, j, color);
            }
        }
    }
}
