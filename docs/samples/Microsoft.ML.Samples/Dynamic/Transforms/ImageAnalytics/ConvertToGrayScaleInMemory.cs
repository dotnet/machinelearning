using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace Samples.Dynamic
{
    class ConvertToGrayScaleInMemory
    {
        public static void Example()
        {
            var mlContext = new MLContext();
            // Create an image list.
            var images = new[]
            {
                new ImageDataPoint(2, 3, red: 0, green: 0, blue: 255),    // Blue color
                new ImageDataPoint(2, 3, red: 255, green: 0, blue: 0) };  // red color

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var data = mlContext.Data.LoadFromEnumerable(images);

            // Convert image to gray scale.
            var pipeline = mlContext.Transforms.ConvertToGrayscale("GrayImage", "Image");

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
                ReadOnlySpan<byte> imageData = image.Get32bbpImageData(out int alphaIndex, out int redIndex, out int greenIndex, out int blueIndex);
                ReadOnlySpan<byte> grayImageData = grayImage.Get32bbpImageData(out int alphaIndex1, out int redIndex1, out int greenIndex1, out int blueIndex1);
                int pixelSize = image.BitsPerPixel / 8;

                for (int i = 0; i < imageData.Length; i += pixelSize)
                {
                    string pixelString = $"[A = {imageData[i + alphaIndex]}, R = {imageData[i + redIndex]}, G = {imageData[i + greenIndex]}, B = {imageData[i + blueIndex]}]";
                    string grayPixelString = $"[A = {grayImageData[i + alphaIndex1]}, R = {grayImageData[i + redIndex1]}, G = {grayImageData[i + greenIndex1]}, B = {grayImageData[i + blueIndex1]}]";

                    Console.WriteLine($"The original pixel is {pixelString} and its pixel in gray is {grayPixelString}");
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
            public ImageBase Image { get; set; }

            [ImageType(3, 4)]
            public ImageBase GrayImage { get; set; }

            public ImageDataPoint()
            {
                Image = null;
                GrayImage = null;
            }

            public ImageDataPoint(int width, int height, byte red, byte green, byte blue)
            {
                byte[] imageData = new byte[width * height * 4]; // 4 for the red, green, blue and alpha colors
                for (int i = 0; i < imageData.Length; i += 4)
                {
                    // Fill the buffer with the Bgra32 format
                    imageData[i] = blue;
                    imageData[i + 1] = green;
                    imageData[i + 2] = red;
                    imageData[i + 3] = 255;
                }

                Image = ImageBase.CreateBgra32Image(width, height, imageData);
            }
        }
    }
}
