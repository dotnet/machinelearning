// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Image;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class ImageTests : TestDataPipeBase
    {
        private static bool IsNotArm => RuntimeInformation.ProcessArchitecture != Architecture.Arm && RuntimeInformation.ProcessArchitecture != Architecture.Arm64;
        public ImageTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestEstimatorChain()
        {
            var env = new MLContext(1);
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var invalidData = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.Single, 0),
                }
            }, new MultiFileSource(dataFile));

            var pipe = new ImageLoadingEstimator(env, imageFolder, ("ImageReal", "ImagePath"))
                .Append(new ImageResizingEstimator(env, "ImageReal", 100, 100, "ImageReal"))
                .Append(new ImagePixelExtractingEstimator(env, "ImagePixels", "ImageReal"))
                .Append(new ImageGrayscalingEstimator(env, ("ImageGray", "ImageReal")));

            TestEstimatorCore(pipe, data, null, invalidData);
            Done();
        }

        [Fact]
        public void TestEstimatorSaveLoad()
        {
            IHostEnvironment env = new MLContext(1);
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));

            var pipe = new ImageLoadingEstimator(env, imageFolder, ("ImageReal", "ImagePath"))
                .Append(new ImageResizingEstimator(env, "ImageReal", 100, 100, "ImageReal"))
                .Append(new ImagePixelExtractingEstimator(env, "ImagePixels", "ImageReal"))
                .Append(new ImageGrayscalingEstimator(env, ("ImageGray", "ImageReal")));

            pipe.GetOutputSchema(SchemaShape.Create(data.Schema));
            var model = pipe.Fit(data);

            var tempPath = Path.GetTempFileName();
            using (var file = new SimpleFileHandle(env, tempPath, true, true))
            {
                using (var fs = file.CreateWriteStream())
                    ML.Model.Save(model, null, fs);
                ITransformer model2;
                using (var fs = file.OpenReadStream())
                    model2 = ML.Model.Load(fs, out var schema);

                var transformerChain = model2 as TransformerChain<ITransformer>;
                Assert.NotNull(transformerChain);

                var newCols = ((ImageLoadingTransformer)transformerChain.First()).Columns;
                var oldCols = ((ImageLoadingTransformer)model.First()).Columns;
                Assert.True(newCols
                    .Zip(oldCols, (x, y) => x == y)
                    .All(x => x));
            }
            Done();
        }

        [Fact]
        public void TestLoadImages()
        {
            var env = new MLContext(1);
            var dataFile = GetDataPath("images/images.tsv");
            var correctImageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));

            // Testing for invalid imageFolder path, should throw an ArgumentException
            var incorrectImageFolder = correctImageFolder + "-nonExistantDirectory";
            Assert.Throws<ArgumentException>(() => new ImageLoadingTransformer(env, incorrectImageFolder, ("ImageReal", "ImagePath")).Transform(data));

            // Testing for empty imageFolder path, should not throw an exception
            var emptyImageFolder = String.Empty;
            var imagesEmptyImageFolder = new ImageLoadingTransformer(env, emptyImageFolder, ("ImageReal", "ImagePath")).Transform(data);

            // Testing for null imageFolder path, should not throw an exception
            var imagesNullImageFolder = new ImageLoadingTransformer(env, null, ("ImageReal", "ImagePath")).Transform(data);

            // Testing for correct imageFolder path, should not throw an exception
            var imagesCorrectImageFolder = new ImageLoadingTransformer(env, correctImageFolder, ("ImageReal", "ImagePath")).Transform(data);

            Done();
        }

        [Fact]
        public void TestSaveImages()
        {
            var env = new MLContext(1);
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", 100, 100, "ImageReal", ImageResizingEstimator.ResizingKind.IsoPad).Transform(images);

            using (var cursor = cropped.GetRowCursorForAllColumns())
            {
                var pathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cropped.Schema["ImagePath"]);
                ReadOnlyMemory<char> path = default;
                var imageCropGetter = cursor.GetGetter<MLImage>(cropped.Schema["ImageCropped"]);
                MLImage image = default;
                while (cursor.MoveNext())
                {
                    pathGetter(ref path);
                    imageCropGetter(ref image);
                    Assert.NotNull(image);
                    var fileToSave = GetOutputPath(Path.GetFileNameWithoutExtension(path.ToString()) + ".cropped.jpg");
                    image.Save(fileToSave);
                }
            }
            Done();
        }

        [Fact]
        public void TestGrayscaleTransformImages()
        {
            IHostEnvironment env = new MLContext(1);
            var imageHeight = 150;
            var imageWidth = 100;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);

            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            IDataView grey = new ImageGrayscalingTransformer(env, ("ImageGrey", "ImageCropped")).Transform(cropped);
            var fname = nameof(TestGrayscaleTransformImages) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(grey));

            grey = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            grey.Schema.TryGetColumnIndex("ImageGrey", out int greyColumn);
            using (var cursor = grey.GetRowCursorForAllColumns())
            {
                var imageGetter = cursor.GetGetter<MLImage>(grey.Schema["ImageGrey"]);
                MLImage image = default;
                while (cursor.MoveNext())
                {
                    imageGetter(ref image);
                    Assert.NotNull(image);

                    ReadOnlySpan<byte> imageData = image.Pixels;
                    (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = image.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };
                    int pixelSize = image.BitsPerPixel / 8;

                    for (int i = 0; i < imageData.Length; i += pixelSize)
                    {
                        // grayscale image has same values for R,G and B
                        Assert.True(imageData[i + redIndex] == imageData[i + greenIndex] && imageData[i + greenIndex] == imageData[i + blueIndex]);
                    }
                }
            }
            Done();
        }

        [Fact]
        public void TestGrayScaleInMemory()
        {
            // Create an image list.
            var images = new List<ImageDataPoint>() { new ImageDataPoint(10, 10, red: 0, green: 0, blue: 255), new ImageDataPoint(10, 10, red: 255, green: 0, blue: 0) };

            // Convert the list of data points to an IDataView object, which is consumable by ML.NET API.
            var data = ML.Data.LoadFromEnumerable(images);

            // Convert image to gray scale.
            var pipeline = ML.Transforms.ConvertToGrayscale("GrayImage", "Image");

            // Fit the model.
            var model = pipeline.Fit(data);

            // Test path: image files -> IDataView -> Enumerable of images.
            var transformedData = model.Transform(data);

            // Load images in DataView back to Enumerable.
            var transformedDataPoints = ML.Data.CreateEnumerable<ImageDataPoint>(transformedData, false);

            foreach (var dataPoint in transformedDataPoints)
            {
                var image = dataPoint.Image;
                var grayImage = dataPoint.GrayImage;

                Assert.NotNull(grayImage);

                Assert.Equal(image.Width, grayImage.Width);
                Assert.Equal(image.Height, grayImage.Height);

                ReadOnlySpan<byte> imageData = grayImage.Pixels;
                (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = grayImage.PixelFormat switch
                {
                    MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                    MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                    _ => throw new InvalidOperationException($"Image pixel format is not supported")
                };
                int pixelSize = grayImage.BitsPerPixel / 8;

                for (int i = 0; i < imageData.Length; i += pixelSize)
                {
                    // grayscale image has same values for R,G and B
                    Assert.True(imageData[i + redIndex] == imageData[i + greenIndex] && imageData[i + greenIndex] == imageData[i + blueIndex]);
                }
            }

            var engine = ML.Model.CreatePredictionEngine<ImageDataPoint, ImageDataPoint>(model);
            var singleImage = new ImageDataPoint(17, 36, red: 255, green: 192, blue: 203); // Pink color (255, 192, 203)
            var transformedSingleImage = engine.Predict(singleImage);

            Assert.Equal(singleImage.Image.Height, transformedSingleImage.GrayImage.Height);
            Assert.Equal(singleImage.Image.Width, transformedSingleImage.GrayImage.Width);

            ReadOnlySpan<byte> imageData1 = transformedSingleImage.GrayImage.Pixels;
            (int alphaIndex1, int redIndex1, int greenIndex1, int blueIndex1) = transformedSingleImage.GrayImage.PixelFormat switch
            {
                MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                _ => throw new InvalidOperationException($"Image pixel format is not supported")
            };
            int pixelSize1 = transformedSingleImage.GrayImage.BitsPerPixel / 8;

            for (int i = 0; i < imageData1.Length; i += pixelSize1)
            {
                // grayscale image has same values for R,G and B
                Assert.True(imageData1[i + redIndex1] == imageData1[i + greenIndex1] && imageData1[i + greenIndex1] == imageData1[i + blueIndex1]);
            }
        }

        private class ImageDataPoint
        {
            [ImageType(10, 10)]
            public MLImage Image { get; set; }

            [ImageType(10, 10)]
            public MLImage GrayImage { get; set; }

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

                Image = MLImage.CreateFromPixels(width, height, MLPixelFormat.Bgra32, imageData);
            }
        }

        [Fact]
        public void TestBackAndForthConversionWithAlphaInterleave()
        {
            IHostEnvironment env = new MLContext(1);
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, interleavePixelColors: true, scaleImage: 2f / 19, offsetImage: 30).Transform(cropped);
            IDataView backToImages = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               ImagePixelExtractingEstimator.ColorBits.All, interleavedColors: true, scaleImage: 19 / 2f, offsetImage: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToImages));

            backToImages = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToImages.GetRowCursorForAllColumns())
            {
                var imageGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageRestored"]);
                MLImage restoredImage = default;

                var imageCropGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageCropped"]);
                MLImage croppedImage = default;
                while (cursor.MoveNext())
                {
                    imageGetter(ref restoredImage);
                    Assert.NotNull(restoredImage);
                    imageCropGetter(ref croppedImage);
                    Assert.NotNull(croppedImage);

                    ReadOnlySpan<byte> restoredImageData = restoredImage.Pixels;
                    (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = restoredImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    ReadOnlySpan<byte> croppedImageData = croppedImage.Pixels;
                    (int alphaIndex1, int redIndex1, int greenIndex1, int blueIndex1) = croppedImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    int pixelSize = restoredImage.BitsPerPixel / 8;

                    for (int i = 0; i < restoredImageData.Length; i += pixelSize)
                    {
                        Assert.Equal(restoredImageData[i + redIndex], croppedImageData[i + redIndex1]);
                        Assert.Equal(restoredImageData[i + greenIndex], croppedImageData[i + greenIndex1]);
                        Assert.Equal(restoredImageData[i + blueIndex], croppedImageData[i + blueIndex1]);
                    }
                }
            }
            Done();
        }

        [Fact]
        public void TestBackAndForthConversionWithoutAlphaInterleave()
        {
            IHostEnvironment env = new MLContext(1);
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", interleavePixelColors: true, scaleImage: 2f / 19, offsetImage: 30).Transform(cropped);

            IDataView backToImages = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               interleavedColors: true, scaleImage: 19 / 2f, offsetImage: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToImages));

            backToImages = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToImages.GetRowCursorForAllColumns())
            {
                var imageGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageRestored"]);
                MLImage restoredImage = default;

                var imageCropGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageCropped"]);
                MLImage croppedImage = default;
                while (cursor.MoveNext())
                {
                    imageGetter(ref restoredImage);
                    Assert.NotNull(restoredImage);
                    imageCropGetter(ref croppedImage);
                    Assert.NotNull(croppedImage);

                    ReadOnlySpan<byte> restoredImageData = restoredImage.Pixels;
                    (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = restoredImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    ReadOnlySpan<byte> croppedImageData = croppedImage.Pixels;
                    (int alphaIndex1, int redIndex1, int greenIndex1, int blueIndex1) = croppedImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    int pixelSize = restoredImage.BitsPerPixel / 8;

                    for (int i = 0; i < restoredImageData.Length; i += pixelSize)
                    {
                        Assert.True(
                            croppedImageData[i + redIndex1] == restoredImageData[i + redIndex] &&
                            croppedImageData[i + greenIndex1] == restoredImageData[i + greenIndex] &&
                            croppedImageData[i + blueIndex1] == restoredImageData[i + blueIndex]);
                    }
                }
            }
            Done();
        }

        [Fact]
        public void TestBackAndForthConversionWithDifferentOrder()
        {
            IHostEnvironment env = new MLContext(1);
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, orderOfExtraction: ImagePixelExtractingEstimator.ColorsOrder.ABRG).Transform(cropped);
            IDataView backToImages = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               ImagePixelExtractingEstimator.ColorBits.All, orderOfColors: ImagePixelExtractingEstimator.ColorsOrder.ABRG).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithDifferentOrder) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToImages));

            backToImages = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToImages.GetRowCursorForAllColumns())
            {
                var imageGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageRestored"]);
                MLImage restoredImage = default;

                var imageCropGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageCropped"]);
                MLImage croppedImage = default;
                while (cursor.MoveNext())
                {
                    imageGetter(ref restoredImage);
                    Assert.NotNull(restoredImage);
                    imageCropGetter(ref croppedImage);
                    Assert.NotNull(croppedImage);

                    ReadOnlySpan<byte> restoredImageData = restoredImage.Pixels;
                    (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = restoredImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    ReadOnlySpan<byte> croppedImageData = croppedImage.Pixels;
                    (int alphaIndex1, int redIndex1, int greenIndex1, int blueIndex1) = croppedImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    int pixelSize = restoredImage.BitsPerPixel / 8;

                    for (int i = 0; i < restoredImageData.Length; i += pixelSize)
                    {
                        Assert.Equal(restoredImageData[i + redIndex], croppedImageData[i + redIndex1]);
                        Assert.Equal(restoredImageData[i + greenIndex], croppedImageData[i + greenIndex1]);
                        Assert.Equal(restoredImageData[i + blueIndex], croppedImageData[i + blueIndex1]);
                    }
                }
            }
            Done();
        }

        [ConditionalFact(nameof(IsNotArm))] //"System.Drawing has some issues on ARM. Disabling this test for CI stability. Tracked in https://github.com/dotnet/machinelearning/issues/6043"
        public void TestBackAndForthConversionWithAlphaNoInterleave()
        {
            IHostEnvironment env = new MLContext(1);
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, scaleImage: 2f / 19, offsetImage: 30).Transform(cropped);

            IDataView backToImages = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                ImagePixelExtractingEstimator.ColorBits.All, scaleImage: 19 / 2f, offsetImage: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaNoInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToImages));

            backToImages = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToImages.GetRowCursorForAllColumns())
            {
                var imageGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageRestored"]);
                MLImage restoredImage = default;

                var imageCropGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageCropped"]);
                MLImage croppedImage = default;
                while (cursor.MoveNext())
                {
                    imageGetter(ref restoredImage);
                    Assert.NotNull(restoredImage);
                    imageCropGetter(ref croppedImage);
                    Assert.NotNull(croppedImage);

                    ReadOnlySpan<byte> restoredImageData = restoredImage.Pixels;
                    (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = restoredImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    ReadOnlySpan<byte> croppedImageData = croppedImage.Pixels;
                    (int alphaIndex1, int redIndex1, int greenIndex1, int blueIndex1) = croppedImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    int pixelSize = restoredImage.BitsPerPixel / 8;

                    for (int i = 0; i < restoredImageData.Length; i += pixelSize)
                    {
                        Assert.Equal(restoredImageData[i + redIndex], croppedImageData[i + redIndex1]);
                        Assert.Equal(restoredImageData[i + greenIndex], croppedImageData[i + greenIndex1]);
                        Assert.Equal(restoredImageData[i + blueIndex], croppedImageData[i + blueIndex1]);
                    }
                }
            }
            Done();
        }

        [ConditionalFact(nameof(IsNotArm))] //"System.Drawing has some issues on ARM. Disabling this test for CI stability. Tracked in https://github.com/dotnet/machinelearning/issues/6043"
        public void TestBackAndForthConversionWithoutAlphaNoInterleave()
        {
            IHostEnvironment env = new MLContext(1);
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", scaleImage: 2f / 19, offsetImage: 30).Transform(cropped);

            IDataView backToImages = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                scaleImage: 19 / 2f, offsetImage: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaNoInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToImages));

            backToImages = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToImages.GetRowCursorForAllColumns())
            {
                var imageGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageRestored"]);
                MLImage restoredImage = default;

                var imageCropGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageCropped"]);
                MLImage croppedImage = default;
                while (cursor.MoveNext())
                {
                    imageGetter(ref restoredImage);
                    Assert.NotNull(restoredImage);
                    imageCropGetter(ref croppedImage);
                    Assert.NotNull(croppedImage);

                    ReadOnlySpan<byte> restoredImageData = restoredImage.Pixels;
                    (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = restoredImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    ReadOnlySpan<byte> croppedImageData = croppedImage.Pixels;
                    (int alphaIndex1, int redIndex1, int greenIndex1, int blueIndex1) = croppedImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    int pixelSize = restoredImage.BitsPerPixel / 8;

                    for (int i = 0; i < restoredImageData.Length; i += pixelSize)
                    {
                        Assert.Equal(restoredImageData[i + redIndex], croppedImageData[i + redIndex1]);
                        Assert.Equal(restoredImageData[i + greenIndex], croppedImageData[i + greenIndex1]);
                        Assert.Equal(restoredImageData[i + blueIndex], croppedImageData[i + blueIndex1]);
                    }
                }
            }
            Done();
        }

        [ConditionalFact(nameof(IsNotArm))] //"System.Drawing has some issues on ARM. Disabling this test for CI stability. Tracked in https://github.com/dotnet/machinelearning/issues/6043"
        public void TestBackAndForthConversionWithAlphaInterleaveNoOffset()
        {
            IHostEnvironment env = new MLContext(1);
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, interleavePixelColors: true).Transform(cropped);

            IDataView backToImages = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                ImagePixelExtractingEstimator.ColorBits.All, interleavedColors: true).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToImages));

            backToImages = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToImages.GetRowCursorForAllColumns())
            {
                var imageGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageRestored"]);
                MLImage restoredImage = default;

                var imageCropGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageCropped"]);
                MLImage croppedImage = default;
                while (cursor.MoveNext())
                {
                    imageGetter(ref restoredImage);
                    Assert.NotNull(restoredImage);
                    imageCropGetter(ref croppedImage);
                    Assert.NotNull(croppedImage);

                    ReadOnlySpan<byte> restoredImageData = restoredImage.Pixels;
                    (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = restoredImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    ReadOnlySpan<byte> croppedImageData = croppedImage.Pixels;
                    (int alphaIndex1, int redIndex1, int greenIndex1, int blueIndex1) = croppedImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    int pixelSize = restoredImage.BitsPerPixel / 8;

                    for (int i = 0; i < restoredImageData.Length; i += pixelSize)
                    {
                        Assert.Equal(restoredImageData[i + redIndex], croppedImageData[i + redIndex1]);
                        Assert.Equal(restoredImageData[i + greenIndex], croppedImageData[i + greenIndex1]);
                        Assert.Equal(restoredImageData[i + blueIndex], croppedImageData[i + blueIndex1]);
                    }
                }
            }
            Done();
        }

        [ConditionalFact(nameof(IsNotArm))] //"System.Drawing has some issues on ARM. Disabling this test for CI stability. Tracked in https://github.com/dotnet/machinelearning/issues/6043"
        public void TestBackAndForthConversionWithoutAlphaInterleaveNoOffset()
        {
            IHostEnvironment env = new MLContext(1);
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", interleavePixelColors: true).Transform(cropped);

            IDataView backToImages = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels", interleavedColors: true).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToImages));

            backToImages = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToImages.GetRowCursorForAllColumns())
            {
                var imageGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageRestored"]);
                MLImage restoredImage = default;

                var imageCropGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageCropped"]);
                MLImage croppedImage = default;
                while (cursor.MoveNext())
                {
                    imageGetter(ref restoredImage);
                    Assert.NotNull(restoredImage);
                    imageCropGetter(ref croppedImage);
                    Assert.NotNull(croppedImage);

                    ReadOnlySpan<byte> restoredImageData = restoredImage.Pixels;
                    (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = restoredImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    ReadOnlySpan<byte> croppedImageData = croppedImage.Pixels;
                    (int alphaIndex1, int redIndex1, int greenIndex1, int blueIndex1) = croppedImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    int pixelSize = restoredImage.BitsPerPixel / 8;

                    for (int i = 0; i < restoredImageData.Length; i += pixelSize)
                    {
                        Assert.Equal(restoredImageData[i + redIndex], croppedImageData[i + redIndex1]);
                        Assert.Equal(restoredImageData[i + greenIndex], croppedImageData[i + greenIndex1]);
                        Assert.Equal(restoredImageData[i + blueIndex], croppedImageData[i + blueIndex1]);
                    }
                }
            }
            Done();
        }

        [ConditionalFact(nameof(IsNotArm))] //"System.Drawing has some issues on ARM. Disabling this test for CI stability. Tracked in https://github.com/dotnet/machinelearning/issues/6043"
        public void TestBackAndForthConversionWithAlphaNoInterleaveNoOffset()
        {
            IHostEnvironment env = new MLContext(1);
            const int imageHeight = 100;
            var imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All).Transform(cropped);

            IDataView backToImages = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                 ImagePixelExtractingEstimator.ColorBits.All).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaNoInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToImages));

            backToImages = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToImages.GetRowCursorForAllColumns())
            {
                var imageGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageRestored"]);
                MLImage restoredImage = default;

                var imageCropGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageCropped"]);
                MLImage croppedImage = default;
                while (cursor.MoveNext())
                {
                    imageGetter(ref restoredImage);
                    Assert.NotNull(restoredImage);
                    imageCropGetter(ref croppedImage);
                    Assert.NotNull(croppedImage);

                    ReadOnlySpan<byte> restoredImageData = restoredImage.Pixels;
                    (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = restoredImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    ReadOnlySpan<byte> croppedImageData = croppedImage.Pixels;
                    (int alphaIndex1, int redIndex1, int greenIndex1, int blueIndex1) = croppedImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    int pixelSize = restoredImage.BitsPerPixel / 8;

                    for (int i = 0; i < restoredImageData.Length; i += pixelSize)
                    {
                        Assert.Equal(restoredImageData[i + redIndex], croppedImageData[i + redIndex1]);
                        Assert.Equal(restoredImageData[i + greenIndex], croppedImageData[i + greenIndex1]);
                        Assert.Equal(restoredImageData[i + blueIndex], croppedImageData[i + blueIndex1]);
                    }
                }
            }
            Done();
        }

        [Fact]
        public void TestBackAndForthConversionWithoutAlphaNoInterleaveNoOffset()
        {
            IHostEnvironment env = new MLContext(1);
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped").Transform(cropped);

            IDataView backToImages = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels").Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaNoInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToImages));

            backToImages = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToImages.GetRowCursorForAllColumns())
            {
                var imageGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageRestored"]);
                MLImage restoredImage = default;

                var imageCropGetter = cursor.GetGetter<MLImage>(backToImages.Schema["ImageCropped"]);
                MLImage croppedImage = default;
                while (cursor.MoveNext())
                {
                    imageGetter(ref restoredImage);
                    Assert.NotNull(restoredImage);
                    imageCropGetter(ref croppedImage);
                    Assert.NotNull(croppedImage);

                    ReadOnlySpan<byte> restoredImageData = restoredImage.Pixels;
                    (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = restoredImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    ReadOnlySpan<byte> croppedImageData = croppedImage.Pixels;
                    (int alphaIndex1, int redIndex1, int greenIndex1, int blueIndex1) = croppedImage.PixelFormat switch
                    {
                        MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                        MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                        _ => throw new InvalidOperationException($"Image pixel format is not supported")
                    };

                    int pixelSize = restoredImage.BitsPerPixel / 8;

                    for (int i = 0; i < restoredImageData.Length; i += pixelSize)
                    {
                        Assert.Equal(restoredImageData[i + redIndex], croppedImageData[i + redIndex1]);
                        Assert.Equal(restoredImageData[i + greenIndex], croppedImageData[i + greenIndex1]);
                        Assert.Equal(restoredImageData[i + blueIndex], croppedImageData[i + blueIndex1]);
                    }
                }
                Done();
            }
        }

        [Fact]
        public void ImageResizerTransformResizingModeFill()
        {
            var env = new MLContext(1);
            var dataFile = GetDataPath("images/fillmode.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0)
                }
            }, new MultiFileSource(dataFile));

            const int targetDimension = 50;
            var pipe = new ImageLoadingEstimator(env, imageFolder, ("ImageReal", "ImagePath"))
                .Append(new ImageResizingEstimator(env, "ImageReal", targetDimension, targetDimension, "ImageReal",
                    resizing: ImageResizingEstimator.ResizingKind.Fill));

            var rowView = pipe.Preview(data).RowView;
            Assert.Single(rowView);

            using (var image = (MLImage)rowView.First().Values.Last().Value)
            {
                ReadOnlySpan<byte> imageData = image.Pixels;
                (int alphaIndex, int redIndex, int greenIndex, int blueIndex) = image.PixelFormat switch
                {
                    MLPixelFormat.Bgra32 => (3, 2, 1, 0),
                    MLPixelFormat.Rgba32 => (3, 0, 1, 2),
                    _ => throw new InvalidOperationException($"Image pixel format is not supported")
                };
                int pixelSize = image.BitsPerPixel / 8;

                // these points must be white
                (int red, int green, int blue) topLeft = (imageData[redIndex], imageData[greenIndex], imageData[blueIndex]);
                int index = pixelSize * (image.Width - 1);
                (int red, int green, int blue) topRight = (imageData[index + redIndex], imageData[index + greenIndex], imageData[index + blueIndex]);
                index = pixelSize * (image.Width) * (image.Height - 1);
                (int red, int green, int blue) bottomLeft = (imageData[index + redIndex], imageData[index + greenIndex], imageData[index + blueIndex]);
                index = pixelSize * image.Width * image.Height - pixelSize;
                (int red, int green, int blue) bottomRight = (imageData[index + redIndex], imageData[index + greenIndex], imageData[index + blueIndex]);
                index = pixelSize * image.Width * ((image.Height / 2) - 1) + pixelSize * ((image.Width / 2) - 1);
                (int red, int green, int blue) middle = (imageData[index + redIndex], imageData[index + greenIndex], imageData[index + blueIndex]);

                // these points must be red
                index = pixelSize * image.Width * ((image.Height / 3) - 1) + pixelSize * ((image.Width / 2) - 1);
                (int red, int green, int blue) midTop = (imageData[index + redIndex], imageData[index + greenIndex], imageData[index + blueIndex]);

                index = pixelSize * image.Width * ((image.Height / 3 * 2) - 1) + pixelSize * ((image.Width / 2) - 1);
                (int red, int green, int blue) midBottom = (imageData[index + redIndex], imageData[index + greenIndex], imageData[index + blueIndex]);

                index = pixelSize * image.Width * ((image.Height / 2) - 1) + pixelSize * ((image.Width / 3) - 1);
                (int red, int green, int blue) leftMid = (imageData[index + redIndex], imageData[index + greenIndex], imageData[index + blueIndex]);

                index = pixelSize * image.Width * ((image.Height / 2) - 1) + pixelSize * ((image.Width / 3 * 2) - 1);
                (int red, int green, int blue) rightMid = (imageData[index + redIndex], imageData[index + greenIndex], imageData[index + blueIndex]);

                // it turns out rounding errors on certain platforms may lead to a test failure
                // instead of checking for exactly FFFFFF and FF0000 we allow a small interval here to be safe
                Assert.All(new[] { topLeft, topRight, bottomLeft, bottomRight, middle }, c =>
                {
                    Assert.True(c.red >= 250);
                    Assert.True(c.green >= 250);
                    Assert.True(c.blue >= 250);
                });
                Assert.All(new[] { midTop, midBottom, leftMid, rightMid }, c =>
                {
                    Assert.True(c.red >= 250);
                    Assert.True(c.green < 6);
                    Assert.True(c.blue < 6);
                });
            }

            Done();
        }

        [Fact]
        public void TestConvertToImage()
        {
            var mlContext = new MLContext(0);

            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(10);

            // Convert the list of data points to an IDataView object, which is consumable by ML.NET API.
            var data = mlContext.Data.LoadFromEnumerable(dataPoints);

            var pipeline = mlContext.Transforms.ConvertToImage(224, 224, "Features");

            TestEstimatorCore(pipeline, data);
            Done();
        }

        private const int InputSize = 3 * 224 * 224;

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count, int seed = 0)
        {
            var random = new Random(seed);

            for (int i = 0; i < count; i++)
            {
                yield return new DataPoint
                {
                    Features = Enumerable.Repeat(0, InputSize).Select(x => random.NextDouble() * 100).ToArray()
                };
            }
        }

        private class DataPoint
        {
            [VectorType(InputSize)]
            public double[] Features { get; set; }
        }

        public class InMemoryImage
        {
            [ImageType(229, 299)]
            public MLImage LoadedImage;
            public string Label;

            public static List<InMemoryImage> LoadFromTsv(MLContext mlContext, string tsvPath, string imageFolder)
            {
                var inMemoryImages = new List<InMemoryImage>();
                var tsvFile = mlContext.Data.LoadFromTextFile(tsvPath, columns: new[]
                    {
                            new TextLoader.Column("ImagePath", DataKind.String, 0),
                            new TextLoader.Column("Label", DataKind.String, 1),
                    }
                );

                using (var cursor = tsvFile.GetRowCursorForAllColumns())
                {
                    var pathBuffer = default(ReadOnlyMemory<char>);
                    var labelBuffer = default(ReadOnlyMemory<char>);
                    var pathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(tsvFile.Schema["ImagePath"]);
                    var labelGetter = cursor.GetGetter<ReadOnlyMemory<char>>(tsvFile.Schema["Label"]);
                    while (cursor.MoveNext())
                    {
                        pathGetter(ref pathBuffer);
                        labelGetter(ref labelBuffer);

                        var label = labelBuffer.ToString();
                        var fileName = pathBuffer.ToString();
                        var imagePath = Path.Combine(imageFolder, fileName);

                        inMemoryImages.Add(
                                new InMemoryImage()
                                {
                                    Label = label,
                                    LoadedImage = LoadImageFromFile(imagePath)
                                }
                            );
                    }
                }

                return inMemoryImages;

            }

            private static MLImage LoadImageFromFile(string imagePath) => MLImage.CreateFromFile(imagePath);
        }

        public class InMemoryImageOutput : InMemoryImage
        {
            [ImageType(100, 100)]
            public MLImage ResizedImage;
        }

        [Fact]
        public void ResizeInMemoryImages()
        {
            var mlContext = new MLContext(seed: 1);
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var dataObjects = InMemoryImage.LoadFromTsv(mlContext, dataFile, imageFolder);

            var dataView = mlContext.Data.LoadFromEnumerable<InMemoryImage>(dataObjects);
            var pipeline = mlContext.Transforms.ResizeImages("ResizedImage", 100, 100, nameof(InMemoryImage.LoadedImage));

            // Check that the output is resized, and that it didn't resize the original image object
            var model = pipeline.Fit(dataView);
            var resizedDV = model.Transform(dataView);
            var rowView = resizedDV.Preview().RowView;
            var resizedImage = (MLImage)rowView.First().Values.Last().Value;
            Assert.Equal(100, resizedImage.Height);
            Assert.NotEqual(100, dataObjects[0].LoadedImage.Height);

            // Also check usage of prediction Engine
            // And that the references to the original image objects aren't lost
            var predEngine = mlContext.Model.CreatePredictionEngine<InMemoryImage, InMemoryImageOutput>(model);
            for (int i = 0; i < dataObjects.Count(); i++)
            {
                var prediction = predEngine.Predict(dataObjects[i]);
                Assert.Equal(100, prediction.ResizedImage.Height);
                Assert.NotEqual(100, prediction.LoadedImage.Height);
                Assert.True(prediction.LoadedImage == dataObjects[i].LoadedImage);
                Assert.False(prediction.ResizedImage == dataObjects[i].LoadedImage);
            }

            // Check that the last in-memory image hasn't been disposed
            // By running ResizeImageTransformer (see https://github.com/dotnet/machinelearning/issues/4126)
            bool disposed = false;
            try
            {
                int i = dataObjects.Last().LoadedImage.Height;
            }
            catch
            {
                disposed = true;
            }

            Assert.False(disposed, "The last in memory image had been disposed by running ResizeImageTransformer");
        }

        public static IEnumerable<object[]> ImageListData()
        {
            yield return new object[] { "tomato.bmp" };
            yield return new object[] { "hotdog.jpg" };
            yield return new object[] { "banana.jpg" };
            yield return new object[] { "tomato.jpg" };
        }

        [Theory]
        [MemberData(nameof(ImageListData))]
        public void MLImageCreationTests(string imageName)
        {
            var dataFile = GetDataPath($"images/{imageName}");

            using MLImage image1 = MLImage.CreateFromFile(dataFile);
            using FileStream imageStream = new FileStream(dataFile, FileMode.Open, FileAccess.Read);
            using MLImage image2 = MLImage.CreateFromStream(imageStream);

            Assert.Equal(image1.Tag, image2.Tag);
            Assert.Equal(image1.Width, image2.Width);
            Assert.Equal(image1.Height, image2.Height);
            Assert.Equal(32, image1.BitsPerPixel);
            Assert.Equal(image1.BitsPerPixel, image2.BitsPerPixel);
            Assert.Equal(image1.PixelFormat, image2.PixelFormat);
            Assert.Equal(image1.Pixels.ToArray(), image2.Pixels.ToArray());
            Assert.Equal(image1.Width * image1.Height * (image1.BitsPerPixel / 8), image1.Pixels.Length);
            Assert.True(image1.PixelFormat == MLPixelFormat.Rgba32 || image1.PixelFormat == MLPixelFormat.Bgra32);

            image1.Tag = "image1";
            Assert.Equal("image1", image1.Tag);
            image2.Tag = "image2";
            Assert.Equal("image2", image2.Tag);

            using MLImage image3 = MLImage.CreateFromPixels(image1.Width, image1.Height, image1.PixelFormat, image1.Pixels);
            Assert.Equal(image1.Width, image3.Width);
            Assert.Equal(image1.Height, image3.Height);
            Assert.Equal(image1.BitsPerPixel, image3.BitsPerPixel);
            Assert.Equal(image1.PixelFormat, image3.PixelFormat);
            Assert.Equal(image1.Pixels.ToArray(), image3.Pixels.ToArray());
        }

        [Fact]
        public void MLImageCreateThrowingTest()
        {
            Assert.Throws<ArgumentNullException>(() => MLImage.CreateFromFile(null));
            Assert.Throws<ArgumentException>(() => MLImage.CreateFromFile("This is Invalid Path"));
            Assert.Throws<ArgumentNullException>(() => MLImage.CreateFromStream(null));
            Assert.Throws<ArgumentException>(() => MLImage.CreateFromStream(new MemoryStream(new byte[10])));
            Assert.Throws<ArgumentException>(() => MLImage.CreateFromPixels(10, 10, MLPixelFormat.Unknown, Array.Empty<byte>()));
            Assert.Throws<ArgumentException>(() => MLImage.CreateFromPixels(10, 10, MLPixelFormat.Bgra32, Array.Empty<byte>()));
            Assert.Throws<ArgumentException>(() => MLImage.CreateFromPixels(0, 10, MLPixelFormat.Bgra32, new byte[10]));
            Assert.Throws<ArgumentException>(() => MLImage.CreateFromPixels(10, 0, MLPixelFormat.Bgra32, new byte[10]));
            Assert.Throws<ArgumentException>(() => MLImage.CreateFromPixels(10, 10, MLPixelFormat.Bgra32, new byte[401]));
        }

        [Theory]
        [MemberData(nameof(ImageListData))]
        public void MLImageSaveTests(string imageName)
        {
            var dataFile = GetDataPath($"images/{imageName}");
            using MLImage image1 = MLImage.CreateFromFile(dataFile);
            string extension = Path.GetExtension(imageName);
            string imageTempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + extension);

            if (extension.Equals(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                extension.Equals(".jpg", StringComparison.OrdinalIgnoreCase) ||
                extension.Equals(".png", StringComparison.OrdinalIgnoreCase) ||
                extension.Equals(".webp", StringComparison.OrdinalIgnoreCase))
            {
                image1.Save(imageTempPath);
                using MLImage image2 = MLImage.CreateFromFile(imageTempPath);

                Assert.Equal(image1.Width, image2.Width);
                Assert.Equal(image1.Height, image2.Height);
                Assert.Equal(image1.BitsPerPixel, image2.BitsPerPixel);
                Assert.Equal(image1.PixelFormat, image2.PixelFormat);

                // When saving the image with specific encoding, the image decoder can manipulate the color
                // and don't have to keep the exact original colors.
            }
            else
            {
                Assert.Throws<ArgumentException>(() => image1.Save(imageTempPath));
            }
        }

        [Fact]
        public void MLImageDisposingTest()
        {
            MLImage image = MLImage.CreateFromPixels(10, 10, MLPixelFormat.Bgra32, new byte[10 * 10 * 4]);
            image.Tag = "Blank";

            Assert.Equal(10, image.Width);
            Assert.Equal(10, image.Height);
            Assert.Equal(32, image.BitsPerPixel);
            Assert.Equal(MLPixelFormat.Bgra32, image.PixelFormat);

            image.Dispose();

            Assert.Throws<InvalidOperationException>(() => image.Tag);
            Assert.Throws<InvalidOperationException>(() => image.Tag = "Something");
            Assert.Throws<InvalidOperationException>(() => image.Width);
            Assert.Throws<InvalidOperationException>(() => image.Height);
            Assert.Throws<InvalidOperationException>(() => image.PixelFormat);
            Assert.Throws<InvalidOperationException>(() => image.BitsPerPixel);
            Assert.Throws<InvalidOperationException>(() => image.Pixels[0]);
        }

        [Theory]
        [MemberData(nameof(ImageListData))]
        public void MLImageSourceDisposingTest(string imageName)
        {
            var imageFile = GetDataPath($"images/{imageName}");
            using MLImage image1 = MLImage.CreateFromFile(imageFile);

            // Create image from stream then close the stream and then try to access the image data
            FileStream stream = new FileStream(imageFile, FileMode.Open, FileAccess.Read, FileShare.None);
            MLImage image2 = MLImage.CreateFromStream(stream);
            stream.Dispose();
            Assert.Equal(image1.Pixels.ToArray(), image2.Pixels.ToArray());
            image2.Dispose();

            // Create image from non-seekable stream
            stream = new FileStream(imageFile, FileMode.Open, FileAccess.Read, FileShare.None);
            ReadOnlyNonSeekableStream nonSeekableStream = new ReadOnlyNonSeekableStream(stream);
            image2 = MLImage.CreateFromStream(nonSeekableStream);
            Assert.Equal(image1.Pixels.ToArray(), image2.Pixels.ToArray());
            stream.Close();
            Assert.Equal(image1.Pixels.ToArray(), image2.Pixels.ToArray());
            image2.Dispose();

            // Now test image stream contains image data prepended and appended with extra unrelated data.
            stream = new FileStream(imageFile, FileMode.Open, FileAccess.Read, FileShare.None);
            MemoryStream ms = new MemoryStream((int)stream.Length);
            for (int i = 0; i < stream.Length; i++)
            {
                ms.WriteByte((byte)(i % 255));
            }

            long position = ms.Position;

            stream.CopyTo(ms);
            for (int i = 0; i < stream.Length; i++)
            {
                ms.WriteByte((byte)(i % 255));
            }

            ms.Seek(position, SeekOrigin.Begin);
            image2 = MLImage.CreateFromStream(ms);
            stream.Close();
            ms.Close();
            Assert.Equal(image1.Width, image2.Width);
            Assert.Equal(image1.Height, image2.Height);
            Assert.Equal(image1.Pixels.ToArray(), image2.Pixels.ToArray());
            image2.Dispose();
        }

        private class ReadOnlyNonSeekableStream : Stream
        {
            private Stream _stream;

            public ReadOnlyNonSeekableStream(Stream stream) => _stream = stream;

            public override bool CanRead => _stream.CanRead;

            public override bool CanSeek => false;

            public override bool CanWrite => false;

            public override long Length => _stream.Length;

            public override long Position { get => _stream.Position; set => throw new InvalidOperationException($"The stream is not seekable"); }

            public override void Flush() => _stream.Flush();

            public override int Read(byte[] buffer, int offset, int count) => _stream.Read(buffer, offset, count);

            public override long Seek(long offset, SeekOrigin origin) => throw new InvalidOperationException($"The stream is not seekable");

            public override void SetLength(long value) => throw new InvalidOperationException($"The stream is not seekable");

            public override void Write(byte[] buffer, int offset, int count) => throw new InvalidOperationException($"The stream is not writable");
        }
    }
}
