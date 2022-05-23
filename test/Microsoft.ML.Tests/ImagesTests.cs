// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
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
                var bitmapCropGetter = cursor.GetGetter<Bitmap>(cropped.Schema["ImageCropped"]);
                Bitmap bitmap = default;
                while (cursor.MoveNext())
                {
                    pathGetter(ref path);
                    bitmapCropGetter(ref bitmap);
                    Assert.NotNull(bitmap);
                    var fileToSave = GetOutputPath(Path.GetFileNameWithoutExtension(path.ToString()) + ".cropped.jpg");
                    bitmap.Save(fileToSave, System.Drawing.Imaging.ImageFormat.Jpeg);
                }
            }
            Done();
        }

        [Fact]
        public void TestGreyscaleTransformImages()
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
            var fname = nameof(TestGreyscaleTransformImages) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(grey));

            grey = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            grey.Schema.TryGetColumnIndex("ImageGrey", out int greyColumn);
            using (var cursor = grey.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(grey.Schema["ImageGrey"]);
                Bitmap bitmap = default;
                while (cursor.MoveNext())
                {
                    bitmapGetter(ref bitmap);
                    Assert.NotNull(bitmap);
                    for (int x = 0; x < imageWidth; x++)
                        for (int y = 0; y < imageHeight; y++)
                        {
                            var pixel = bitmap.GetPixel(x, y);
                            // greyscale image has same values for R,G and B
                            Assert.True(pixel.R == pixel.G && pixel.G == pixel.B);
                        }
                }
            }
            Done();
        }

        [Fact]
        public void TestGrayScaleInMemory()
        {
            // Create an image list.
            var images = new List<ImageDataPoint>() { new ImageDataPoint(10, 10, Color.Blue), new ImageDataPoint(10, 10, Color.Red) };

            // Convert the list of data points to an IDataView object, which is consumable by ML.NET API.
            var data = ML.Data.LoadFromEnumerable(images);

            // Convert image to gray scale.
            var pipeline = ML.Transforms.ConvertToGrayscale("GrayImage", "Image");

            // Fit the model.
            var model = pipeline.Fit(data);

            // Test path: image files -> IDataView -> Enumerable of Bitmaps.
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

                for (int x = 0; x < grayImage.Width; ++x)
                {
                    for (int y = 0; y < grayImage.Height; ++y)
                    {
                        var pixel = grayImage.GetPixel(x, y);
                        // greyscale image has same values for R, G and B.
                        Assert.True(pixel.R == pixel.G && pixel.G == pixel.B);
                    }
                }
            }

            var engine = ML.Model.CreatePredictionEngine<ImageDataPoint, ImageDataPoint>(model);
            var singleImage = new ImageDataPoint(17, 36, Color.Pink);
            var transformedSingleImage = engine.Predict(singleImage);

            Assert.Equal(singleImage.Image.Height, transformedSingleImage.GrayImage.Height);
            Assert.Equal(singleImage.Image.Width, transformedSingleImage.GrayImage.Width);

            for (int x = 0; x < transformedSingleImage.GrayImage.Width; ++x)
            {
                for (int y = 0; y < transformedSingleImage.GrayImage.Height; ++y)
                {
                    var pixel = transformedSingleImage.GrayImage.GetPixel(x, y);
                    // greyscale image has same values for R, G and B.
                    Assert.True(pixel.R == pixel.G && pixel.G == pixel.B);
                }
            }
        }

        private class ImageDataPoint
        {
            [ImageType(10, 10)]
            public Bitmap Image { get; set; }

            [ImageType(10, 10)]
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
            IDataView backToBitmaps = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               ImagePixelExtractingEstimator.ColorBits.All, interleavedColors: true, scaleImage: 19 / 2f, offsetImage: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageRestored"]);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageCropped"]);
                Bitmap croppedBitmap = default;
                while (cursor.MoveNext())
                {
                    bitmapGetter(ref restoredBitmap);
                    Assert.NotNull(restoredBitmap);
                    bitmapCropGetter(ref croppedBitmap);
                    Assert.NotNull(croppedBitmap);
                    for (int x = 0; x < imageWidth; x++)
                        for (int y = 0; y < imageHeight; y++)
                        {
                            var c = croppedBitmap.GetPixel(x, y);
                            var r = restoredBitmap.GetPixel(x, y);
                            Assert.True(c == r);
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

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               interleavedColors: true, scaleImage: 19 / 2f, offsetImage: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageRestored"]);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageCropped"]);
                Bitmap croppedBitmap = default;
                while (cursor.MoveNext())
                {
                    bitmapGetter(ref restoredBitmap);
                    Assert.NotNull(restoredBitmap);
                    bitmapCropGetter(ref croppedBitmap);
                    Assert.NotNull(croppedBitmap);
                    for (int x = 0; x < imageWidth; x++)
                        for (int y = 0; y < imageHeight; y++)
                        {
                            var c = croppedBitmap.GetPixel(x, y);
                            var r = restoredBitmap.GetPixel(x, y);
                            Assert.True(c.R == r.R && c.G == r.G && c.B == r.B);
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
            IDataView backToBitmaps = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               ImagePixelExtractingEstimator.ColorBits.All, orderOfColors: ImagePixelExtractingEstimator.ColorsOrder.ABRG).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithDifferentOrder) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageRestored"]);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageCropped"]);
                Bitmap croppedBitmap = default;
                while (cursor.MoveNext())
                {
                    bitmapGetter(ref restoredBitmap);
                    Assert.NotNull(restoredBitmap);
                    bitmapCropGetter(ref croppedBitmap);
                    Assert.NotNull(croppedBitmap);
                    for (int x = 0; x < imageWidth; x++)
                        for (int y = 0; y < imageHeight; y++)
                        {
                            var c = croppedBitmap.GetPixel(x, y);
                            var r = restoredBitmap.GetPixel(x, y);
                            if (c != r)
                                Assert.False(true);
                            Assert.True(c == r);
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

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                ImagePixelExtractingEstimator.ColorBits.All, scaleImage: 19 / 2f, offsetImage: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaNoInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageRestored"]);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageCropped"]);
                Bitmap croppedBitmap = default;
                while (cursor.MoveNext())
                {
                    bitmapGetter(ref restoredBitmap);
                    Assert.NotNull(restoredBitmap);
                    bitmapCropGetter(ref croppedBitmap);
                    Assert.NotNull(croppedBitmap);
                    for (int x = 0; x < imageWidth; x++)
                        for (int y = 0; y < imageHeight; y++)
                        {
                            var c = croppedBitmap.GetPixel(x, y);
                            var r = restoredBitmap.GetPixel(x, y);
                            Assert.True(c == r);
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

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                scaleImage: 19 / 2f, offsetImage: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaNoInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageRestored"]);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageCropped"]);
                Bitmap croppedBitmap = default;
                while (cursor.MoveNext())
                {
                    bitmapGetter(ref restoredBitmap);
                    Assert.NotNull(restoredBitmap);
                    bitmapCropGetter(ref croppedBitmap);
                    Assert.NotNull(croppedBitmap);
                    for (int x = 0; x < imageWidth; x++)
                        for (int y = 0; y < imageHeight; y++)
                        {
                            var c = croppedBitmap.GetPixel(x, y);
                            var r = restoredBitmap.GetPixel(x, y);
                            Assert.True(c.R == r.R && c.G == r.G && c.B == r.B);
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

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                ImagePixelExtractingEstimator.ColorBits.All, interleavedColors: true).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageRestored"]);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageCropped"]);
                Bitmap croppedBitmap = default;
                while (cursor.MoveNext())
                {
                    bitmapGetter(ref restoredBitmap);
                    Assert.NotNull(restoredBitmap);
                    bitmapCropGetter(ref croppedBitmap);
                    Assert.NotNull(croppedBitmap);
                    for (int x = 0; x < imageWidth; x++)
                        for (int y = 0; y < imageHeight; y++)
                        {
                            var c = croppedBitmap.GetPixel(x, y);
                            var r = restoredBitmap.GetPixel(x, y);
                            Assert.True(c == r);
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

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels", interleavedColors: true).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageRestored"]);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageCropped"]);
                Bitmap croppedBitmap = default;
                while (cursor.MoveNext())
                {
                    bitmapGetter(ref restoredBitmap);
                    Assert.NotNull(restoredBitmap);
                    bitmapCropGetter(ref croppedBitmap);
                    Assert.NotNull(croppedBitmap);
                    for (int x = 0; x < imageWidth; x++)
                        for (int y = 0; y < imageHeight; y++)
                        {
                            var c = croppedBitmap.GetPixel(x, y);
                            var r = restoredBitmap.GetPixel(x, y);
                            Assert.True(c.R == r.R && c.G == r.G && c.B == r.B);
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

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                 ImagePixelExtractingEstimator.ColorBits.All).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaNoInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageRestored"]);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageCropped"]);
                Bitmap croppedBitmap = default;
                while (cursor.MoveNext())
                {
                    bitmapGetter(ref restoredBitmap);
                    Assert.NotNull(restoredBitmap);
                    bitmapCropGetter(ref croppedBitmap);
                    Assert.NotNull(croppedBitmap);
                    for (int x = 0; x < imageWidth; x++)
                        for (int y = 0; y < imageHeight; y++)
                        {
                            var c = croppedBitmap.GetPixel(x, y);
                            var r = restoredBitmap.GetPixel(x, y);
                            Assert.True(c == r);
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

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(env, "ImageRestored", imageHeight, imageWidth, "ImagePixels").Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaNoInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageRestored"]);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(backToBitmaps.Schema["ImageCropped"]);
                Bitmap croppedBitmap = default;
                while (cursor.MoveNext())
                {
                    bitmapGetter(ref restoredBitmap);
                    Assert.NotNull(restoredBitmap);
                    bitmapCropGetter(ref croppedBitmap);
                    Assert.NotNull(croppedBitmap);
                    for (int x = 0; x < imageWidth; x++)
                        for (int y = 0; y < imageHeight; y++)
                        {
                            var c = croppedBitmap.GetPixel(x, y);
                            var r = restoredBitmap.GetPixel(x, y);
                            Assert.True(c.R == r.R && c.G == r.G && c.B == r.B);
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

            using (var bitmap = (Bitmap)rowView.First().Values.Last().Value)
            {
                // these points must be white
                var topLeft = bitmap.GetPixel(0, 0);
                var topRight = bitmap.GetPixel(bitmap.Width - 1, 0);
                var bottomLeft = bitmap.GetPixel(0, bitmap.Height - 1);
                var bottomRight = bitmap.GetPixel(bitmap.Width - 1, bitmap.Height - 1);
                var middle = bitmap.GetPixel(bitmap.Width / 2, bitmap.Height / 2);

                // these points must be red
                var midTop = bitmap.GetPixel(bitmap.Width / 2, bitmap.Height / 3);
                var midBottom = bitmap.GetPixel(bitmap.Width / 2, bitmap.Height / 3 * 2);
                var leftMid = bitmap.GetPixel(bitmap.Width / 3, bitmap.Height / 2);
                var rightMid = bitmap.GetPixel(bitmap.Width / 3 * 2, bitmap.Height / 2);

                // it turns out rounding errors on certain platforms may lead to a test failure
                // instead of checking for exactly FFFFFF and FF0000 we allow a small interval here to be safe
                Assert.All(new[] { topLeft, topRight, bottomLeft, bottomRight, middle }, c =>
                {
                    Assert.True(c.R >= 250);
                    Assert.True(c.G >= 250);
                    Assert.True(c.B >= 250);
                });
                Assert.All(new[] { midTop, midBottom, leftMid, rightMid }, c =>
                {
                    Assert.True(c.R >= 250);
                    Assert.True(c.G < 6);
                    Assert.True(c.B < 6);
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
            public Bitmap LoadedImage;
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
                                    LoadedImage = (Bitmap)Image.FromFile(imagePath)
                                }
                            );
                    }
                }

                return inMemoryImages;

            }
        }

        public class InMemoryImageOutput : InMemoryImage
        {
            [ImageType(100, 100)]
            public Bitmap ResizedImage;
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
            var resizedImage = (Bitmap)rowView.First().Values.Last().Value;
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
    }
}
