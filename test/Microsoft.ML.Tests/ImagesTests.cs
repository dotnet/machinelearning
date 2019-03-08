// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class ImageTests : TestDataPipeBase
    {
        public ImageTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestEstimatorChain()
        {
            var mlContext = new MLContext();
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));
            var invalidData = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.Single, 0),
                    }
            }, new MultiFileSource(dataFile));

            var pipe = mlContext.Transforms.LoadImages(imageFolder, ("ImageReal", "ImagePath"))
                .Append(mlContext.Transforms.ResizeImages("ImageReal", 100, 100, "ImageReal"))
                .Append(mlContext.Transforms.ExtractPixels("ImagePixels", "ImageReal"))
                .Append(mlContext.Transforms.ConvertToGrayscale(("ImageGray", "ImageReal")));

            TestEstimatorCore(pipe, data, null, invalidData);
            Done();
        }

        [Fact]
        public void TestEstimatorSaveLoad()
        {
            var mlContext = new MLContext();
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));

            var pipe = mlContext.Transforms.LoadImages(imageFolder, ("ImageReal", "ImagePath"))
                .Append(mlContext.Transforms.ResizeImages("ImageReal", 100, 100, "ImageReal"))
                .Append(mlContext.Transforms.ExtractPixels("ImagePixels", "ImageReal"))
                .Append(mlContext.Transforms.ConvertToGrayscale(("ImageGray", "ImageReal")));

            pipe.GetOutputSchema(SchemaShape.Create(data.Schema));
            var model = pipe.Fit(data);

            var tempPath = Path.GetTempFileName();
            using (var file = new SimpleFileHandle(mlContext, tempPath, true, true))
            {
                using (var fs = file.CreateWriteStream())
                    model.SaveTo(mlContext, fs);
                var model2 = TransformerChain.LoadFrom(mlContext, file.OpenReadStream());

                var newCols = ((ImageLoadingTransformer)model2.First()).Columns;
                var oldCols = ((ImageLoadingTransformer)model.First()).Columns;
                Assert.True(newCols
                    .Zip(oldCols, (x, y) => x == y)
                    .All(x => x));
            }
            Done();
        }

        [Fact]
        public void TestSaveImages()
        {
            var mlContext = new MLContext();
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", 100, 100, "ImageReal", ImageResizingEstimator.ResizingKind.IsoPad).Transform(images);

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
            var mlContext = new MLContext();
            var imageHeight = 150;
            var imageWidth = 100;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);

            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            IDataView grey = new ImageGrayscalingTransformer(mlContext, ("ImageGrey", "ImageCropped")).Transform(cropped);
            var fname = nameof(TestGreyscaleTransformImages) + "_model.zip";

            var fh = mlContext.CreateOutputFile(fname);
            using (var ch = ((IHostEnvironment) mlContext).Start("save"))
                TrainUtils.SaveModel(mlContext, ch, fh, null, new RoleMappedData(grey));

            grey = ModelFileUtils.LoadPipeline(mlContext, fh.OpenReadStream(), new MultiFileSource(dataFile));
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
        public void TestBackAndForthConversionWithAlphaInterleave()
        {
            var mlContext = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, interleave: true, scale: 2f/19, offset: 30).Fit(cropped).Transform(cropped);
            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               ImagePixelExtractingEstimator.ColorBits.All, interleave: true, scale: 19/2f, offset: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaInterleave) + "_model.zip";

            var fh = mlContext.CreateOutputFile(fname);
            using (var ch = ((IHostEnvironment) mlContext).Start("save"))
                TrainUtils.SaveModel(mlContext, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(mlContext, fh.OpenReadStream(), new MultiFileSource(dataFile));
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
            var mlContext = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", interleave: true, scale: 2f / 19, offset: 30).Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               interleave: true, scale: 19 / 2f, offset: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaInterleave) + "_model.zip";

            var fh = mlContext.CreateOutputFile(fname);
            using (var ch = ((IHostEnvironment) mlContext).Start("save"))
                TrainUtils.SaveModel(mlContext, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(mlContext, fh.OpenReadStream(), new MultiFileSource(dataFile));
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
            var mlContext = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, order:ImagePixelExtractingEstimator.ColorsOrder.ABRG).Fit(cropped).Transform(cropped);
            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               ImagePixelExtractingEstimator.ColorBits.All,order: ImagePixelExtractingEstimator.ColorsOrder.ABRG).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithDifferentOrder) + "_model.zip";

            var fh = mlContext.CreateOutputFile(fname);
            using (var ch = ((IHostEnvironment) mlContext).Start("save"))
                TrainUtils.SaveModel(mlContext, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(mlContext, fh.OpenReadStream(), new MultiFileSource(dataFile));
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

        [Fact]
        public void TestBackAndForthConversionWithAlphaNoInterleave()
        {
            var mlContext = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, scale: 2f / 19, offset: 30).Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                ImagePixelExtractingEstimator.ColorBits.All, scale: 19 / 2f, offset: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaNoInterleave) + "_model.zip";

            var fh = mlContext.CreateOutputFile(fname);
            using (var ch = ((IHostEnvironment) mlContext).Start("save"))
                TrainUtils.SaveModel(mlContext, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(mlContext, fh.OpenReadStream(), new MultiFileSource(dataFile));
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
        public void TestBackAndForthConversionWithoutAlphaNoInterleave()
        {
            var mlContext = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", scale: 2f / 19, offset: 30).Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                scale: 19 / 2f, offset: -30).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaNoInterleave) + "_model.zip";

            var fh = mlContext.CreateOutputFile(fname);
            using (var ch = ((IHostEnvironment) mlContext).Start("save"))
                TrainUtils.SaveModel(mlContext, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(mlContext, fh.OpenReadStream(), new MultiFileSource(dataFile));
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
        public void TestBackAndForthConversionWithAlphaInterleaveNoOffset()
        {
            var mlContext = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, interleave: true).Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                ImagePixelExtractingEstimator.ColorBits.All, interleave: true).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaInterleaveNoOffset) + "_model.zip";

            var fh = mlContext.CreateOutputFile(fname);
            using (var ch = ((IHostEnvironment) mlContext).Start("save"))
                TrainUtils.SaveModel(mlContext, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(mlContext, fh.OpenReadStream(), new MultiFileSource(dataFile));
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
        public void TestBackAndForthConversionWithoutAlphaInterleaveNoOffset()
        {
            var mlContext = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", interleave: true).Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels", interleave: true).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaInterleaveNoOffset) + "_model.zip";

            var fh = mlContext.CreateOutputFile(fname);
            using (var ch = ((IHostEnvironment) mlContext).Start("save"))
                TrainUtils.SaveModel(mlContext, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(mlContext, fh.OpenReadStream(), new MultiFileSource(dataFile));
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
        public void TestBackAndForthConversionWithAlphaNoInterleaveNoOffset()
        {
            var mlContext = new MLContext();
            const int imageHeight = 100;
            var imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels =  mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All).Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                 ImagePixelExtractingEstimator.ColorBits.All).Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaNoInterleaveNoOffset) + "_model.zip";

            var fh = mlContext.CreateOutputFile(fname);
            using (var ch = ((IHostEnvironment) mlContext).Start("save"))
                TrainUtils.SaveModel(mlContext, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(mlContext, fh.OpenReadStream(), new MultiFileSource(dataFile));
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
            var mlContext = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped").Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels").Transform(pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaNoInterleaveNoOffset) + "_model.zip";

            var fh = mlContext.CreateOutputFile(fname);
            using (var ch = ((IHostEnvironment) mlContext).Start("save"))
                TrainUtils.SaveModel(mlContext, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(mlContext, fh.OpenReadStream(), new MultiFileSource(dataFile));
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
            var mlContext = new MLContext();
            var dataFile = GetDataPath("images/fillmode.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0)
                }
            }, new MultiFileSource(dataFile));

            const int targetDimension = 50;
            var pipe = mlContext.Transforms.LoadImages(imageFolder, ("ImageReal", "ImagePath"))
                .Append(mlContext.Transforms.ResizeImages("ImageReal", targetDimension, targetDimension, "ImageReal",
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
    }
}
