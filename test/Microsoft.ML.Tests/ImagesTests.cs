// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using System.IO;
using System.Linq;
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

            var pipe = new ImageLoadingEstimator(mlContext, imageFolder, ("ImageReal", "ImagePath"))
                .Append(new ImageResizingEstimator(mlContext, "ImageReal", 100, 100, "ImageReal"))
                .Append(mlContext.Transforms.ExtractPixels("ImagePixels", "ImageReal"))
                .Append(new ImageGrayscalingEstimator(mlContext, ("ImageGray", "ImageReal")));

            TestEstimatorCore(pipe, data, null, invalidData);
            Done();
        }

        [Fact]
        public void TestImagePixelExtractOptions()
        {
            var options1 = new ImagePixelExtractingEstimator.Options();
            Assert.Null(options1.ColumnOptions);

            var options2 = new ImagePixelExtractingEstimator.Options
            {
                ColumnOptions = new[]
                {
                    new ImagePixelExtractingEstimator.ColumnOptions { OutputColumnName = "outputColumn1", InputColumnName = "inputColumn1" },
                    new ImagePixelExtractingEstimator.ColumnOptions { OutputColumnName = "outputColumn2", InputColumnName = "inputColumn2" },
                    new ImagePixelExtractingEstimator.ColumnOptions { OutputColumnName = "outputColumn3", InputColumnName = "inputColumn3" }
                }
            };
            var options3 = new ImagePixelExtractingEstimator.Options(("outputColumn1", "inputColumn1"), ("outputColumn2", "inputColumn2"), ("outputColumn3", "inputColumn3"));
            var options4 = new ImagePixelExtractingEstimator.Options("outputColumn1", "outputColumn2", "outputColumn3");

            for (int i = 0; i < 3; i++)
            {
                // options2 and 3 should be exactly the same.
                Assert.True(options2.ColumnOptions[i].OutputColumnName == options3.ColumnOptions[i].OutputColumnName);
                Assert.True(options2.ColumnOptions[i].InputColumnName == options3.ColumnOptions[i].InputColumnName);

                // options4 should have the same OutputColumnName.
                Assert.True(options2.ColumnOptions[i].OutputColumnName == options4.ColumnOptions[i].OutputColumnName);
            }
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

            var pipe = new ImageLoadingEstimator(mlContext, imageFolder, ("ImageReal", "ImagePath"))
                .Append(new ImageResizingEstimator(mlContext, "ImageReal", 100, 100, "ImageReal"))
                .Append(mlContext.Transforms.ExtractPixels("ImagePixels", "ImageReal"))
                .Append(new ImageGrayscalingEstimator(mlContext, ("ImageGray", "ImageReal")));

            pipe.GetOutputSchema(SchemaShape.Create(data.Schema));
            var model = pipe.Fit(data);

            var tempPath = Path.GetTempFileName();
            using (var file = new SimpleFileHandle(mlContext, tempPath, true, true))
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

            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, interleavePixelColors: true, scaleImage: 2f/19, offsetImage: 30).Fit(cropped).Transform(cropped);
            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               ImagePixelExtractingEstimator.ColorBits.All, interleavedColors: true, scaleImage: 19/2f, offsetImage: -30).Transform(pixels);

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
            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", interleavePixelColors: true, scaleImage: 2f / 19, offsetImage: 30).Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               interleavedColors: true, scaleImage: 19 / 2f, offsetImage: -30).Transform(pixels);

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

            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, orderOfExtraction:ImagePixelExtractingEstimator.ColorsOrder.ABRG).Fit(cropped).Transform(cropped);
            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
               ImagePixelExtractingEstimator.ColorBits.All,orderOfColors: ImagePixelExtractingEstimator.ColorsOrder.ABRG).Transform(pixels);

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
            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, scaleImage: 2f / 19, offsetImage: 30).Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                ImagePixelExtractingEstimator.ColorBits.All, scaleImage: 19 / 2f, offsetImage: -30).Transform(pixels);

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
            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", scaleImage: 2f / 19, offsetImage: 30).Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                scaleImage: 19 / 2f, offsetImage: -30).Transform(pixels);

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

            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, interleavePixelColors: true).Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels",
                ImagePixelExtractingEstimator.ColorBits.All, interleavedColors: true).Transform(pixels);

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

            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", interleavePixelColors: true).Fit(cropped).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageConvertingTransformer(mlContext, "ImageRestored", imageHeight, imageWidth, "ImagePixels", interleavedColors: true).Transform(pixels);

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

            var pixels = mlContext.Transforms.ExtractPixels("ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All).Fit(cropped).Transform(cropped);

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
            var pipe = new ImageLoadingEstimator(mlContext, imageFolder, ("ImageReal", "ImagePath"))
                .Append(new ImageResizingEstimator(mlContext, "ImageReal", targetDimension, targetDimension, "ImageReal",
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
