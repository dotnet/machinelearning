// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
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
            var env = new MLContext();
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var invalidData = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.R4, 0),
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
            IHostEnvironment env = new MLContext();
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
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
                    model.SaveTo(env, fs);
                var model2 = TransformerChain.LoadFrom(env, file.OpenReadStream());

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
            var env = new MLContext();
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", 100, 100, "ImageReal", ImageResizingEstimator.ResizingKind.IsoPad).Transform(images);

            cropped.Schema.TryGetColumnIndex("ImagePath", out int pathColumn);
            cropped.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
            using (var cursor = cropped.GetRowCursorForAllColumns())
            {
                var pathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(pathColumn);
                ReadOnlyMemory<char> path = default;
                var bitmapCropGetter = cursor.GetGetter<Bitmap>(cropBitmapColumn);
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
            IHostEnvironment env = new MLContext();
            var imageHeight = 150;
            var imageWidth = 100;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
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
                var bitmapGetter = cursor.GetGetter<Bitmap>(greyColumn);
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
            IHostEnvironment env = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, true, 2f / 255, 127.5f).Transform(cropped);
            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = true,
                Offset = -1f,
                Scale = 255f / 2,
                Columns = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Name = "ImageRestored" , Source= "ImagePixels", ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=true}
                    }
            }, pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);


            backToBitmaps.Schema.TryGetColumnIndex("ImageRestored", out int bitmapColumn);
            backToBitmaps.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(bitmapColumn);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(cropBitmapColumn);
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
            IHostEnvironment env = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.Rgb, true, 2f / 255, 127.5f).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = true,
                Offset = -1f,
                Scale = 255f / 2,
                Columns = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Name = "ImageRestored" , Source= "ImagePixels", ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=false}
                    }
            }, pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);


            backToBitmaps.Schema.TryGetColumnIndex("ImageRestored", out int bitmapColumn);
            backToBitmaps.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(bitmapColumn);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(cropBitmapColumn);
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
        public void TestBackAndForthConversionWithAlphaNoInterleave()
        {
            IHostEnvironment env = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, false, 2f / 255, 127.5f).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = false,
                Offset = -1f,
                Scale = 255f / 2,
                Columns = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Name = "ImageRestored" , Source= "ImagePixels", ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=true}
                    }
            }, pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaNoInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);


            backToBitmaps.Schema.TryGetColumnIndex("ImageRestored", out int bitmapColumn);
            backToBitmaps.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(bitmapColumn);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(cropBitmapColumn);
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
            IHostEnvironment env = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.Rgb, false, 2f / 255, 127.5f).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = false,
                Offset = -1f,
                Scale = 255f / 2,
                Columns = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Name = "ImageRestored" , Source= "ImagePixels", ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=false}
                    }
            }, pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaNoInterleave) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);


            backToBitmaps.Schema.TryGetColumnIndex("ImageRestored", out int bitmapColumn);
            backToBitmaps.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(bitmapColumn);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(cropBitmapColumn);
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
            IHostEnvironment env = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All, true).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = true,
                Columns = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Name = "ImageRestored" , Source= "ImagePixels", ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=true}
                    }
            }, pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);


            backToBitmaps.Schema.TryGetColumnIndex("ImageRestored", out int bitmapColumn);
            backToBitmaps.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(bitmapColumn);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(cropBitmapColumn);
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
            IHostEnvironment env = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.Rgb, true).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = true,
                Columns = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Name = "ImageRestored" , Source= "ImagePixels", ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=false}
                    }
            }, pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);


            backToBitmaps.Schema.TryGetColumnIndex("ImageRestored", out int bitmapColumn);
            backToBitmaps.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(bitmapColumn);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(cropBitmapColumn);
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
            IHostEnvironment env = new MLContext();
            const int imageHeight = 100;
            var imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);

            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped", ImagePixelExtractingEstimator.ColorBits.All).Transform(cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = false,
                Columns = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Name = "ImageRestored" , Source= "ImagePixels", ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=true}
                    }
            }, pixels);

            var fname = nameof(TestBackAndForthConversionWithAlphaNoInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);


            backToBitmaps.Schema.TryGetColumnIndex("ImageRestored", out int bitmapColumn);
            backToBitmaps.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(bitmapColumn);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(cropBitmapColumn);
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
            IHostEnvironment env = new MLContext();
            const int imageHeight = 100;
            const int imageWidth = 130;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(env, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(env, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = new ImagePixelExtractingTransformer(env, "ImagePixels", "ImageCropped").Transform(cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = false,
                Columns = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Name = "ImageRestored" , Source= "ImagePixels", ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=false}
                    }
            }, pixels);

            var fname = nameof(TestBackAndForthConversionWithoutAlphaNoInterleaveNoOffset) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(backToBitmaps));

            backToBitmaps = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);


            backToBitmaps.Schema.TryGetColumnIndex("ImageRestored", out int bitmapColumn);
            backToBitmaps.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
            using (var cursor = backToBitmaps.GetRowCursorForAllColumns())
            {
                var bitmapGetter = cursor.GetGetter<Bitmap>(bitmapColumn);
                Bitmap restoredBitmap = default;

                var bitmapCropGetter = cursor.GetGetter<Bitmap>(cropBitmapColumn);
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
            var env = new MLContext();
            var dataFile = GetDataPath("images/fillmode.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.TX, 0)
                }
            }, new MultiFileSource(dataFile));

            const int targetDimension = 50;
            var pipe = new ImageLoadingEstimator(env, imageFolder, ("ImageReal", "ImagePath"))
                .Append(new ImageResizingEstimator(env, "ImageReal", targetDimension, targetDimension, "ImageReal",
                    resizing: ImageResizingEstimator.ResizingKind.Fill));

            var rowView = pipe.Preview(data).RowView;
            Assert.Single(rowView);

            using (var bitmap = (Bitmap) rowView.First().Values.Last().Value)
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
