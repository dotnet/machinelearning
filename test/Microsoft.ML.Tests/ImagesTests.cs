﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using System;
using System.Drawing;
using System.IO;
using System.Linq;
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
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var invalidData = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.R4, 0),
                    }
            }, new MultiFileSource(dataFile));

            var pipe = new ImageLoadingEstimator(env, imageFolder, ("ImagePath", "ImageReal"))
                .Append(new ImageResizingEstimator(env, "ImageReal", "ImageReal", 100, 100))
                .Append(new ImagePixelExtractingEstimator(env, "ImageReal", "ImagePixels"))
                .Append(new ImageGrayscalingEstimator(env, ("ImageReal", "ImageGray")));

            TestEstimatorCore(pipe, data, null, invalidData);
            Done();
        }

        [Fact]
        public void TestEstimatorSaveLoad()
        {
            IHostEnvironment env = new MLContext();
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));

            var pipe = new ImageLoadingEstimator(env, imageFolder, ("ImagePath", "ImageReal"))
                .Append(new ImageResizingEstimator(env, "ImageReal", "ImageReal", 100, 100))
                .Append(new ImagePixelExtractingEstimator(env, "ImageReal", "ImagePixels"))
                .Append(new ImageGrayscalingEstimator(env, ("ImageReal", "ImageGray")));

            pipe.GetOutputSchema(Core.Data.SchemaShape.Create(data.Schema));
            var model = pipe.Fit(data);

            var tempPath = Path.GetTempFileName();
            using (var file = new SimpleFileHandle(env, tempPath, true, true))
            {
                using (var fs = file.CreateWriteStream())
                    model.SaveTo(env, fs);
                var model2 = TransformerChain.LoadFrom(env, file.OpenReadStream());

                var newCols = ((ImageLoaderTransform)model2.First()).Columns;
                var oldCols = ((ImageLoaderTransform)model.First()).Columns;
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
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);

            IDataView cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() {  Name= "ImageCropped", Source = "ImageReal", ImageHeight =100, ImageWidth = 100, Resizing = ImageResizerTransform.ResizingKind.IsoPad}
                    }
            }, images);

            cropped.Schema.TryGetColumnIndex("ImagePath", out int pathColumn);
            cropped.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
            using (var cursor = cropped.GetRowCursor((x) => true))
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
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() {  Name= "ImageCropped", Source = "ImageReal", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);

            IDataView grey = ImageGrayscaleTransform.Create(env, new ImageGrayscaleTransform.Arguments()
            {
                Column = new ImageGrayscaleTransform.Column[1]{
                        new ImageGrayscaleTransform.Column() {  Name= "ImageGrey", Source = "ImageCropped"}
                    }
            }, cropped);

            var fname = nameof(TestGreyscaleTransformImages) + "_model.zip";

            var fh = env.CreateOutputFile(fname);
            using (var ch = env.Start("save"))
                TrainUtils.SaveModel(env, ch, fh, null, new RoleMappedData(grey));

            grey = ModelFileUtils.LoadPipeline(env, fh.OpenReadStream(), new MultiFileSource(dataFile));
            DeleteOutputPath(fname);

            grey.Schema.TryGetColumnIndex("ImageGrey", out int greyColumn);
            using (var cursor = grey.GetRowCursor((x) => true))
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
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);

            var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
            {
                InterleaveArgb = true,
                Offset = 127.5f,
                Scale = 2f / 255,
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "ImagePixels", UseAlpha=true}
                    }
            }, cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = true,
                Offset = -1f,
                Scale = 255f / 2,
                Column = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Source= "ImagePixels", Name = "ImageRestored" , ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=true}
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
            using (var cursor = backToBitmaps.GetRowCursor((x) => true))
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
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);

            var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
            {
                InterleaveArgb = true,
                Offset = 127.5f,
                Scale = 2f / 255,
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "ImagePixels", UseAlpha=false}
                    }
            }, cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = true,
                Offset = -1f,
                Scale = 255f / 2,
                Column = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Source= "ImagePixels", Name = "ImageRestored" , ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=false}
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
            using (var cursor = backToBitmaps.GetRowCursor((x) => true))
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
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);

            var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
            {
                InterleaveArgb = false,
                Offset = 127.5f,
                Scale = 2f / 255,
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "ImagePixels", UseAlpha=true}
                    }
            }, cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = false,
                Offset = -1f,
                Scale = 255f / 2,
                Column = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Source= "ImagePixels", Name = "ImageRestored" , ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=true}
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
            using (var cursor = backToBitmaps.GetRowCursor((x) => true))
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
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);

            var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
            {
                InterleaveArgb = false,
                Offset = 127.5f,
                Scale = 2f / 255,
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "ImagePixels", UseAlpha=false}
                    }
            }, cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = false,
                Offset = -1f,
                Scale = 255f / 2,
                Column = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Source= "ImagePixels", Name = "ImageRestored" , ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=false}
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
            using (var cursor = backToBitmaps.GetRowCursor((x) => true))
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
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);

            var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
            {
                InterleaveArgb = true,
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "ImagePixels", UseAlpha=true}
                    }
            }, cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = true,
                Column = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Source= "ImagePixels", Name = "ImageRestored" , ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=true}
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
            using (var cursor = backToBitmaps.GetRowCursor((x) => true))
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
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);

            var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
            {
                InterleaveArgb = true,
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "ImagePixels", UseAlpha=false}
                    }
            }, cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = true,
                Column = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Source= "ImagePixels", Name = "ImageRestored" , ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=false}
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
            using (var cursor = backToBitmaps.GetRowCursor((x) => true))
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
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);

            var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
            {
                InterleaveArgb = false,
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "ImagePixels", UseAlpha=true}
                    }
            }, cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = false,
                Column = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Source= "ImagePixels", Name = "ImageRestored" , ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=true}
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
            using (var cursor = backToBitmaps.GetRowCursor((x) => true))
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
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);

            var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
            {
                InterleaveArgb = false,
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "ImagePixels", UseAlpha=false}
                    }
            }, cropped);

            IDataView backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
            {
                InterleaveArgb = false,
                Column = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Source= "ImagePixels", Name = "ImageRestored" , ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=false}
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
            using (var cursor = backToBitmaps.GetRowCursor((x) => true))
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
    }
}
