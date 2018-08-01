﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.TestFramework;
using System.Drawing;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class ImageTests : BaseTestClass
    {
        public ImageTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestSaveImages()
        {
            using (var env = new TlcEnvironment())
            {
                var dataFile = GetDataPath("images/images.tsv");
                var imageFolder = Path.GetDirectoryName(dataFile);
                var data = env.CreateLoader("Text{col=ImagePath:TX:0 col=Name:TX:1}", new MultiFileSource(dataFile));
                var images = new ImageLoaderTransform(env, new ImageLoaderTransform.Arguments()
                {
                    Column = new ImageLoaderTransform.Column[1]
                    {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                    },
                    ImageFolder = imageFolder
                }, data);

                var cropped = new ImageResizerTransform(env, new ImageResizerTransform.Arguments()
                {
                    Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() {  Name= "ImageCropped", Source = "ImageReal", ImageHeight =100, ImageWidth = 100, Resizing = ImageResizerTransform.ResizingKind.IsoPad}
                    }
                }, images);

                cropped.Schema.TryGetColumnIndex("ImagePath", out int pathColumn);
                cropped.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
                using (var cursor = cropped.GetRowCursor((x) => true))
                {
                    var pathGetter = cursor.GetGetter<DvText>(pathColumn);
                    DvText path = default;
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
            }
        }

        [Fact]
        public void TestGreyscaleTransformImages()
        {
            using (var env = new TlcEnvironment())
            {
                var imageHeight = 150;
                var imageWidth = 100;
                var dataFile = GetDataPath("images/images.tsv");
                var imageFolder = Path.GetDirectoryName(dataFile);
                var data = env.CreateLoader("Text{col=ImagePath:TX:0 col=Name:TX:1}", new MultiFileSource(dataFile));
                var images = new ImageLoaderTransform(env, new ImageLoaderTransform.Arguments()
                {
                    Column = new ImageLoaderTransform.Column[1]
                    {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                    },
                    ImageFolder = imageFolder
                }, data);
                var cropped = new ImageResizerTransform(env, new ImageResizerTransform.Arguments()
                {
                    Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() {  Name= "ImageCropped", Source = "ImageReal", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
                }, images);

                var grey = new ImageGrayscaleTransform(env, new ImageGrayscaleTransform.Arguments()
                {
                    Column = new ImageGrayscaleTransform.Column[1]{
                        new ImageGrayscaleTransform.Column() {  Name= "ImageGrey", Source = "ImageCropped"}
                    }
                }, cropped);

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
            }
        }

        [Fact]
        public void TestBackAndForthConversion()
        {
            using (var env = new TlcEnvironment())
            {
                var imageHeight = 100;
                var imageWidth = 130;
                var dataFile = GetDataPath("images/images.tsv");
                var imageFolder = Path.GetDirectoryName(dataFile);
                var data = env.CreateLoader("Text{col=ImagePath:TX:0 col=Name:TX:1}", new MultiFileSource(dataFile));
                var images = new ImageLoaderTransform(env, new ImageLoaderTransform.Arguments()
                {
                    Column = new ImageLoaderTransform.Column[1]
                    {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                    },
                    ImageFolder = imageFolder
                }, data);
                var cropped = new ImageResizerTransform(env, new ImageResizerTransform.Arguments()
                {
                    Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
                }, images);

                var pixels = new ImagePixelExtractorTransform(env, new ImagePixelExtractorTransform.Arguments()
                {
                    Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "ImagePixels", UseAlpha=true}
                    }
                }, cropped);

                var backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
                {
                    Column = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Source= "ImagePixels", Name = "ImageRestored" , ImageHeight=imageHeight, ImageWidth=imageWidth, ContainsAlpha=true}
                    }
                }, pixels);

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
                                Assert.True(croppedBitmap.GetPixel(x, y) == restoredBitmap.GetPixel(x, y));
                            }
                    }
                }
            }
        }
    }
}
