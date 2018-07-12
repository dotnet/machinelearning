using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
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
        public void TestImages()
        {
            using (var env = new TlcEnvironment())
            {
                var dataFile = GetDataPath("images/images.tsv");
                var data = env.CreateLoader("Text{col=ImagePath:TX:0 col=Name:TX:1}", new MultiFileSource(dataFile));
                var images = new ImageLoaderTransform(env, new ImageLoaderTransform.Arguments()
                {
                    Column = new ImageLoaderTransform.Column[1]
                    {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                    },
                    ImageFolder = Path.GetDirectoryName(dataFile)
                }, data);
                var cropped = new ImageResizerTransform(env, new ImageResizerTransform.Arguments()
                {
                    Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() {  Name= "ImageCropped", Source = "ImageReal", ImageHeight =100, ImageWidth = 100, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
                }, images);

                var pixels = new ImagePixelExtractorTransform(env, new ImagePixelExtractorTransform.Arguments()
                {
                    Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "ImagePixels"}
                    }
                }, cropped);


                var backToBitmaps = new VectorToImageTransform(env, new VectorToImageTransform.Arguments()
                {
                    Column = new VectorToImageTransform.Column[1]{
                        new VectorToImageTransform.Column() {  Source= "ImagePixels", Name = "ImageBitmaps" , ImageHeight=100, ImageWidth=100}
                    }
                }, pixels);

                backToBitmaps.Schema.TryGetColumnIndex("ImagePixels", out int cropColumn);
                backToBitmaps.Schema.TryGetColumnIndex("ImageBitmaps", out int bitmapColumn);
                backToBitmaps.Schema.TryGetColumnIndex("ImageCropped", out int cropBitmapColumn);
                using (var cursor = backToBitmaps.GetRowCursor((x) => true))
                {
                    var pixelsGetter = cursor.GetGetter<VBuffer<float>>(cropColumn);
                    VBuffer<float> pixelcolumn = new VBuffer<float>();
                    var bitmapGetter = cursor.GetGetter<Bitmap>(bitmapColumn);
                    Bitmap bitmapcolumn = default;
                    var bitmapCropGetter = cursor.GetGetter<Bitmap>(cropBitmapColumn);
                    Bitmap bitmapCropcolumn = default;
                    while (cursor.MoveNext())
                    {
                        pixelsGetter(ref pixelcolumn);
                        bitmapGetter(ref bitmapcolumn);
                        bitmapCropGetter(ref bitmapCropcolumn);
                    }
                }

            }

        }
    }
}
