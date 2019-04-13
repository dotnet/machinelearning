using System;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class ExtractPixels
    {
        // Sample that loads the images from the file system, resizes them (ExtractPixels requires a resizing operation), and extracts the 
        // values of the pixels as a vector. 
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Downloading a few images, and an images.tsv file, which contains a list of the files from the dotnet/machinelearning/test/data/images/.
            // If you inspect the fileSystem, after running this line, an "images" folder will be created, containing 4 images, and a .tsv file
            // enumerating the images. 
            var imagesDataFile = Microsoft.ML.SamplesUtils.DatasetUtils.DownloadImages();

            // Preview of the content of the images.tsv file
            //
            // imagePath    imageType
            // tomato.bmp   tomato
            // banana.jpg   banana
            // hotdog.jpg   hotdog
            // tomato.jpg   tomato

            var data = mlContext.Data.CreateTextLoader(new TextLoader.Options()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                }
            }).Load(imagesDataFile);

            var imagesFolder = Path.GetDirectoryName(imagesDataFile);
            // Image loading pipeline. 
            var pipeline = mlContext.Transforms.LoadImages("ImageObject", imagesFolder, "ImagePath")
                          .Append(mlContext.Transforms.ResizeImages("ImageObjectResized", inputColumnName: "ImageObject", imageWidth: 100, imageHeight: 100))
                          .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObjectResized"));

            var transformedData = pipeline.Fit(data).Transform(data);

            // Preview the transformedData. 
            PrintColumns(transformedData);

            // ImagePath    Name         ImageObject            ImageObjectResized      Pixels
            // tomato.bmp   tomato       System.Drawing.Bitmap  System.Drawing.Bitmap   255,255,255,255,255...
            // banana.jpg   banana       System.Drawing.Bitmap  System.Drawing.Bitmap   255,255,255,255,255...
            // hotdog.jpg   hotdog       System.Drawing.Bitmap  System.Drawing.Bitmap   255,255,255,255,255...
            // tomato.jpg   tomato       System.Drawing.Bitmap  System.Drawing.Bitmap   255,255,255,255,255...
        }
        private static void PrintColumns(IDataView transformedData)
        {
            var imagePathColumn = transformedData.GetColumn<string>("ImagePath").GetEnumerator();
            var namePathColumn = transformedData.GetColumn<string>("Name").GetEnumerator();
            var imageObjectColumn = transformedData.GetColumn<Bitmap>("ImageObject").GetEnumerator();
            var imageObjectResizedColumn = transformedData.GetColumn<Bitmap>("ImageObjectResized").GetEnumerator();
            var pixelsColumn = transformedData.GetColumn<VBuffer<float>>("Pixels").GetEnumerator();

            Console.WriteLine("{0, -25} {1, -25} {2, -25} {3, -25} {4, -25}", "ImagePath", "Name", "ImageObject", "ImageObjectResized", "Pixels");
            while (imagePathColumn.MoveNext() && namePathColumn.MoveNext() && imageObjectColumn.MoveNext() && imageObjectResizedColumn.MoveNext() && pixelsColumn.MoveNext())
            {
                Console.WriteLine("{0, -25} {1, -25} {2, -25} {3, -25} {4, -25}",
                    imagePathColumn.Current,
                    namePathColumn.Current,
                    imageObjectColumn.Current,
                    imageObjectResizedColumn.Current,
                    string.Join(",", pixelsColumn.Current.DenseValues().Take(5)) + "...");

                //Dispose bitmap image.
                imageObjectResizedColumn.Current.Dispose();
                imageObjectColumn.Current.Dispose();
            }
        }
    }
}
