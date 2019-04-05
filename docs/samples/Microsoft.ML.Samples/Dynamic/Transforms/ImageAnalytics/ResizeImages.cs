using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class ResizeImages
    {
        // Example on how to load the images from the file system, and resize them. 
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
                        .Append(mlContext.Transforms.ResizeImages("ImageObjectResized", inputColumnName: "ImageObject", imageWidth: 100, imageHeight: 100));

            var transformedData = pipeline.Fit(data).Transform(data);
            // The transformedData IDataView contains the resized images now.

            // Preview the transformedData. 
            var transformedDataPreview = transformedData.Preview();
            PrintPreview(transformedDataPreview);
            // ImagePath    Name         ImageObject            ImageObjectResized
            // tomato.bmp   tomato       System.Drawing.Bitmap  System.Drawing.Bitmap
            // banana.jpg   banana       System.Drawing.Bitmap  System.Drawing.Bitmap
            // hotdog.jpg   hotdog       System.Drawing.Bitmap  System.Drawing.Bitmap
            // tomato.jpg   tomato       System.Drawing.Bitmap  System.Drawing.Bitmap
        }

        private static void PrintPreview(DataDebuggerPreview data)
        {
            foreach (var colInfo in data.ColumnView)
                Console.Write("{0,-25}", colInfo.Column.Name);

            Console.WriteLine();
            foreach (var row in data.RowView)
            {
                foreach (var kvPair in row.Values)
                    Console.Write("{0,-25}", kvPair.Value);
                Console.WriteLine();
            }
        }
    }
}
