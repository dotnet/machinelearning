using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
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
            var imagesDataFile = SamplesUtils.DatasetUtils.DownloadImages();

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
                          .Append(mlContext.Transforms.ResizeImages("ImageObjectResized", inputColumnName:"ImageObject", imageWidth: 100, imageHeight: 100 ))
                          .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObjectResized"));


            var transformedData = pipeline.Fit(data).Transform(data);

            // The transformedData IDataView contains the loaded images now

            // Preview 1 row of the transformedData. 
            var transformedDataPreview = transformedData.Preview(1);
            foreach (var kvPair in transformedDataPreview.RowView[0].Values)
            {
                Console.WriteLine("{0} : {1}", kvPair.Key, kvPair.Value);
            }

            // ImagePath: tomato.bmp
            // Name : tomato
            // ImageObject : System.Drawing.Bitmap
            // ImageObjectResized : System.Drawing.Bitmap
            // Pixels : Dense vector of size 30000

            Console.WriteLine("--------------------------------------------------");

            // Using schema comprehension to display raw pixels for each row.
            // Display original columns 'ImagePath' and 'Name', and extracted pixels in column 'Pixels'.
            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, true);
            foreach (var item in convertedData)
            {
                var pixels = item.Pixels.Take(5);
                Console.WriteLine("{0} {1}  pixels:{2}...", item.ImagePath, item.Name, string.Join(",", pixels));
            }

            // tomato.bmp tomato pixels:255,255,255,255,255...
            // banana.jpg banana pixels:255,255,255,255,255...
            // hotdog.jpg hotdog pixels:255,255,255,255,255...
            // tomato.jpg tomato pixels:255,255,255,255,255...
        }

        private sealed class TransformedData
        {
            public string ImagePath { get; set; }
            public string Name { get; set; }
            public float[] Pixels { get; set; }
        }
    }
}
