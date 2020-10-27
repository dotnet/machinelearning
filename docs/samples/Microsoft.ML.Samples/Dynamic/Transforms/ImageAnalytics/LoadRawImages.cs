using System;
using System.Drawing;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class LoadImages
    {
        // Loads the images of the imagesFolder into an IDataView. 
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Downloading a few images, and an images.tsv file, which contains a
            // list of the files from the dotnet/machinelearning/test/data/images/.
            // If you inspect the fileSystem, after running this line, an "images"
            // folder will be created, containing 4 images, and a .tsv file
            // enumerating the images. 
            var imagesDataFile = Microsoft.ML.SamplesUtils.DatasetUtils
                .GetSampleImages();

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
            var pipeline = mlContext.Transforms.LoadRawImageBytes("Image",
                imagesFolder, "ImagePath");

            var transformedData = pipeline.Fit(data).Transform(data);

            PrintColumns(transformedData);
            // ImagePath    Name       
            // tomato.bmp   tomato
            // banana.jpg   banana
            // hotdog.jpg   hotdog
            // tomato.jpg   tomato
        }

        private static void PrintColumns(IDataView transformedData)
        {
            // The transformedData IDataView contains the loaded images now.
            Console.WriteLine("{0, -25} {1, -25} {2, -25}", "ImagePath", "Name",
                "Image");

            using (var cursor = transformedData.GetRowCursor(transformedData
                .Schema))
            {
                // Note that it is best to get the getters and values *before*
                // iteration, so as to facilitate buffer sharing (if applicable),
                // and column-type validation once, rather than many times.
                ReadOnlyMemory<char> imagePath = default;
                ReadOnlyMemory<char> name = default;

                var imagePathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["ImagePath"]);

                var nameGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["Name"]);

                while (cursor.MoveNext())
                {

                    imagePathGetter(ref imagePath);
                    nameGetter(ref name);

                    Console.WriteLine("{0, -25} {1, -25}", imagePath, name);
                }
            }
        }
    }
}