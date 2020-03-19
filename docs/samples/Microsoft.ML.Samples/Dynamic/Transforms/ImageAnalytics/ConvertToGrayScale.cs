using System;
using System.Drawing;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class ConvertToGrayscale
    {
        // Sample that loads images from the file system, and converts them to
        // grayscale. 
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
            var pipeline = mlContext.Transforms.LoadImages("ImageObject",
                imagesFolder, "ImagePath")
                .Append(mlContext.Transforms.ConvertToGrayscale("Grayscale",
                "ImageObject"));

            var transformedData = pipeline.Fit(data).Transform(data);

            PrintColumns(transformedData);
            // ImagePath    Name         ImageObject              Grayscale
            // tomato.bmp   tomato       {Width=800, Height=534}  {Width=800, Height=534}
            // banana.jpg   banana       {Width=800, Height=288}  {Width=800, Height=288}
            // hotdog.jpg   hotdog       {Width=800, Height=391}  {Width=800, Height=391}
            // tomato.jpg   tomato       {Width=800, Height=534}  {Width=800, Height=534}
        }

        private static void PrintColumns(IDataView transformedData)
        {
            Console.WriteLine("{0, -25} {1, -25} {2, -25} {3, -25}", "ImagePath",
                "Name", "ImageObject", "Grayscale");

            using (var cursor = transformedData.GetRowCursor(transformedData
                .Schema))
            {
                // Note that it is best to get the getters and values *before*
                // iteration, so as to faciliate buffer sharing (if applicable), and
                // column -type validation once, rather than many times.
                ReadOnlyMemory<char> imagePath = default;
                ReadOnlyMemory<char> name = default;
                Bitmap imageObject = null;
                Bitmap grayscaleImageObject = null;

                var imagePathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["ImagePath"]);

                var nameGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["Name"]);

                var imageObjectGetter = cursor.GetGetter<Bitmap>(cursor.Schema[
                    "ImageObject"]);

                var grayscaleGetter = cursor.GetGetter<Bitmap>(cursor.Schema[
                    "Grayscale"]);

                while (cursor.MoveNext())
                {
                    imagePathGetter(ref imagePath);
                    nameGetter(ref name);
                    imageObjectGetter(ref imageObject);
                    grayscaleGetter(ref grayscaleImageObject);

                    Console.WriteLine("{0, -25} {1, -25} {2, -25} {3, -25}",
                        imagePath, name, imageObject.PhysicalDimension,
                        grayscaleImageObject.PhysicalDimension);
                }

                // Dispose the image.
                imageObject.Dispose();
                grayscaleImageObject.Dispose();
            }
        }
    }
}
