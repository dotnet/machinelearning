using System;
using System.IO;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class LoadImages
    {
        // Loads the images of the imagesFolder into an IDataView. 
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
            var pipeline = mlContext.Transforms.LoadImages("ImageReal", imagesFolder, "ImagePath");
            var transformedData = pipeline.Fit(data).Transform(data);

            // The transformedData IDataView contains the loaded images now

            // Preview 1 row of the transformedData. 
            var transformedDataPreview = transformedData.Preview(1);
            foreach (var kvPair in transformedDataPreview.RowView[0].Values)
            {
                Console.WriteLine("{0} : {1}", kvPair.Key, kvPair.Value);
            }
            
            // ImagePath : tomato.bmp
            // Name : tomato
            // ImageReal : System.Drawing.Bitmap
        }
    }
}
