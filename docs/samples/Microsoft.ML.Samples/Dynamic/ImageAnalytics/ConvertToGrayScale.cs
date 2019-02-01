using System.IO;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public class ConvertToGrayscaleExample
    {
        public static void ConvertToGrayscale()
        {
            var mlContext = new MLContext();

            // Downloading a few images, and an images.tsv file, that contains a list of the files, from the dotnet/machinelearning/test/data/images/
            // if you inspect the fileSystem, after running this line, the 
            var imagesDataFile = SamplesUtils.DatasetUtils.DownloadImages();

            // Preview of the content of the images.tsv file
            //
            // imagePath    imageType
            // tomato.bmp   tomato
            // banana.jpg   banana
            // hotdog.jpg   hotdog
            // tomato.jpg   tomato

            var data = mlContext.Data.CreateTextLoader(new TextLoader.Arguments()
            {
                Columns = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                 }
            }).Read(imagesDataFile);

            var imagesFolder = Path.GetDirectoryName(imagesDataFile);
            // Image loading pipeline. 
            var pipeline = mlContext.Transforms.LoadImages(imagesFolder, ("ImageObject", "ImagePath"))
                           .Append(mlContext.Transforms.ConvertToGrayscale(("GrayScale", "ImageObject")));

            var transformedData = pipeline.Fit(data).Transform(data);

            // The transformedData IDataView contains the loaded images now
            //Preview of the transformedData
            var transformedDataPreview = transformedData.Preview();

            // Preview of the content of the images.tsv file
            // The actual images, in the ImageReal column are of type System.Drawing.Bitmap.
            //
            // ImagePath    Name        ImageObject                   GrayScale
            // tomato.bmp   tomato      {System.Drawing.Bitmap}     {System.Drawing.Bitmap}
            // banana.jpg   banana      {System.Drawing.Bitmap}     {System.Drawing.Bitmap}
            // hotdog.jpg   hotdog      {System.Drawing.Bitmap}     {System.Drawing.Bitmap}
            // tomato.jpg   tomato      {System.Drawing.Bitmap}     {System.Drawing.Bitmap}

        }
    }
}
