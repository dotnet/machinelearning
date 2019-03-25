using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class DnnFeaturizeImage
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Downloading a few images, and an images.tsv file, which contains a list of the files from the dotnet/machinelearning/test/data/images/.
            // If you inspect the fileSystem, after running this line, an "images" folder will be created, containing 4 images, and a .tsv file
            // enumerating the images. 
            var imagesDataFile = SamplesUtils.DatasetUtils.DownloadImages();

            // Preview of the content of the images.tsv file, which lists the images to operate on
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

            // Installing the Microsoft.ML.DNNImageFeaturizer packages copies the models in the
            // `DnnImageModels` folder. 
            // Image loading pipeline. 
            var pipeline = mlContext.Transforms.LoadImages(imagesFolder, "ImageObject", "ImagePath")
                          .Append(mlContext.Transforms.ResizeImages("ImageObject", imageWidth: 224, imageHeight: 224))
                          .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
                          .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturizedImage", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), "Pixels"));

            var transformedData = pipeline.Fit(data).Transform(data);

            var FeaturizedImageColumnsPerRow = transformedData.GetColumn<float[]>("FeaturizedImage").ToArray();

            // Preview of FeaturizedImageColumnsPerRow for the first row, FeaturizedImageColumnsPerRow[0]
            //
            // 0.696136236
            // 0.2661711
            // 0.440882325
            // 0.157903448
            // 0.0339231342
            // 0
            // 0.0936501548
            // 0.159010679
            // 0.394427955

        }
    }
}
