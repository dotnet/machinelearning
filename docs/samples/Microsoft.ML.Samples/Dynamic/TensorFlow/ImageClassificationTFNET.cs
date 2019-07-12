using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class ImageClassificationsTFNET
    {
        /// <summary>
        /// Example use of the TensorFlow image model in a ML.NET pipeline.
        /// </summary>
        public static void Example()
        {
            // Download the ResNet 101 model from the location below.
            // https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz

            string modelLocation = "resnet_v2_101_299_frozen.pb";
            if (!File.Exists(modelLocation))
            {
                modelLocation = Download(@"https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz", @"resnet_v2_101_299_frozen.tgz");
                Unzip(Path.Join(Directory.GetCurrentDirectory(), modelLocation), Directory.GetCurrentDirectory());
                modelLocation = "resnet_v2_101_299_frozen.pb";
            }
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            var mlContext = new MLContext();
            var data = GetTensorData(mlContext);
            var idv = mlContext.Data.LoadFromEnumerable(data);
            //modelLocation = @"C:\Users\t-prdug\source\repos\machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Samples\netcoreapp2.1\retrain_images\_retrain_checkpoint.meta";
            modelLocation = "_retrain_checkpoint.meta";
            //modelLocation = @"C:\Users\t-prdug\source\repos\TensorFlow.NET\graph\WorkingInceptionV3.meta";
            // Create a ML pipeline.
            var pipeline = mlContext.Model.LoadTensorFlowModelNet(modelLocation).ScoreTensorFlowModel(
                new[] { nameof(OutputScores.output) },
                new[] { nameof(TensorData.input) }, modelLocation, addBatchDimensionInput: true);

            //// Run the pipeline and get the transformed values.
            var estimator = pipeline.Fit(idv);
            var transformedValues = estimator.Transform(idv);

            //// Retrieve model scores.
            var outScores = mlContext.Data.CreateEnumerable<OutputScores>(transformedValues, reuseRowObject: false);

            //// Display scores. (for the sake of brevity we display scores of the first 3 classes)
            foreach (var prediction in outScores)
            {
                int numClasses = 0;
                foreach (var classScore in prediction.output.Take(3))
                {
                    Console.WriteLine($"Class #{numClasses++} score = {classScore}");
                }
                Console.WriteLine(new string('-', 10));
            }
            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;

            // Format and display the TimeSpan value.
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            Console.WriteLine("TF.NET RunTime " + elapsedTime);

            // Results look like below...
            //Class #0 score = -0.8092947
            //Class #1 score = -0.3310375
            //Class #2 score = 0.1119193
            //----------
            //Class #0 score = -0.7807726
            //Class #1 score = -0.2158062
            //Class #2 score = 0.1153686
            //----------
        }

        private const int imageHeight = 224; 
        private const int imageWidth = 224;
        private const int numChannels = 3;
        private const int inputSize = imageHeight * imageWidth * numChannels;

        /// <summary>
        /// A class to hold sample tensor data. 
        /// Member name should match the inputs that the model expects (in this case, input).
        /// </summary>
        public class TensorData
        {
            [VectorType(imageHeight, imageWidth, numChannels)]
            public float[] input { get; set; }
        }


        /// <summary>
        /// Method to generate sample test data. Returns 2 sample rows.
        /// </summary>
        public static TensorData[] GetTensorData(MLContext mlContext)
        {
            

            var imagesDataFile = Microsoft.ML.SamplesUtils.DatasetUtils
               .DownloadImages();

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
                .Append(mlContext.Transforms.ResizeImages("ImageObjectResized",
                    inputColumnName: "ImageObject", imageWidth: 224, imageHeight:
                    224))
                .Append(mlContext.Transforms.ExtractPixels("Pixels",
                    "ImageObjectResized"));

            var transformedData = pipeline.Fit(data).Transform(data);
            // This can be any numerical data. Assume image pixel values.
            var image1 = Enumerable.Range(0, inputSize).Select(x => (float)x / inputSize).ToArray();
            var image2 = Enumerable.Range(0, inputSize).Select(x => (float)(x + 10000) / inputSize).ToArray();
            using (var cursor = transformedData.GetRowCursor(transformedData
                .Schema))
            {
                // Note that it is best to get the getters and values *before*
                // iteration, so as to faciliate buffer sharing (if applicable), and
                // column -type validation once, rather than many times.


                ReadOnlyMemory<char> name = default;
                VBuffer<float> pixels = default;

                var imagePathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["ImagePath"]);

                var nameGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["Name"]);


                var pixelsGetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[
                    "Pixels"]);
                TensorData[] inputSample = new TensorData[4];
                int count = 0;
                while (cursor.MoveNext())
                {

                    
                    nameGetter(ref name);
                    pixelsGetter(ref pixels);


                    inputSample[count] = new TensorData() { input = pixels.DenseValues().ToArray()};
                    count++;
                }

                return inputSample;
            }
        }

        /// <summary>
        /// Class to contain the output values from the transformation.
        /// </summary>
        class OutputScores
        {
            public float[] output { get; set; }
        }

        private static string Download(string baseGitPath, string dataFile)
        {
            using (WebClient client = new WebClient())
            {
                client.DownloadFile(new Uri($"{baseGitPath}"), dataFile);
            }

            return dataFile;
        }

        /// <summary>
        /// Taken from https://github.com/icsharpcode/SharpZipLib/wiki/GZip-and-Tar-Samples.
        /// </summary>
        private static void Unzip(string path, string targetDir)
        {
            Stream inStream = File.OpenRead(path);
            Stream gzipStream = new GZipInputStream(inStream);

            TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream);
            tarArchive.ExtractContents(targetDir);
            tarArchive.Close();

            gzipStream.Close();
            inStream.Close();
        }
    }
}