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
    public static class TransferLearning
    {
        /// <summary>
        /// Example use of the TensorFlow image model in a ML.NET pipeline.
        /// </summary>
        public static void Example()
        {
            var sw = new Stopwatch();

            var mlContext = new MLContext();
            var data = GetTensorData();
            var idv = mlContext.Data.LoadFromEnumerable(data);

            // Create a ML pipeline.
            var pipeline = 
                mlContext.Transforms.Conversion.MapValueToKey(
                    nameof(TensorData.Label))
                .Append(mlContext.Model.ImageClassification(
                    nameof(TensorData.input),
                    nameof(TensorData.Label), batchSize: 2,
                        addBatchDimensionInput: true));

            // Run the pipeline and get the transformed values.
            var estimator = pipeline.Fit(idv);
            var transformedValues = estimator.Transform(idv);

            // Retrieve model scores.
            var outScores = mlContext.Data.CreateEnumerable<OutputScores>(
                transformedValues, reuseRowObject: false);

            // Display scores. (for the sake of brevity we display scores of the
            // first 3 classes)
            foreach (var prediction in outScores)
            {
                int numClasses = 0;
                foreach (var classScore in prediction.Scores.Take(2))
                {
                    Console.WriteLine(
                        $"Class #{numClasses++} score = {classScore}");
                }
                Console.WriteLine(
                    $"Predicted Label: {prediction.PredictedLabel}");


                Console.WriteLine(new string('-', 10));
            }

            Console.WriteLine(sw.Elapsed);
            // Class #0 score = 0.03416626
            // Class #1 score = 0.9658337
            // Predicted Label: 1
            // ----------
            // Class #0 score = 0.02959618
            // Class #1 score = 0.9704038
            // Predicted Label: 1
            // ----------
        }

        private const int imageHeight = 224; 
        private const int imageWidth = 224;
        private const int numChannels = 3;
        private const int inputSize = imageHeight * imageWidth * numChannels;

        /// <summary>
        /// A class to hold sample tensor data. 
        /// Member name should match the inputs that the model expects (in this
        /// case, input).
        /// </summary>
        public class TensorData
        {
            [VectorType(imageHeight, imageWidth, numChannels)]
            public float[] input { get; set; }

            public Int64 Label { get; set; }
        }

        /// <summary>
        /// Method to generate sample test data. Returns 2 sample rows.
        /// </summary>
        public static TensorData[] GetTensorData()
        {
            // This can be any numerical data. Assume image pixel values.
            var image1 = Enumerable.Range(0, inputSize).Select(
                x => (float)x / inputSize).ToArray();
            
            var image2 = Enumerable.Range(0, inputSize).Select(
                x => (float)(x + 10000) / inputSize).ToArray();
            return new TensorData[] { new TensorData() { input = image1, Label = 0 },
                new TensorData() { input = image2, Label = 1 } };
        }

        /// <summary>
        /// Class to contain the output values from the transformation.
        /// </summary>
        class OutputScores
        {
            public float[] Scores { get; set; }
            public Int64 PredictedLabel { get; set; }
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
        /// Taken from 
        /// https://github.com/icsharpcode/SharpZipLib/wiki/GZip-and-Tar-Samples.
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
