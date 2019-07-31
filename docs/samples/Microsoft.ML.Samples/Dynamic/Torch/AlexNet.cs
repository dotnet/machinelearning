using System;
using System.IO;
using System.Linq;
using System.Net;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic.Torch
{
    public static class AlexNet
    {
        /// <summary>
        /// Example use of the TensorFlow image model in a ML.NET pipeline.
        /// </summary>
        public static void Example()
        {
            // Download the TorchScript AlexNet model obtained by tracing the 
            // torchvision pretrained alexnet model for imagenet.

            // Here is the code that produced the model:

            //      import torch
            //      import torchvision
            //      traced_net = torch.jit.trace(
            //                      torchvision.models.alexnet(pretrained = True),
            //                      torch.rand(1, 3, 224, 224))
            //      torch.jit.save(traced_net, 'alexnet.pt')

            string modelLocation = "alexnet.pt";
            if (!File.Exists(modelLocation))
            {
                modelLocation = Download(
                    @"https://aka.ms/mlnet-resources/torch/alexnet.pt",
                    @"alexnet.pt");
            }

            var mlContext = new MLContext();
            var data = GetTensorData();
            var idv = mlContext.Data.LoadFromEnumerable(data);

            // Create a ML pipeline.
            var pipeline = mlContext.Model.LoadTorchModel(modelLocation)
                .ScoreTorchModel(nameof(OutputScores.output),
                                 new long[] { 1, 3, 224, 224 },
                                 nameof(TensorData.input));

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
                foreach (var classScore in prediction.output.Take(3))
                {
                    Console.WriteLine(
                        $"Class #{numClasses++} score = {classScore}");
                }
                Console.WriteLine(new string('-', 10));
            }

            // Results look like below...
            // Class #0 score = 0.8008841
            // Class #1 score = 1.181702
            // Class #2 score = -0.02895377
            // ----------
            // Class #0 score = 0.3972791
            // Class #1 score = 1.154241
            // Class #2 score = 0.202249
            // ----------
        }
        private const int imageHeight = 224; 
        private const int imageWidth = 224;
        private const int numChannels = 3;
        private const int batchSize = 1;
        private const int inputSize = imageHeight * imageWidth * numChannels;

        /// <summary>
        /// A class to hold sample tensor data. 
        /// Member name should match the inputs that the model expects (in this
        /// case, input).
        /// </summary>
        public class TensorData
        {
            [VectorType(batchSize, numChannels, imageHeight, imageWidth)]
            public float[] input { get; set; }
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
            return new TensorData[] { new TensorData() { input = image1 },
                new TensorData() { input = image2 } };
        }

        /// <summary>
        /// Class to contain the output values from the transformation.
        /// </summary>
        class OutputScores
        {
            [VectorType(1000)]
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
    }
}
