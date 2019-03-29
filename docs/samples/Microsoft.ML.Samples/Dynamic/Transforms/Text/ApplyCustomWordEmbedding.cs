using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class ApplyCustomWordEmbedding
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Create an empty data sample list. The 'ApplyWordEmbedding' does not require training data as
            // the estimator ('WordEmbeddingEstimator') created by 'ApplyWordEmbedding' API is not a trainable estimator.
            // The empty list is only needed to pass input schema to the pipeline.
            var emptySamples = new List<TextData>();

            // Convert sample list to an empty IDataView.
            var emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples);

            var pathToCustomModel = @".\custommodel.txt";
            using (StreamWriter file = new StreamWriter(pathToCustomModel, false))
            {

                file.WriteLine("This is custom file for 4 words with 3 dimensional word embedding vector. This first line in this file does not confirm to the '<word> <float> <float> <float>' pattern, and is therefore ignored");
                file.WriteLine("greate" + " " + string.Join(" ", 1.0f, 2.0f, 3.0f));
                file.WriteLine("product" + " " + string.Join(" ", -1.0f, -2.0f, -3.0f));
                file.WriteLine("like" + " " + string.Join(" ", -1f, 100.0f, -100f));
                file.WriteLine("buy" + " " + string.Join(" ", 0f, 0f, 20f));
            }

            // A pipeline for converting text into a 9-dimension word embedding vector using the custom word embedding model.
            // The 'ApplyWordEmbedding' computes the minimum, average and maximum values for each token's embedding vector.
            // Tokens in 'custommodel.txt' model are represented as 3-dimension vector.
            // Therefore, the output is of 9-dimension [min, avg, max].
            //
            // The 'ApplyWordEmbedding' API requires vector of text as input.
            // The pipeline first normalizes and tokenizes text then applies word embedding transformation.
            var textPipeline = mlContext.Transforms.Text.NormalizeText("Text")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text"))
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Features", pathToCustomModel, "Tokens"));

            // Fit to data.
            var textTransformer = textPipeline.Fit(emptyDataView);

            // Create the prediction engine to get the embedding vector from the input text/string.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(textTransformer);

            // Call the prediction API to convert the text into embedding vector.
            var data = new TextData() { Text = "This is a greate product. I would like to buy it again."  };
            var prediction = predictionEngine.Predict(data);

            // Print the length of the embedding vector.
            Console.WriteLine($"Number of Features: {prediction.Features.Length}");

            // Print the embedding vector.
            Console.Write("Features: ");
            foreach (var f in prediction.Features)
                Console.Write($"{f:F4} ");

            //  Expected output:
            //   Number of Features: 9
            //   Features: -1.0000 0.0000 -100.0000 0.0000 34.0000 -25.6667 1.0000 100.0000 20.0000
        }

        public class TextData
        {
            public string Text { get; set; }
        }

        public class TransformedTextData : TextData
        {
            public float[] Features { get; set; }
        }
    }
}
