using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class LatentDirichletAllocation
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "ML.NET's LatentDirichletAllocation API computes topic model." },
                new TextData(){ Text = "ML.NET's LatentDirichletAllocation API is the best for topic model." },
                new TextData(){ Text = "I like to eat broccoli and banana." },
                new TextData(){ Text = "I eat a banana in the breakfast." },
                new TextData(){ Text = "This car is expensive compared to last week's price." },
                new TextData(){ Text = "This car was $X last week." },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for featurizing the text/string using LatentDirichletAllocation API.
            // To be more accurate in computing the LDA features, the pipeline first normalizes text and removes stop words
            // before passing tokens to LatentDirichletAllocation.
            var pipeline = mlContext.Transforms.Text.NormalizeText("normText", "Text")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "normText"))
                .Append(mlContext.Transforms.Text.RemoveStopWords("Tokens"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Tokens"))
                .Append(mlContext.Transforms.Text.ProduceNgrams("Tokens"))
                .Append(mlContext.Transforms.Text.LatentDirichletAllocation("Features", "Tokens", numberOfTopics: 3));

            // Fit to data.
            var transformer = pipeline.Fit(dataview);

            // Create the prediction engine to get the LDA features extracted from the text.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(transformer);

            // Convert the sample text into LDA features and print it.
            PrintPredictions(predictionEngine.Predict(samples[0]));
            PrintPredictions(predictionEngine.Predict(samples[1]));

            // Features obtained post-transformation.
            // For LatentDirichletAllocation, we had specified numTopic:3. Hence each prediction has been featurized as a vector of floats with length 3.

            //  Topic1  Topic2  Topic3
            //  0.6364  0.3636  0.0000
            //  0.4118  0.1765  0.4118
        }

        private static void PrintPredictions(TransformedTextData prediction)
        {
            for (int i = 0; i < prediction.Features.Length; i++)
                Console.Write($"{prediction.Features[i]:F4}  ");
            Console.WriteLine();
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
