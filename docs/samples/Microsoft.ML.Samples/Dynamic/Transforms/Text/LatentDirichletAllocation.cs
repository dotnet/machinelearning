using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class LatentDirichletAllocation
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "ML.NET's LatentDirichletAllocation API " +
                "computes topic models." },

                new TextData(){ Text = "ML.NET's LatentDirichletAllocation API " +
                "is the best for topic models." },

                new TextData(){ Text = "I like to eat broccoli and bananas." },
                new TextData(){ Text = "I eat bananas for breakfast." },
                new TextData(){ Text = "This car is expensive compared to last " +
                "week's price." },

                new TextData(){ Text = "This car was $X last week." },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for featurizing the text/string using 
            // LatentDirichletAllocation API. o be more accurate in computing the
            // LDA features, the pipeline first normalizes text and removes stop
            // words before passing tokens (the individual words, lower cased, with
            // common words removed) to LatentDirichletAllocation.
            var pipeline = mlContext.Transforms.Text.NormalizeText("NormalizedText",
                "Text")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens",
                    "NormalizedText"))
                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("Tokens"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Tokens"))
                .Append(mlContext.Transforms.Text.ProduceNgrams("Tokens"))
                .Append(mlContext.Transforms.Text.LatentDirichletAllocation(
                    "Features", "Tokens", numberOfTopics: 3));

            // Fit to data.
            var transformer = pipeline.Fit(dataview);

            // Create the prediction engine to get the LDA features extracted from
            // the text.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData,
                TransformedTextData>(transformer);

            // Convert the sample text into LDA features and print it.
            PrintLdaFeatures(predictionEngine.Predict(samples[0]));
            PrintLdaFeatures(predictionEngine.Predict(samples[1]));

            // Features obtained post-transformation.
            // For LatentDirichletAllocation, we had specified numTopic:3. Hence
            // each prediction has been featurized as a vector of floats with length
            // 3.

            //  Topic1  Topic2  Topic3
            //  0.6364  0.2727  0.0909
            //  0.5455  0.1818  0.2727
        }

        private static void PrintLdaFeatures(TransformedTextData prediction)
        {
            for (int i = 0; i < prediction.Features.Length; i++)
                Console.Write($"{prediction.Features[i]:F4}  ");
            Console.WriteLine();
        }

        private class TextData
        {
            public string Text { get; set; }
        }

        private class TransformedTextData : TextData
        {
            public float[] Features { get; set; }
        }
    }
}
