using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

namespace Samples.Dynamic
{
    public static class RemoveDefaultStopWords
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create an empty list as the dataset. The 'RemoveDefaultStopWords'
            // does not require training data as the estimator 
            // ('StopWordsRemovingEstimator') created by 'RemoveDefaultStopWords'
            // API is not a trainable estimator. The empty list is only needed to
            // pass input schema to the pipeline.
            var emptySamples = new List<TextData>();

            // Convert sample list to an empty IDataView.
            var emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples);

            // A pipeline for removing stop words from input text/string.
            // The pipeline first tokenizes text into words then removes stop words.
            // The 'RemoveDefaultStopWords' API ignores casing of the text/string
            // e.g. 'tHe' and 'the' are considered the same stop words.
            var textPipeline = mlContext.Transforms.Text.TokenizeIntoWords("Words",
                "Text")
                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords(
                "WordsWithoutStopWords", "Words", language:
                StopWordsRemovingEstimator.Language.English));

            // Fit to data.
            var textTransformer = textPipeline.Fit(emptyDataView);

            // Create the prediction engine to remove the stop words from the input
            // text /string.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData,
                TransformedTextData>(textTransformer);

            // Call the prediction API to remove stop words.
            var data = new TextData()
            {
                Text = "ML.NET's RemoveDefaultStopWords " +
                "API removes stop words from tHe text/string. It requires the " +
                "text/string to be tokenized beforehand."
            };

            var prediction = predictionEngine.Predict(data);

            // Print the length of the word vector after the stop words removed.
            Console.WriteLine("Number of words: " + prediction.WordsWithoutStopWords
                .Length);

            // Print the word vector without stop words.
            Console.WriteLine("\nWords without stop words: " + string.Join(",",
                prediction.WordsWithoutStopWords));

            //  Expected output:
            //   Number of words: 11
            //   Words without stop words: ML.NET's,RemoveDefaultStopWords,API,removes,stop,words,text/string.,requires,text/string,tokenized,beforehand.
        }

        private class TextData
        {
            public string Text { get; set; }
        }

        private class TransformedTextData : TextData
        {
            public string[] WordsWithoutStopWords { get; set; }
        }
    }
}
