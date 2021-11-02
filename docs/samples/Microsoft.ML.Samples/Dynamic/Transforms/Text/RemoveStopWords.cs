using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class RemoveStopWords
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create an empty list as the dataset. The 'RemoveStopWords' does not
            // require training data as the estimator
            // ('CustomStopWordsRemovingEstimator') created by 'RemoveStopWords' API
            // is not a trainable estimator. The empty list is only needed to pass
            // input schema to the pipeline.
            var emptySamples = new List<TextData>();

            // Convert sample list to an empty IDataView.
            var emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples);

            // A pipeline for removing stop words from input text/string.
            // The pipeline first tokenizes text into words then removes stop words.
            // The 'RemoveStopWords' API ignores casing of the text/string e.g. 
            // 'tHe' and 'the' are considered the same stop words.
            var textPipeline = mlContext.Transforms.Text.TokenizeIntoWords("Words",
                "Text")
                .Append(mlContext.Transforms.Text.RemoveStopWords(
                "WordsWithoutStopWords", "Words", stopwords:
                new[] { "a", "the", "from", "by" }));

            // Fit to data.
            var textTransformer = textPipeline.Fit(emptyDataView);

            // Create the prediction engine to remove the stop words from the input
            // text /string.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData,
                TransformedTextData>(textTransformer);

            // Call the prediction API to remove stop words.
            var data = new TextData()
            {
                Text = "ML.NET's RemoveStopWords API " +
                "removes stop words from tHe text/string using a list of stop " +
                "words provided by the user."
            };

            var prediction = predictionEngine.Predict(data);

            // Print the length of the word vector after the stop words removed.
            Console.WriteLine("Number of words: " + prediction.WordsWithoutStopWords
                .Length);

            // Print the word vector without stop words.
            Console.WriteLine("\nWords without stop words: " + string.Join(",",
                prediction.WordsWithoutStopWords));

            //  Expected output:
            //   Number of words: 14
            //   Words without stop words: ML.NET's,RemoveStopWords,API,removes,stop,words,text/string,using,list,of,stop,words,provided,user.
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
