using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class TokenizeIntoWords
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create an empty list as the dataset. The 'TokenizeIntoWords' does
            // not require training data as the estimator
            // ('WordTokenizingEstimator') created by 'TokenizeIntoWords' API is not
            // a trainable estimator. The empty list is only needed to pass input
            // schema to the pipeline.
            var emptySamples = new List<TextData>();

            // Convert sample list to an empty IDataView.
            var emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples);

            // A pipeline for converting text into vector of words.
            // The following call to 'TokenizeIntoWords' tokenizes text/string into
            // words using space as a separator. Space is also a default value for
            // the 'separators' argument if it is not specified.
            var textPipeline = mlContext.Transforms.Text.TokenizeIntoWords("Words",
                "Text", separators: new[] { ' ' });

            // Fit to data.
            var textTransformer = textPipeline.Fit(emptyDataView);

            // Create the prediction engine to get the word vector from the input
            // text /string.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData,
                TransformedTextData>(textTransformer);

            // Call the prediction API to convert the text into words.
            var data = new TextData()
            {
                Text = "ML.NET's TokenizeIntoWords API " +
                "splits text/string into words using the list of characters " +
                "provided as separators."
            };

            var prediction = predictionEngine.Predict(data);

            // Print the length of the word vector.
            Console.WriteLine($"Number of words: {prediction.Words.Length}");

            // Print the word vector.
            Console.WriteLine($"\nWords: {string.Join(",", prediction.Words)}");

            //  Expected output:
            //   Number of words: 15
            //   Words: ML.NET's,TokenizeIntoWords,API,splits,text/string,into,words,using,the,list,of,characters,provided,as,separators.
        }

        private class TextData
        {
            public string Text { get; set; }
        }

        private class TransformedTextData : TextData
        {
            public string[] Words { get; set; }
        }
    }
}
