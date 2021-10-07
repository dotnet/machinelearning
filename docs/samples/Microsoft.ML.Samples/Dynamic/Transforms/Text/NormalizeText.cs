using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

namespace Samples.Dynamic
{
    public static class NormalizeText
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create an empty list as the dataset. The 'NormalizeText' API does not
            // require training data as the estimator ('TextNormalizingEstimator')
            // created by 'NormalizeText' API is not a trainable estimator. The
            // empty list is only needed to pass input schema to the pipeline.
            var emptySamples = new List<TextData>();

            // Convert sample list to an empty IDataView.
            var emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples);

            // A pipeline for normalizing text.
            var normTextPipeline = mlContext.Transforms.Text.NormalizeText(
                "NormalizedText", "Text", TextNormalizingEstimator.CaseMode.Lower,
                keepDiacritics: false,
                keepPunctuations: false,
                keepNumbers: false);

            // Fit to data.
            var normTextTransformer = normTextPipeline.Fit(emptyDataView);

            // Create the prediction engine to get the normalized text from the
            // input text/string.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData,
                TransformedTextData>(normTextTransformer);

            // Call the prediction API.
            var data = new TextData()
            {
                Text = "ML.NET's NormalizeText API " +
                "changes the case of the TEXT and removes/keeps diâcrîtîcs, " +
                "punctuations, and/or numbers (123)."
            };

            var prediction = predictionEngine.Predict(data);

            // Print the normalized text.
            Console.WriteLine($"Normalized Text: {prediction.NormalizedText}");

            //  Expected output:
            //   Normalized Text: mlnets normalizetext api changes the case of the text and removeskeeps diacritics punctuations andor numbers
        }

        private class TextData
        {
            public string Text { get; set; }
        }

        private class TransformedTextData : TextData
        {
            public string NormalizedText { get; set; }
        }
    }
}
