using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class FeaturizeText
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "ML.NET's FeaturizeText API uses a " +
                    "composition of several basic transforms to convert text " +
                    "into numeric features." },

                new TextData(){ Text = "This API can be used as a featurizer to " +
                    "perform text classification." },

                new TextData(){ Text = "There are a number of approaches to text " +
                    "classification." },

                new TextData(){ Text = "One of the simplest and most common " +
                    "approaches is called “Bag of Words”." },

                new TextData(){ Text = "Text classification can be used for a " +
                    "wide variety of tasks" },

                new TextData(){ Text = "such as sentiment analysis, topic " +
                    "detection, intent identification etc." },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for converting text into numeric features.
            // The following call to 'FeaturizeText' instantiates 
            // 'TextFeaturizingEstimator' with default parameters.
            // The default settings for the TextFeaturizingEstimator are
            //      * StopWordsRemover: None
            //      * CaseMode: Lowercase
            //      * OutputTokensColumnName: None
            //      * KeepDiacritics: false, KeepPunctuations: true, KeepNumbers:
            //          true
            //      * WordFeatureExtractor: NgramLength = 1
            //      * CharFeatureExtractor: NgramLength = 3, UseAllLengths = false
            // The length of the output feature vector depends on these settings.
            var textPipeline = mlContext.Transforms.Text.FeaturizeText("Features",
                "Text");

            // Fit to data.
            var textTransformer = textPipeline.Fit(dataview);

            // Create the prediction engine to get the features extracted from the
            // text.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData,
                TransformedTextData>(textTransformer);

            // Convert the text into numeric features.
            var prediction = predictionEngine.Predict(samples[0]);

            // Print the length of the feature vector.
            Console.WriteLine($"Number of Features: {prediction.Features.Length}");

            // Print the first 10 feature values.
            Console.Write("Features: ");
            for (int i = 0; i < 10; i++)
                Console.Write($"{prediction.Features[i]:F4}  ");

            //  Expected output:
            //   Number of Features: 332
            //   Features: 0.0857  0.0857  0.0857  0.0857  0.0857  0.0857  0.0857  0.0857  0.0857  0.1715 ...
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
