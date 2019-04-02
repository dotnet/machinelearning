using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class ProduceHashedNgrams
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "This is an example to compute Ngrams using hashing." },
                new TextData(){ Text = "Ngram is a sequence of 'N' consecutive words/tokens." },
                new TextData(){ Text = "ML.NET's ProduceHashedNgrams API produces count of Ngrams and hashes it as an index into a vector of given bit length." },
                new TextData(){ Text = "The hashing schem reduces the size of the output feature vector" },
                new TextData(){ Text = "which is useful in case when number of Ngrams is very large." },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for converting text into numeric hashed Ngram features.
            // The following call to 'ProduceHashedNgrams' requires the tokenized text/string as input.
            // This is acheived by calling 'TokenizeIntoWords' first followed by 'ProduceHashedNgrams'.
            // Please note that the length of the output feature vector depends on the 'numberOfBits' settings.
            var textPipeline = mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Tokens"))
                .Append(mlContext.Transforms.Text.ProduceHashedNgrams("NgramFeatures", "Tokens", numberOfBits: 8, ngramLength: 3, useAllLengths: false));
            
            // Fit to data.
            var textTransformer = textPipeline.Fit(dataview);

            // Create the prediction engine to get the features extracted from the text.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(textTransformer);

            // Convert the text into numeric features.
            var prediction = predictionEngine.Predict(samples[0]);

            // Print the length of the feature vector.
            Console.WriteLine($"Number of Features: {prediction.NgramFeatures.Length}");

            // Print the first 10 feature values.
            Console.Write("Features: ");
            for (int i = 0; i < 10; i++)
                Console.Write($"{prediction.NgramFeatures[i]:F4}  ");

            //  Expected output:
            //   Number of Features: 256
            //   Features:    0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
        }

        public class TextData
        {
            public string Text { get; set; }
        }

        public class TransformedTextData : TextData
        {
            public float[] NgramFeatures { get; set; }
        }
    }
}
