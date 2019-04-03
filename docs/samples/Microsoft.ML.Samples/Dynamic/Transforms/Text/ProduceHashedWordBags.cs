using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class ProduceHashedWordBags
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "This is an example to compute bag-of-word features using hashing." },
                new TextData(){ Text = "ML.NET's ProduceHashedWordBags API produces count of n-grams and hashes it as an index into a vector of given bit length." },
                new TextData(){ Text = "It does so by first tokenizing text/string into words/tokens then " },
                new TextData(){ Text = "computing n-grams and hash them to the index given by hash value." },
                new TextData(){ Text = "The hashing reduces the size of the output feature vector" },
                new TextData(){ Text = "which is useful in case when number of n-grams is very large." },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for converting text into numeric bag-of-word features using hashing.
            // The following call to 'ProduceHashedWordBags' implicitly tokenizes the text/string into words/tokens.
            // Please note that the length of the output feature vector depends on the 'numberOfBits' settings.
            var textPipeline = mlContext.Transforms.Text.ProduceHashedWordBags("BagOfWordFeatures", "Text",
                numberOfBits: 5, ngramLength: 3, useAllLengths: false);

            // Fit to data.
            var textTransformer = textPipeline.Fit(dataview);

            // Create the prediction engine to get the bag-of-word features extracted from the text.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(textTransformer);

            // Convert the text into numeric features.
            var prediction = predictionEngine.Predict(samples[0]);

            // Print the length of the feature vector.
            Console.WriteLine($"Number of Features: {prediction.BagOfWordFeatures.Length}");

            // Print the first 10 feature values.
            Console.Write("Features: ");
            for (int i = 0; i < 10; i++)
                Console.Write($"{prediction.BagOfWordFeatures[i]:F4}  ");

            //  Expected output:
            //   Number of Features: 32
            //   Features:  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  2.0000  0.0000  0.0000
        }

        public class TextData
        {
            public string Text { get; set; }
        }

        public class TransformedTextData : TextData
        {
            public float[] BagOfWordFeatures { get; set; }
        }
    }
}
