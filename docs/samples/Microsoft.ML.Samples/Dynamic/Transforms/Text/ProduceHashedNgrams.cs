using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class ProduceHashedNgrams
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "This is an example to compute n-grams " +
                "using hashing." },

                new TextData(){ Text = "N-gram is a sequence of 'N' consecutive" +
                " words/tokens." },

                new TextData(){ Text = "ML.NET's ProduceHashedNgrams API " +
                "produces count of n-grams and hashes it as an index into a " +
                "vector of given bit length." },

                new TextData(){ Text = "The hashing reduces the size of the " +
                "output feature vector" },

                new TextData(){ Text = "which is useful in case when number of " +
                "n-grams is very large." },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for converting text into numeric hashed n-gram features.
            // The following call to 'ProduceHashedNgrams' requires the tokenized
            // text /string as input. This is achieved by calling 
            // 'TokenizeIntoWords' first followed by 'ProduceHashedNgrams'.
            // Please note that the length of the output feature vector depends on
            // the 'numberOfBits' settings.
            var textPipeline = mlContext.Transforms.Text.TokenizeIntoWords("Tokens",
                "Text")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Tokens"))
                .Append(mlContext.Transforms.Text.ProduceHashedNgrams(
                    "NgramFeatures", "Tokens",
                    numberOfBits: 5,
                    ngramLength: 3,
                    useAllLengths: false,
                    maximumNumberOfInverts: 1));

            // Fit to data.
            var textTransformer = textPipeline.Fit(dataview);
            var transformedDataView = textTransformer.Transform(dataview);

            // Create the prediction engine to get the features extracted from the
            // text.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData,
                TransformedTextData>(textTransformer);

            // Convert the text into numeric features.
            var prediction = predictionEngine.Predict(samples[0]);

            // Print the length of the feature vector.
            Console.WriteLine("Number of Features: " + prediction.NgramFeatures
                .Length);

            // Preview of the produced n-grams.
            // Get the slot names from the column's metadata.
            // The slot names for a vector column corresponds to the names
            // associated with each position in the vector.
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            transformedDataView.Schema["NgramFeatures"].GetSlotNames(ref slotNames);
            var NgramFeaturesColumn = transformedDataView.GetColumn<VBuffer<float>>(
                transformedDataView.Schema["NgramFeatures"]);

            var slots = slotNames.GetValues();
            Console.Write("N-grams: ");
            foreach (var featureRow in NgramFeaturesColumn)
            {
                foreach (var item in featureRow.Items())
                    Console.Write($"{slots[item.Key]}  ");
                Console.WriteLine();
            }

            // Print the first 10 feature values.
            Console.Write("Features: ");
            for (int i = 0; i < 10; i++)
                Console.Write($"{prediction.NgramFeatures[i]:F4}  ");

            //  Expected output:
            //   Number of Features:  32
            //   N-grams:   This|is|an  example|to|compute  compute|n-grams|using  n-grams|using|hashing.  an|example|to  is|an|example  a|sequence|of  of|'N'|consecutive  is|a|sequence  N-gram|is|a  ...
            //   Features:    0.0000          0.0000               2.0000               0.0000               0.0000        1.0000          0.0000        0.0000              1.0000          0.0000  ...
        }

        private class TextData
        {
            public string Text { get; set; }
        }

        private class TransformedTextData : TextData
        {
            public float[] NgramFeatures { get; set; }
        }
    }
}
