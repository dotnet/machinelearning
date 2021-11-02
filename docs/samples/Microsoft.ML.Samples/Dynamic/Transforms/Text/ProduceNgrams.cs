using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace Samples.Dynamic
{
    public static class ProduceNgrams
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "This is an example to compute n-grams." },
                new TextData(){ Text = "N-gram is a sequence of 'N' consecutive " +
                    "words/tokens." },

                new TextData(){ Text = "ML.NET's ProduceNgrams API produces " +
                    "vector of n-grams." },

                new TextData(){ Text = "Each position in the vector corresponds " +
                    "to a particular n-gram." },

                new TextData(){ Text = "The value at each position corresponds " +
                    "to," },

                new TextData(){ Text = "the number of times n-gram occurred in " +
                    "the data (Tf), or" },

                new TextData(){ Text = "the inverse of the number of documents " +
                    "that contain the n-gram (Idf)," },

                new TextData(){ Text = "or compute both and multiply together " +
                    "(Tf-Idf)." },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for converting text into numeric n-gram features.
            // The following call to 'ProduceNgrams' requires the tokenized
            // text /string as input. This is achieved by calling 
            // 'TokenizeIntoWords' first followed by 'ProduceNgrams'. Please note
            // that the length of the output feature vector depends on the n-gram
            // settings.
            var textPipeline = mlContext.Transforms.Text.TokenizeIntoWords("Tokens",
                "Text")
                // 'ProduceNgrams' takes key type as input. Converting the tokens
                // into key type using 'MapValueToKey'.
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Tokens"))
                .Append(mlContext.Transforms.Text.ProduceNgrams("NgramFeatures",
                    "Tokens",
                    ngramLength: 3,
                    useAllLengths: false,
                    weighting: NgramExtractingEstimator.WeightingCriteria.Tf));

            // Fit to data.
            var textTransformer = textPipeline.Fit(dataview);
            var transformedDataView = textTransformer.Transform(dataview);

            // Create the prediction engine to get the n-gram features extracted
            // from the text.
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
            var NgramFeaturesColumn = transformedDataView.GetColumn<VBuffer<
                float>>(transformedDataView.Schema["NgramFeatures"]);
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
            //   Number of Features: 52
            //   N-grams:   This|is|an  is|an|example  an|example|to  example|to|compute  to|compute|n-grams.  N-gram|is|a  is|a|sequence  a|sequence|of  sequence|of|'N'  of|'N'|consecutive  ...
            //   Features:     1.0000      1.0000          1.0000           1.0000             1.0000            0.0000      0.0000          0.0000          0.0000          0.0000          ...
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
