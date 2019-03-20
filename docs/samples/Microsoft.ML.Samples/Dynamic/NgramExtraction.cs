using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static partial class TransformSamples
    {
        public static void NgramTransform()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert to IDataView.
            IEnumerable<SamplesUtils.DatasetUtils.SampleSentimentData> data = SamplesUtils.DatasetUtils.GetSentimentData();
            var trainData = ml.Data.LoadFromEnumerable(data);

            // Preview of the data.
            //
            // Sentiment    SentimentText
            // true         Best game I've ever played.
            // false        ==RUDE== Dude, 2.
            // true          Until the next game, this is the best Xbox game!

            // A pipeline to tokenize text as characters and then combine them together into ngrams
            // The pipeline uses the default settings to featurize.

            var charsPipeline = ml.Transforms.Text.TokenizeIntoCharactersAsKeys("Chars", "SentimentText", useMarkerCharacters: false);
            var ngramOnePipeline = ml.Transforms.Text.ProduceNgrams("CharsUnigrams", "Chars", ngramLength: 1);
            var ngramTwpPipeline = ml.Transforms.Text.ProduceNgrams("CharsTwograms", "Chars");
            var oneCharsPipeline = charsPipeline.Append(ngramOnePipeline);
            var twoCharsPipeline = charsPipeline.Append(ngramTwpPipeline);

            // The transformed data for pipelines.
            var transformedData_onechars = oneCharsPipeline.Fit(trainData).Transform(trainData);
            var transformedData_twochars = twoCharsPipeline.Fit(trainData).Transform(trainData);

            // Small helper to print the text inside the columns, in the console. 
            Action<string, IEnumerable<VBuffer<float>>, VBuffer<ReadOnlyMemory<char>>> printHelper = (columnName, column, names) =>
            {
                Console.WriteLine($"{columnName} column obtained post-transformation.");
                var slots = names.GetValues();
                foreach (var featureRow in column)
                {
                    foreach (var item in featureRow.Items())
                        Console.Write($"'{slots[item.Key]}' - {item.Value} ");
                    Console.WriteLine("");
                }

                Console.WriteLine("===================================================");
            };
            // Preview of the CharsUnigrams column obtained after processing the input.
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            transformedData_onechars.Schema["CharsUnigrams"].GetSlotNames(ref slotNames);
            var charsOneGramColumn = transformedData_onechars.GetColumn<VBuffer<float>>(transformedData_onechars.Schema["CharsUnigrams"]);
            printHelper("CharsUnigrams", charsOneGramColumn, slotNames);

            // CharsUnigrams column obtained post-transformation.
            // 'B' - 1 'e' - 6 's' - 1 't' - 1 '<?>' - 4 'g' - 1 'a' - 2 'm' - 1 'I' - 1 ''' - 1 'v' - 2 ...
            // 'e' - 1 '<?>' - 2 'd' - 1 '=' - 4 'R' - 1 'U' - 1 'D' - 2 'E' - 1 'u' - 1 ',' - 1 '2' - 1
            // 'B' - 0 'e' - 6 's' - 3 't' - 6 '<?>' - 9 'g' - 2 'a' - 2 'm' - 2 'I' - 0 ''' - 0 'v' - 0 ...
            // Preview of the CharsTwoGrams column obtained after processing the input.
            var charsTwoGramColumn = transformedData_twochars.GetColumn<VBuffer<float>>(transformedData_twochars.Schema["CharsTwograms"]);
            transformedData_twochars.Schema["CharsTwograms"].GetSlotNames(ref slotNames);
            printHelper("CharsTwograms", charsTwoGramColumn, slotNames);

            // CharsTwograms column obtained post-transformation.
            // 'B' - 1 'B|e' - 1 'e' - 6 'e|s' - 1 's' - 1 's|t' - 1 't' - 1 't|<?>' - 1 '<?>' - 4 '<?>|g' - 1 ...
            // 'e' - 1 '<?>' - 2 'd' - 1 '=' - 4 '=|=' - 2 '=|R' - 1 'R' - 1 'R|U' - 1 'U' - 1 'U|D' - 1 'D' - 2 ...
            // 'B' - 0 'B|e' - 0 'e' - 6 'e|s' - 1 's' - 3 's|t' - 1 't' - 6 't|<?>' - 2 '<?>' - 9 '<?>|g' - 2 ...
        }
    }
}
