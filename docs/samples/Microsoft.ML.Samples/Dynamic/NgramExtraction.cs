using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Samples.Dynamic
{
    public partial class NgramTransformSamples
    {
        public static void NgramTransform()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert to IDataView.
            IEnumerable<SamplesUtils.DatasetUtils.SampleSentimentData> data = SamplesUtils.DatasetUtils.GetSentimentData();
            var trainData = ml.CreateStreamingDataView(data);

            // Preview of the data.
            //
            // Sentiment    SentimentText
            // true         Best game I've ever played.
            // false        ==RUDE== Dude, 2.
            // true          Until the next game, this is the best Xbox game!

            // A pipeline to tokenize text as characters and then combine them together into ngrams
            // The pipeline uses the default settings to featurize.

            var charsPipeline = ml.Transforms.Text.TokenizeCharacters("SentimentText", "Chars", useMarkerCharacters:false);
            var ngramOnePipeline = ml.Transforms.Text.ProduceNgrams("Chars", "CharsOnegrams", ngramLength:1);
            var ngramTwpPipeline = ml.Transforms.Text.ProduceNgrams("Chars", "CharsTwograms");
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
            // Preview of the CharsOnegrams column obtained after processing the input.
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            transformedData_onechars.Schema["CharsOnegrams"].Metadata.GetValue(MetadataUtils.Kinds.SlotNames, ref slotNames);
            var charsOneGramColumn = transformedData_onechars.GetColumn<VBuffer<float>>(ml, "CharsOnegrams");
            printHelper("CharsOnegrams", charsOneGramColumn, slotNames);

            // CharsTwograms column obtained post-transformation.
            // 'B' - 1 'e' - 6 's' - 1 't' - 1 '<?>' - 4 'g' - 1 'a' - 2 'm' - 1 'I' - 1 ''' - 1 'v' - 2 'r' - 1 'p' - 1 'l' - 1 'y' - 1 'd' - 1 '.' - 1 '=' - 0 'R' - 0 'U' - 0 'D' - 0 'E' - 0 'u' - 0 ',' - 0 '2' - 0 'n' - 0 'i' - 0 'h' - 0 'x' - 0 'b' - 0 'X' - 0 'o' - 0 '!' - 0
            // 'e' - 1 '<?>' - 2 'd' - 1 '=' - 4 'R' - 1 'U' - 1 'D' - 2 'E' - 1 'u' - 1 ',' - 1 '2' - 1
            // 'B' - 0 'e' - 6 's' - 3 't' - 6 '<?>' - 9 'g' - 2 'a' - 2 'm' - 2 'I' - 0 ''' - 0 'v' - 0 'r' - 0 'p' - 0 'l' - 1 'y' - 0 'd' - 0 '.' - 0 ' = ' - 0 'R' - 0 'U' - 1 'D' - 0 'E' - 0 'u' - 0 ',' - 1 '2' - 0 'n' - 2 'i' - 3 'h' - 3 'x' - 2 'b' - 2 'X' - 1 'o' - 1 '!' - 1
            // Preview of the CharsTwoGrams column obtained after processing the input.
            var charsTwoGramColumn = transformedData_twochars.GetColumn<VBuffer<float>>(ml, "CharsTwograms");
            transformedData_twochars.Schema["CharsTwograms"].Metadata.GetValue(MetadataUtils.Kinds.SlotNames, ref slotNames);
            printHelper("CharsTwograms", charsTwoGramColumn, slotNames);

            // CharsTwograms column obtained post-transformation.
            // 'B' - 1 'B|e' - 1 'e' - 6 'e|s' - 1 's' - 1 's|t' - 1 't' - 1 't|<?>' - 1 '<?>' - 4 '<?>|g' - 1 'g' - 1 'g|a' - 1 'a' - 2 'a|m' - 1 'm' - 1 'm|e' - 1 'e|<?>' - 2 '<?>|I' - 1 'I' - 1 'I|'' - 1 ''' - 1 '' | v' - 1 'v' - 2 'v | e' - 2 ' <?>| e' - 1 'e | v' - 1 'e | r' - 1 'r' - 1 'r |<?> ' - 1 ' <?>| p' - 1 'p' - 1 'p | l' - 1 'l' - 1 'l | a' - 1 'a | y' - 1 'y' - 1 'y | e' - 1 'e | d' - 1 'd' - 1 'd |.' - 1 '.' - 1
            // 'e' - 1 '<?>' - 2 'd' - 1 '=' - 4 '=|=' - 2 '=|R' - 1 'R' - 1 'R|U' - 1 'U' - 1 'U|D' - 1 'D' - 2 'D|E' - 1 'E' - 1 'E|=' - 1 '=|<?>' - 1 '<?>|D' - 1 'D|u' - 1 'u' - 1 'u|d' - 1 'd|e' - 1 'e|,' - 1 ',' - 1 ',|<?>' - 1 '<?>|2' - 1 '2' - 1
            // 'B' - 0 'B|e' - 0 'e' - 6 'e|s' - 1 's' - 3 's|t' - 1 't' - 6 't|<?>' - 2 '<?>' - 9 '<?>|g' - 2 'g' - 2 'g|a' - 2 'a' - 2 'a|m' - 2 'm' - 2 'm|e' - 2 'e|<?>' - 2 '<?>|I' - 0 'I' - 0 'I|'' - 0 ''' - 0 '' | v' - 0 'v' - 0 'v | e' - 0 ' <?>| e' - 0 'e | v' - 0 'e | r' - 0 'r' - 0 'r |<?> ' - 0 ' <?>| p' - 0 'p' - 0 'p | l' - 0 'l' - 1 'l | a' - 0 'a | y' - 0 'y' - 0 'y | e' - 0 'e | d' - 0 'd' - 0 'd |.' - 0 '.' - 0 ' = ' - 0 ' =|= ' - 0 ' =| R' - 0 'R' - 0 'R | U' - 0 'U' - 1 'U | D' - 0 'D' - 0 'D | E' - 0 'E' - 0 'E |= ' - 0 ' =|<?> ' - 0 ' <?>| D' - 0 'D | u' - 0 'u' - 0 'u | d' - 0 'd | e' - 0 'e |,' - 1 ',' - 1 ',|<?> ' - 1 ' <?>| 2' - 0 '2' - 0 'U | n' - 1 'n' - 2 'n | t' - 1 't | i' - 1 'i' - 3 'i | l' - 1 'l |<?> ' - 1 ' <?>| t' - 3 't | h' - 3 'h' - 3 'h | e' - 2 ' <?>| n' - 1 'n | e' - 1 'e | x' - 1 'x' - 2 'x | t' - 1 'h | i' - 1 'i | s' - 2 's |<?> ' - 2 ' <?>| i' - 1 ' <?>| b' - 1 'b' - 2 'b | e' - 1 ' <?>| X' - 1 'X' - 1 'X | b' - 1 'b | o' - 1 'o' - 1 'o | x' - 1 'x |<?> ' - 1 'e | !' - 1 '!' - 1
        }
    }
}
