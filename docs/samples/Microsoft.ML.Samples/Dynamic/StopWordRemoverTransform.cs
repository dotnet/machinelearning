using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class StopWordRemoverTransform
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert to IDataView.
            var data = SamplesUtils.DatasetUtils.GetSentimentData();
            var trainData = ml.Data.LoadFromEnumerable(data);

            // Preview of the data.
            //
            // Sentiment    SentimentText
            // true         Best game I've ever played.
            // false        ==RUDE== Dude, 2.
            // true          Until the next game, this is the best Xbox game!

            // Let's take SentimentText column and break it into vector of words.
            string originalTextColumnName = "Words";
            var words = ml.Transforms.Text.TokenizeIntoWords("SentimentText", originalTextColumnName);

            // Default pipeline will apply default stop word remover which is based on predifined set of words for certain languages.
            var defaultPipeline = words.Append(ml.Transforms.Text.RemoveDefaultStopWords(originalTextColumnName, "DefaultRemover"));

            // Another pipeline, that removes words specified by user. We do case insensitive comparison for the stop words.
            var customizedPipeline = words.Append(ml.Transforms.Text.RemoveStopWords(originalTextColumnName, "RemovedWords",
                new[] { "XBOX" }));

            // The transformed data for both pipelines.
            var transformedDataDefault = defaultPipeline.Fit(trainData).Transform(trainData);
            var transformedDataCustomized = customizedPipeline.Fit(trainData).Transform(trainData);

            // Small helper to print the text inside the columns, in the console. 
            Action<string, IEnumerable<VBuffer<ReadOnlyMemory<char>>>> printHelper = (columnName, column) =>
            {
                Console.WriteLine($"{columnName} column obtained post-transformation.");
                foreach (var featureRow in column)
                {
                    foreach (var value in featureRow.GetValues())
                        Console.Write($"{value}|");
                    Console.WriteLine("");
                }

                Console.WriteLine("===================================================");
            };

            // Preview the result of breaking string into array of words.
            var originalText = transformedDataDefault.GetColumn<VBuffer<ReadOnlyMemory<char>>>(transformedDataDefault.Schema[originalTextColumnName]);
            printHelper(originalTextColumnName, originalText);
            // Best|game|I've|ever|played.|
            // == RUDE ==| Dude,| 2 |
            // Until | the | next | game,| this |is| the | best | Xbox | game!|

            // Preview the result of cleaning with default stop word remover.
            var defaultRemoverData = transformedDataDefault.GetColumn<VBuffer<ReadOnlyMemory<char>>>(transformedDataDefault.Schema["DefaultRemover"]);
            printHelper("DefaultRemover", defaultRemoverData);
            // Best|game|I've|played.|
            // == RUDE ==| Dude,| 2 |
            // game,| best | Xbox | game!|
            // As you can see "Until, the, next, this, is" was removed.


            // Preview the result of cleaning with default customized stop word remover.
            var customizeRemoverData = transformedDataCustomized.GetColumn<VBuffer<ReadOnlyMemory<char>>>(transformedDataCustomized.Schema["RemovedWords"]);
            printHelper("RemovedWords", customizeRemoverData);

            // Best|game|I've|ever|played.|
            // == RUDE ==| Dude,| 2 |
            // Until | the | next | game,| this |is| the | best | game!|
            //As you can see Xbox was removed.

        }
    }
}
