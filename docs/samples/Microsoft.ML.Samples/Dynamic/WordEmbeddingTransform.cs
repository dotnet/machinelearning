using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
namespace Microsoft.ML.Samples.Dynamic
{
    public static class WordEmbeddingTransform
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

            // Pipeline which goes through SentimentText and normalizes it, tokenize it by words, and removes default stopwords.
            var wordsPipeline = ml.Transforms.Text.NormalizeText("NormalizedText", "SentimentText", keepDiacritics: false, keepPunctuations: false)
                .Append(ml.Transforms.Text.TokenizeIntoWords("Words", "NormalizedText"))
                .Append(ml.Transforms.Text.RemoveDefaultStopWords("CleanWords", "Words"));

            var wordsDataview = wordsPipeline.Fit(trainData).Transform(trainData);
            // Preview of the CleanWords column obtained after processing SentimentText.
            var cleanWords = wordsDataview.GetColumn<ReadOnlyMemory<char>[]>(wordsDataview.Schema["CleanWords"]);
            Console.WriteLine($" CleanWords column obtained post-transformation.");
            foreach (var featureRow in cleanWords)
            {
                foreach (var value in featureRow)
                    Console.Write($"{value} ");
                Console.WriteLine("");
            }

            Console.WriteLine("===================================================");
            // best game ive played
            // == rude == dude 2
            // game best xbox game

            // Small helper to print wordembeddings in the console. 
            Action<string, IEnumerable<float[]>> printEmbeddings = (columnName, column) =>
            {
                Console.WriteLine($"{columnName} column obtained post-transformation.");
                foreach (var featureRow in column)
                {
                    foreach (var value in featureRow)
                        Console.Write($"{value} ");
                    Console.WriteLine("");
                }

                Console.WriteLine("===================================================");
            };

            // Let's apply pretrained word embedding model GloVeTwitter25D.
            // 25D means each word mapped into 25 dimensional space, basically each word represented by 25 float values.
            var gloveWordEmbedding = ml.Transforms.Text.ApplyWordEmbedding("GloveEmbeddings", "CleanWords",
                WordEmbeddingEstimator.PretrainedModelKind.GloVeTwitter25D);

            // We also have option to apply custom word embedding models.
            // Let's first create one.
            // Format is following:
            // First line is ignored if it is a header for your file.
            // Each next line contains a single word followed by either a tab or space, and a list of floats also separated by a tab or space.
            // Size of array of floats should be same for whole file.
            var pathToCustomModel = @".\custommodel.txt";
            using (StreamWriter file = new StreamWriter(pathToCustomModel, false))
            {

                file.WriteLine("This is custom file for 4 words with 3 dimensional word embedding vector. This first line in this file does not conform to the '<word> <float> <float> <float>' pattern, and is therefore ignored");
                file.WriteLine("xbox" + " " + string.Join(" ", 1.0f, 2.0f, 3.0f));
                file.WriteLine("game" + " " + string.Join(" ", -1.0f, -2.0f, -3.0f));
                file.WriteLine("dude" + " " + string.Join(" ", -1f, 100.0f, -100f));
                file.WriteLine("best" + " " + string.Join(" ", 0f, 0f, 20f));
            }
            // Now let's add custom embedding on top of same words.
            var pipeline = gloveWordEmbedding.Append(ml.Transforms.Text.ApplyWordEmbedding("CustomEmbeddings", @".\custommodel.txt", "CleanWords"));

            // And do all required transformations.
            var embeddingDataview = pipeline.Fit(wordsDataview).Transform(wordsDataview);

            var customEmbeddings = embeddingDataview.GetColumn<float[]>(embeddingDataview.Schema["CustomEmbeddings"]);
            printEmbeddings("GloveEmbeddings", customEmbeddings);

            // -1  -2   -3  -0.5   -1  8.5  0   0   20
            // -1 100 -100    -1  100 -100 -1 100 -100
            //  1  -2   -3 -0.25 -0.5 4.25  1   2   20
            // As you can see above we output 9 values for each line
            // We go through each word present in row and extract 3 floats for it (if we can find that word in model).
            // First 3 floats in output values represent minimum values (for each dimension) for extracted values. 
            // Second set of 3 floats in output represent average (for each dimension) for extracted values.
            // Third set of 3 floats in output represent maximum values (for each dimension) for extracted values.
            // Preview of GloveEmbeddings.
            var gloveEmbeddings = embeddingDataview.GetColumn<float[]>(embeddingDataview.Schema["GloveEmbeddings"]);
            printEmbeddings("GloveEmbeddings", gloveEmbeddings);
            // 0.23166 0.048825 0.26878 -1.3945 -0.86072 -0.026778 0.84075 -0.81987 -1.6681 -1.0658 -0.30596 0.50974 ...
            //-0.094905 0.61109 0.52546 - 0.2516 0.054786 0.022661 1.1801 0.33329 - 0.85388 0.15471 - 0.5984 0.4364  ...
            // 0.23166 0.048825 0.26878 - 1.3945 - 0.30044 - 0.16523 0.47251 0.10276 - 0.20978 - 0.68094 - 0.30596  ...

        }
    }
}
