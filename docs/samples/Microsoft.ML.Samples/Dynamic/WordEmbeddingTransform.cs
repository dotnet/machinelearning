using System;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
namespace Microsoft.ML.Samples.Dynamic
{
    class WordEmbeddingTransform
    {
        public static void ExtractEmbeddings()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert to IDataView.
            var data = SamplesUtils.DatasetUtils.GetSentimentData();
            var trainData = ml.Data.ReadFromEnumerable(data);

            // Preview of the data.
            //
            // Sentiment    SentimentText
            // true         Best game I've ever played.
            // false        ==RUDE== Dude, 2.
            // true          Until the next game, this is the best Xbox game!

            // Pipeline which goes through SentimentText and normalize it, tokenize it by words, and removes default stopwords.
            // After all that cleaning, we apply WordEmbedding with GloVeTwitter25D.
            // 25D means each word mapped into 25 dimensional space, basically each word represented by 25 float values.
            var wordsPipeline = ml.Transforms.Text.NormalizeText("NormalizedText", "SentimentText", keepDiacritics: false, keepPunctuations: false, keepNumbers: false)
                .Append(ml.Transforms.Text.TokenizeWords("Words", "NormalizedText"))
                .Append(ml.Transforms.Text.RemoveDefaultStopWords("CleanWords", "Words"))
                .Append(ml.Transforms.Text.ExtractWordEmbeddings("WordEmbeddings", "CleanWords", WordEmbeddingsExtractingTransformer.PretrainedModelKind.GloVeTwitter25D));

            var embeddingDataview = wordsPipeline.Fit(trainData).Transform(trainData);
            // Small helper to print the text inside the columns, in the console. 

            // Preview of the CleanWords column obtained after processing SentimentText.
            var cleanWords = embeddingDataview.GetColumn<VBuffer<ReadOnlyMemory<char>>>(ml, "CleanWords");
            Console.WriteLine($" CleanWords column obtained post-transformation.");
            foreach (var featureRow in cleanWords)
            {
                foreach (var value in featureRow.GetValues())
                    Console.Write($"{value} ");
                Console.WriteLine("");
            }

            Console.WriteLine("===================================================");
            // best game i've played.
            // == rude == dude
            // game, best xbox game!
            // Preview of WordEmbeddings.
            var embeddings = embeddingDataview.GetColumn<VBuffer<float>>(ml, "WordEmbeddings");
            Console.WriteLine($" WordEmbeddings column obtained post-transformation.");
            foreach (var featureRow in embeddings)
            {
                foreach (var value in featureRow.GetValues())
                    Console.Write($"{value} ");
                Console.WriteLine("");
            }

            Console.WriteLine("===================================================");
            //0.23166 0.048825 0.26878 -1.3945 -0.30044 0.58269 1.3537 0.37393 -0.20978 -0.56694 -0.30596 0.50974 -4.8382 -0.0023269 -0.21906 0.10287 -0.17618 -1.2881 -0.59801 0.26131 -1.2619 0.036199 -1.0729 -0.55232 -0.2744 0.68883 0.1889625 0.279475 -1.07843 -0.03316501 0.68085 1.46845 0.462375 0.1475 -0.5048 -0.177333 0.51185 -4.5759 0.1978616 0.48897 0.169615 0.212885 -1.053785 -0.525 0.41229 -1.01348 0.2141095 -0.239905 -0.051815 -0.1346565 1.146 0.3291 0.29017 -0.76236 0.23411 0.77901 1.5832 0.55082 0.50478 -0.44266 -0.048706 0.51396 -4.3136 0.39805 1.197 0.23636 0.60195 -0.81947 -0.45199 0.56327 -0.76506 0.39202 0.59309 0.44869 0.005087
            // -0.094905 0.61109 0.52546 -0.2516 0.054786 0.022661 1.1801 0.33329 -0.85388 0.15471 -0.5984 0.4364 -4.2488 -0.72646 0.85569 0.47611 0.92135 -0.73681 -0.20161 0.28397 1.0086 0.84342 0.56355 0.15887 0.038018 -0.094905 0.61109 0.52546 -0.2516 0.054786 0.022661 1.1801 0.33329 -0.85388 0.15471 -0.5984 0.4364 -4.2488 -0.72646 0.85569 0.47611 0.92135 -0.73681 -0.20161 0.28397 1.0086 0.84342 0.56355 0.15887 0.038018 -0.094905 0.61109 0.52546 -0.2516 0.054786 0.022661 1.1801 0.33329 -0.85388 0.15471 -0.5984 0.4364 -4.2488 -0.72646 0.85569 0.47611 0.92135 -0.73681 -0.20161 0.28397 1.0086 0.84342 0.56355 0.15887 0.038018
            //0.23166 0.048825 0.29017 - 0.87302 0.23411 - 0.16523 0.47251 0.10276 - 0.20978 - 0.68094 - 0.30596 - 0.35186 - 4.8382 - 0.89017 - 0.21906 0.23636 0.049094 - 0.81947 - 0.45199 0.44728 - 1.2074 - 1.3399 - 1.0729 - 0.15966 - 0.2744 0.75308 0.2041075 0.295245 - 0.81769 0.428985 0.20873 1.027855 0.32679 0.04835 - 0.62394 0.14079 0.07894 - 3.9588 - 0.4462484 0.061645 0.489575 0.325522 - 0.363038 - 0.212627 0.505275 - 0.98623 - 0.6518505 - 0.008400023 0.144515 - 0.07557 1.2745 0.35939 0.30032 - 0.76236 0.62386 0.58269 1.5832 0.55082 0.30648 - 0.56694 0.58754 0.50974 - 3.0794 - 0.0023269 0.34235 0.74279 0.60195 0.093394 0.026736 0.56327 - 0.76506 0.036199 1.0561 0.44869 0.12326

            // As you can see above we output 75 values for each line, despite fact we specify model for with 25 floats for each word.
            // We go through each word present in row and extract 25 floats for it (if we can find that word in model).
            // First 25 floats in output values represent minimum values (for each dimension) for extracted values. 
            // Second set of 25 floats in output represent average (for each dimension) for extracted values.
            // Third set of 25 floats in output represent maximum values (for each dimension) for extracted values.

        }
    }
}
