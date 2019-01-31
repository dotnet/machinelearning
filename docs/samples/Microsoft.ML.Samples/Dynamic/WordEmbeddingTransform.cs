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
            // best game ive played.
            // == rude == dude
            // game best xbox game!
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
            //0.23166 0.048825 0.26878 -1.3945 -0.86072 -0.026778 0.84075 -0.81987 -1.6681 -1.0658 -0.30596 0.50974 -4.8382 -0.0023269 -0.21906 0.10287 -0.17618 -1.2881 -0.59801 -0.40189 -1.2619 0.036199 -1.0729 -0.75203 -0.2744 0.5431625 0.4797162 0.64641 -0.982955 -0.2745475 0.436228 1.271213 -0.006100006 -0.521805 -0.6021875 -0.05507075 0.77407 -4.275 0.2976883 0.4192 0.43081 0.569865 -0.488355 -0.2157775 0.05228249 -0.4918525 0.4082547 0.06644799 0.02899 -0.07697375 1.146 0.94124 1.0502 -0.75076 0.23411 0.77901 1.5832 0.55082 0.50478 -0.33335 0.1933 1.5332 -3.7655 0.50173 1.197 1.0368 1.1852 0.23622 0.41609 0.56327 0.60865 1.082 0.68583 0.97162 0.082578
            // -0.094905 0.61109 0.52546 -0.2516 0.054786 0.022661 1.1801 0.33329 -0.85388 0.15471 -0.5984 0.4364 -4.2488 -0.72646 0.85569 0.47611 0.92135 -0.73681 -0.20161 0.28397 1.0086 0.84342 0.56355 0.15887 0.038018 -0.094905 0.61109 0.52546 -0.2516 0.054786 0.022661 1.1801 0.33329 -0.85388 0.15471 -0.5984 0.4364 -4.2488 -0.72646 0.85569 0.47611 0.92135 -0.73681 -0.20161 0.28397 1.0086 0.84342 0.56355 0.15887 0.038018 -0.094905 0.61109 0.52546 -0.2516 0.054786 0.022661 1.1801 0.33329 -0.85388 0.15471 -0.5984 0.4364 -4.2488 -0.72646 0.85569 0.47611 0.92135 -0.73681 -0.20161 0.28397 1.0086 0.84342 0.56355 0.15887 0.038018
            //0.23166 0.048825 0.26878 -1.3945 -0.30044 -0.16523 0.47251 0.10276 -0.20978 -0.68094 -0.30596 -0.35186 -4.8382 -0.89017 -0.21906 0.10287 -0.17618 -1.2881 -0.59801 0.26131 -1.2619 -1.3399 -1.0729 -0.55232 -0.2744 0.94954 0.2666038 0.2820125 -1.106095 0.06427249 0.49387 1.190778 0.35036 0.276565 -0.5333 0.046042 0.29645 -4.1362 -0.02409922 0.6293225 0.2962225 0.07467099 -0.825569 -0.4053185 0.3832925 -1.124065 -0.1299153 0.292345 -0.2039025 -0.0352415 1.2745 0.35939 0.30032 -0.76236 0.62386 0.77901 1.5832 0.55082 0.50478 -0.44266 0.58754 0.51396 -3.0794 0.39805 1.197 0.74279 0.60195 0.093394 0.026736 0.56327 -0.76506 0.39202 1.0561 0.44869 0.12326

            // As you can see above we output 75 values for each line, despite fact we specify model for with 25 floats for each word.
            // We go through each word present in row and extract 25 floats for it (if we can find that word in model).
            // First 25 floats in output values represent minimum values (for each dimension) for extracted values. 
            // Second set of 25 floats in output represent average (for each dimension) for extracted values.
            // Third set of 25 floats in output represent maximum values (for each dimension) for extracted values.

        }
    }
}
