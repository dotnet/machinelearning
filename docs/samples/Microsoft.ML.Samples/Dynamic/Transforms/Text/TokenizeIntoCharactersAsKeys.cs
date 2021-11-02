using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class TokenizeIntoCharactersAsKeys
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create an empty list as the dataset. The
            // 'TokenizeIntoCharactersAsKeys' does not require training data as
            // the estimator ('TokenizingByCharactersEstimator') created by
            // 'TokenizeIntoCharactersAsKeys' API is not a trainable estimator.
            // The empty list is only needed to pass input schema to the pipeline.
            var emptySamples = new List<TextData>();

            // Convert sample list to an empty IDataView.
            var emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples);

            // A pipeline for converting text into vector of characters.
            // The 'TokenizeIntoCharactersAsKeys' produces result as key type.
            // 'MapKeyToValue' is need to map keys back to their original values.
            var textPipeline = mlContext.Transforms.Text
                .TokenizeIntoCharactersAsKeys("CharTokens", "Text",
                    useMarkerCharacters: false)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    "CharTokens"));

            // Fit to data.
            var textTransformer = textPipeline.Fit(emptyDataView);

            // Create the prediction engine to get the character vector from the
            // input text/string.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData,
                TransformedTextData>(textTransformer);

            // Call the prediction API to convert the text into characters.
            var data = new TextData()
            {
                Text = "ML.NET's " +
                "TokenizeIntoCharactersAsKeys API splits text/string into " +
                "characters."
            };

            var prediction = predictionEngine.Predict(data);

            // Print the length of the character vector.
            Console.WriteLine($"Number of tokens: {prediction.CharTokens.Length}");

            // Print the character vector.
            Console.WriteLine("\nCharacter Tokens: " + string.Join(",", prediction
                .CharTokens));

            //  Expected output:
            //   Number of tokens: 77
            //   Character Tokens: M,L,.,N,E,T,',s,<?>,T,o,k,e,n,i,z,e,I,n,t,o,C,h,a,r,a,c,t,e,r,s,A,s,K,e,y,s,<?>,A,P,I,<?>,
            //                     s,p,l,i,t,s,<?>,t,e,x,t,/,s,t,r,i,n,g,<?>,i,n,t,o,<?>,c,h,a,r,a,c,t,e,r,s,.
            //
            // <?>: is a unicode control character used instead of spaces ('\u2400').
        }

        private class TextData
        {
            public string Text { get; set; }
        }

        private class TransformedTextData : TextData
        {
            public string[] CharTokens { get; set; }
        }
    }
}
