// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Transforms.Text;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public static void WordBags()
        {
            var mlContext = new MLContext(1);
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "This is an example to compute bag-of-word features." },
                new TextData(){ Text = "ML.NET's ProduceWordBags API produces bag-of-word features from input text." },
                new TextData(){ Text = "It does so by first tokenizing text/string into words/tokens then " },
                new TextData(){ Text = "computing n-grams and their numeric values." },
                new TextData(){ Text = "Each position in the output vector corresponds to a particular n-gram." },
                new TextData(){ Text = "The value at each position corresponds to," },
                new TextData(){ Text = "the number of times n-gram occurred in the data (Tf), or" },
                new TextData(){ Text = "the inverse of the number of documents contain the n-gram (Idf)," },
                new TextData(){ Text = "or compute both and multiply together (Tf-Idf)." },
            };

            var dataview = mlContext.Data.LoadFromEnumerable(samples);
            var textPipeline =
                mlContext.Transforms.Text.ProduceWordBags("Text", "Text",
                ngramLength: 3, useAllLengths: false, weighting: NgramExtractingEstimator.WeightingCriteria.Tf).Append(
                mlContext.Transforms.Text.ProduceWordBags("Text2", new[] { "Text2", "Text2" },
                    ngramLength: 3, useAllLengths: false, weighting: NgramExtractingEstimator.WeightingCriteria.Tf));


            var textTransformer = textPipeline.Fit(dataview);
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(textTransformer);
            var prediction = predictionEngine.Predict(samples[0]);
            Assert.Equal(prediction.Text, new float[] {
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

            Assert.Equal(prediction.Text2, new float[] { 2, 2, 2, 2, 2, 2, 1, 1 });
        }

        [Fact]
        public static void WordBagsHash()
        {
            var mlContext = new MLContext(1);
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "This is an example to compute bag-of-word features." },
                new TextData(){ Text = "ML.NET's ProduceWordBags API produces bag-of-word features from input text." },
                new TextData(){ Text = "It does so by first tokenizing text/string into words/tokens then " },
                new TextData(){ Text = "computing n-grams and their numeric values." },
                new TextData(){ Text = "Each position in the output vector corresponds to a particular n-gram." },
                new TextData(){ Text = "The value at each position corresponds to," },
                new TextData(){ Text = "the number of times n-gram occurred in the data (Tf), or" },
                new TextData(){ Text = "the inverse of the number of documents contain the n-gram (Idf)," },
                new TextData(){ Text = "or compute both and multiply together (Tf-Idf)." },
            };

            var dataview = mlContext.Data.LoadFromEnumerable(samples);
            var textPipeline =
                mlContext.Transforms.Text.ProduceHashedWordBags("Text", "Text", ngramLength: 3, useAllLengths: false).Append(
                mlContext.Transforms.Text.ProduceHashedWordBags("Text2", new[] { "Text2", "Text2" }, ngramLength: 3, useAllLengths: false));


            var textTransformer = textPipeline.Fit(dataview);
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(textTransformer);
            var prediction = predictionEngine.Predict(samples[0]);
            Assert.Equal(65536, prediction.Text.Length);
        }

        private class TextData
        {
            public string Text { get; set; }
#pragma warning disable 414
            public string Text2 = "This is an example to compute bag-of-word features.";
#pragma warning restore 414
        }

        private class TransformedTextData
        {
            public float[] Text { get; set; }
            public float[] Text2 { get; set; }
        }
    }

}
