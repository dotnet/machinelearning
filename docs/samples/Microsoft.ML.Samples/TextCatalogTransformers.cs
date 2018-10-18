// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.Collections.Generic;

namespace Microsoft.ML.Samples
{
    public class TextCatalogTransformers
    {

       

        public static void Concat()
        {

        }

        public static void TextTransform()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext(seed: 1, conc: 1);

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleSentimentData> data = SamplesUtils.DatasetUtils.GetSentimentData();
            var trainData = ml.CreateStreamingDataView(data);

            // Preview of the data.
            // Sentiment    SentimentText
            // true         Best game I've ever played.
            // false        ==RUDE== Dude, 2.
            // true          Until the next game, this is the best Xbox game!

            // A pipeline for featurization of the "SentimentText" column, and placing the output in a new column named "TextFeatures"
            // making use of default settings.
            string defaultColumnName = "DefaultTextFeatures";
            var default_pipeline = ml.Transforms.Text.FeaturizeText("SentimentText", defaultColumnName);

            // Another pipeline, that customizes the advanced settings of the FeaturizeText transformer.
            string customizedColumnName = "CustomizedTextFeatures";
            var customized_pipeline = ml.Transforms.Text.FeaturizeText("SentimentText", customizedColumnName, s=>
            {
                s.KeepPunctuations = false;
                s.KeepNumbers = false;
                s.OutputTokens = true;
                s.TextLanguage = Runtime.Data.TextTransform.Language.English; // supports  English, French, German, Dutch, Italian, Spanish, Japanese
            });

            // The transformed data.
            var transformedData_default = default_pipeline.Fit(trainData).Transform(trainData);
            var transformedData_customized = customized_pipeline.Fit(trainData).Transform(trainData);

            // small helper to print the text inside the columns, in the console. 
            Action<string, VBuffer<float>[]> printHelper = (columnName, column) =>
            {
                Console.WriteLine($"{columnName} column obtained post-transformation.");
                foreach (var featureRow in column)
                {
                    foreach (var value in featureRow.Values)
                        Console.Write($"{value} ");
                    Console.WriteLine("");
                }

                Console.WriteLine("===================================================");
            };

            // Preview of the TextFeatures column obtained after processing the input.
            var defaultColumn = transformedData_default.GetColumn<VBuffer<float>>(ml, defaultColumnName).ToArray();
            printHelper(defaultColumnName, defaultColumn);

            // Transformed data  REVIEW: why are the first two lines identical? Log a bug. 
            // 0.2581989 0.2581989 0.2581989 0.2581989 0.5163978 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.7071068 0.7071068 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0.2581989 0.2581989 0.2581989 0.2581989 0.5163978 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.7071068 0.7071068 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0 0.1230915 0.1230915 0.1230915 0.1230915 0.246183 0.246183 0.246183 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.3692745 0.246183 0.246183 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.246183 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.2886751 0 0 0 0 0 0 0.2886751 0.5773503 0.2886751 0.2886751 0.2886751 0.2886751 0.2886751 0.2886751

            // Preview of the TextFeatures column obtained after processing the input.
            var customizedColumn = transformedData_customized.GetColumn<VBuffer<float>>(ml, customizedColumnName).ToArray();
            printHelper(customizedColumnName, customizedColumn);

            // Transformed data
            // 0.25 0.25 0.25 0.25 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.7071068 0.7071068 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0.25 0.25 0.25 0.25 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.7071068 0.7071068 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0 0.125 0.125 0.125 0.125 0.25 0.25 0.25 0.125 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.125 0.125 0.125 0.125 0.125 0.125 0.375 0.25 0.25 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.25 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.2672612 0.5345225 0 0 0 0 0 0.2672612 0.5345225 0.2672612 0.2672612 0.2672612 0.2672612
        }

        public static void MinMaxNormalizer()
        {

        }
    }
}
