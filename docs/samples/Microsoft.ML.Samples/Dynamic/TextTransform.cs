using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class TextTransform
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

            // A pipeline for featurization of the "SentimentText" column, and placing the output in a new column named "DefaultTextFeatures"
            // The pipeline uses the default settings to featurize.
            string defaultColumnName = "DefaultTextFeatures";
            var default_pipeline = ml.Transforms.Text.FeaturizeText(defaultColumnName , "SentimentText");

            // Another pipeline, that customizes the advanced settings of the FeaturizeText transformer.
            string customizedColumnName = "CustomizedTextFeatures";
            var customized_pipeline = ml.Transforms.Text.FeaturizeText(customizedColumnName, new TextFeaturizingEstimator.Options
            {
                KeepPunctuations = false,
                KeepNumbers = false,
                OutputTokensColumnName = "OutputTokens",
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English }, // supports  English, French, German, Dutch, Italian, Spanish, Japanese
            }, "SentimentText");

            // The transformed data for both pipelines.
            var transformedData_default = default_pipeline.Fit(trainData).Transform(trainData);
            var transformedData_customized = customized_pipeline.Fit(trainData).Transform(trainData);

            // Small helper to print the text inside the columns, in the console. 
            Action<string, IEnumerable<VBuffer<float>>> printHelper = (columnName, column) =>
            {
                Console.WriteLine($"{columnName} column obtained post-transformation.");
                foreach (var featureRow in column)
                {
                    foreach (var value in featureRow.GetValues())
                        Console.Write($"{value} ");
                    Console.WriteLine("");
                }

                Console.WriteLine("===================================================");
            };

            // Preview of the DefaultTextFeatures column obtained after processing the input.
            var defaultColumn = transformedData_default.GetColumn<VBuffer<float>>(transformedData_default.Schema[defaultColumnName]);
            printHelper(defaultColumnName, defaultColumn);

            // DefaultTextFeatures column obtained post-transformation.
            //
            // 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0.2357023 0.2357023 0.2357023 0.2357023 0.4714046 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.5773503 0.5773503 0.5773503 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0 0.1230915 0.1230915 0.1230915 0.1230915 0.246183 0.246183 0.246183 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1230915 0 0 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.3692745 0.246183 0.246183 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.246183 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.2886751 0 0 0 0 0 0 0 0.2886751 0.5773503 0.2886751 0.2886751 0.2886751 0.2886751 0.2886751 0.2886751

            // Preview of the CustomizedTextFeatures column obtained after processing the input.
            var customizedColumn = transformedData_customized.GetColumn<VBuffer<float>>(transformedData_customized.Schema[customizedColumnName]);
            printHelper(customizedColumnName, customizedColumn);

            // CustomizedTextFeatures column obtained post-transformation.
            //
            // 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0.25 0.25 0.25 0.25 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.7071068 0.7071068 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0 0.125 0.125 0.125 0.125 0.25 0.25 0.25 0.125 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.125 0.125 0.125 0.125 0.125 0.125 0.375 0.25 0.25 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.25 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.2672612 0.5345225 0 0 0 0 0 0.2672612 0.5345225 0.2672612 0.2672612 0.2672612 0.2672612        }
        }
    }
}
