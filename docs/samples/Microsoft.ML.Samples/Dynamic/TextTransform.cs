        // the alignment of the usings with the methods is intentional so they can display on the same level in the docs site.
        using Microsoft.ML.Runtime.Data;
        using Microsoft.ML.Runtime.Api;
        using Microsoft.ML.Data;
        using Microsoft.ML.Transforms.Text;
        using System;
        using System.Collections.Generic;

namespace Microsoft.ML.Samples.Dynamic
{
    public partial class TransformSamples
    {
        public static void TextTransform()
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

            // A pipeline for featurization of the "SentimentText" column, and placing the output in a new column named "DefaultTextFeatures"
            // The pipeline uses the default settings to featurize.
            string defaultColumnName = "DefaultTextFeatures";
            var default_pipeline = ml.Transforms.Text.FeaturizeText("SentimentText", defaultColumnName);

            // Another pipeline, that customizes the advanced settings of the FeaturizeText transformer.
            string customizedColumnName = "CustomizedTextFeatures";
            var customized_pipeline = ml.Transforms.Text.FeaturizeText("SentimentText", customizedColumnName, s =>
            {
                s.KeepPunctuations = false;
                s.KeepNumbers = false;
                s.OutputTokens = true;
                s.TextLanguage = TextFeaturizingEstimator.Language.English; // supports  English, French, German, Dutch, Italian, Spanish, Japanese
            });

            // The transformed data for both pipelines.
            var transformedData_default = default_pipeline.Fit(trainData).Transform(trainData);
            var transformedData_customized = customized_pipeline.Fit(trainData).Transform(trainData);

            // Small helper to print the text inside the columns, in the console. 
            Action<string, IEnumerable<VBuffer<float>>> printHelper = (columnName, column) =>
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

            // Preview of the DefaultTextFeatures column obtained after processing the input.
            var defaultColumn = transformedData_default.GetColumn<VBuffer<float>>(ml, defaultColumnName);
            printHelper(defaultColumnName, defaultColumn);

            // DefaultTextFeatures column obtained post-transformation.
            //
            // 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0.2357023 0.2357023 0.2357023 0.2357023 0.4714046 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.2357023 0.5773503 0.5773503 0.5773503 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0 0.1230915 0.1230915 0.1230915 0.1230915 0.246183 0.246183 0.246183 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1230915 0 0 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.3692745 0.246183 0.246183 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.246183 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.2886751 0 0 0 0 0 0 0 0.2886751 0.5773503 0.2886751 0.2886751 0.2886751 0.2886751 0.2886751 0.2886751

            // Preview of the CustomizedTextFeatures column obtained after processing the input.
            var customizedColumn = transformedData_customized.GetColumn<VBuffer<float>>(ml, customizedColumnName);
            printHelper(customizedColumnName, customizedColumn);

            // CustomizedTextFeatures column obtained post-transformation.
            //
            // 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0.25 0.25 0.25 0.25 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.7071068 0.7071068 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0 0.125 0.125 0.125 0.125 0.25 0.25 0.25 0.125 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.125 0.125 0.125 0.125 0.125 0.125 0.375 0.25 0.25 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.25 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.2672612 0.5345225 0 0 0 0 0 0.2672612 0.5345225 0.2672612 0.2672612 0.2672612 0.2672612        }
        }
    }
}
