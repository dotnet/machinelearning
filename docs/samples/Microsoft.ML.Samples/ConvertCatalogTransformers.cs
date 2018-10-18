// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Samples
{
    public class ConvertCatalogTransformers
    {
        public static void KeyToValue_Term()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext(seed: 1, conc: 1);

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleTopicsData> data = SamplesUtils.DatasetUtils.GetTopicsData();
            var trainData = ml.CreateStreamingDataView(data);

            // Preview of the data.
            // Review                                    ReviewReverse,                 Label
            // "animals birds cats dogs fish horse", "radiation galaxy universe duck",     1
            // "horse birds house fish duck cats",   "space galaxy universe radiation",    0
            // "car truck driver bus pickup",        "bus pickup",                         1
            // "car truck driver bus pickup horse",  "car truck",                          0

            // A pipeline to convert the terms of the review_reverse column in 
            // making use of default settings.
            string defaultColumnName = "DefaultKeys";
            // REVIEW create through the catalog extension
            var default_pipeline = new WordTokenizer(ml, "ReviewReverse", "ReviewReverse")
                .Append(new TermEstimator(ml, "ReviewReverse" , defaultColumnName));

            // Another pipeline, that customizes the advanced settings of the FeaturizeText transformer.
            string customizedColumnName = "CustomizedKeys";
            var customized_pipeline = new WordTokenizer(ml, "ReviewReverse", "ReviewReverse")
                .Append(new TermEstimator(ml, "ReviewReverse", customizedColumnName, maxNumTerms: 3, sort:TermTransform.SortOrder.Value));

            // The transformed data.
            var transformedData_default = default_pipeline.Fit(trainData).Transform(trainData);
            var transformedData_customized = customized_pipeline.Fit(trainData).Transform(trainData);

            // small helper to print the text inside the columns, in the console. 
            Action<string, VBuffer<uint>[]> printHelper = (columnName, column) =>
            {
                Console.WriteLine($"{columnName} column obtained post-transformation.");
                foreach (var row in column)
                {
                    foreach (var value in row.Values)
                        Console.Write($"{value} ");
                    Console.WriteLine("");
                }

                Console.WriteLine("===================================================");
            };

            // Preview of the TextFeatures column obtained after processing the input.
             var defaultColumn = transformedData_default.GetColumn<VBuffer<uint>>(ml, defaultColumnName).ToArray();
             printHelper(defaultColumnName, defaultColumn);

            // DefaultKeys column obtained post-transformation
            // 8 9 3 1
            // 8 9 3 1
            // 8 9 3 1
            // 8 9 3 1

            // Preview of the TextFeatures column obtained after processing the input.
            var customizedColumn = transformedData_customized.GetColumn<VBuffer<uint>>(ml, customizedColumnName).ToArray();
            printHelper(customizedColumnName, customizedColumn);

            // CustomizedKeys column obtained post-transformation.
            // 0 1 3 2
            // 0 1 3 2
            // 0 1 3 2
            // 0 1 3 2

            // retrieve the original values, by appending the KeyToValue etimator to the existing pipelines
            var pipeline = default_pipeline.Append(new KeyToValueEstimator(ml, defaultColumnName));

            // The transformed data.
            transformedData_default = pipeline.Fit(trainData).Transform(trainData);

            // Preview of the TextFeatures column obtained after processing the input.
            var originalColumnBack = transformedData_default.GetColumn<VBuffer<ReadOnlyMemory<char>>>(ml, defaultColumnName).ToArray();

            foreach (var row in originalColumnBack)
            {
                foreach (var value in row.Values)
                    Console.Write($"{value} ");
                Console.WriteLine("");
            }

            // car truck universe radiation
            // car truck universe radiation
            // car truck universe radiation
            // car truck universe radiation
        }
    }
}
