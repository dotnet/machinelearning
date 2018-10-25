// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

        // the alignment of the usings with the methods is intentional so they can display on the same level in the docs site.
        using Microsoft.ML.Data;
        using Microsoft.ML.Runtime.Api;
        using Microsoft.ML.Runtime.Data;
        using Microsoft.ML.Transforms.Text;
        using System;
        using System.Collections.Generic;
        using System.Linq;

namespace Microsoft.ML.Samples.Dynamic
{
    public partial class TransformSamples
    {
        public static void KeyToValue_Term()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext(seed: 1, conc: 1);

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleTopicsData> data = SamplesUtils.DatasetUtils.GetTopicsData();
            var trainData = ml.CreateStreamingDataView(data);

            // Preview of the topics data; a dataset that contains two columns containing keys independently assigned to a body of text, 
            // Review and ReviewReverse. The Label colum indicates whether the set of keys in the ReviewReverse match the ones in the review column.
            // The dataset will be used to classify how accurately the keys are assigned to the text. 
            //
            // Review,                                    ReviewReverse,                 Label
            // "animals birds cats dogs fish horse", "radiation galaxy universe duck",     1
            // "horse birds house fish duck cats",   "space galaxy universe radiation",    0
            // "car truck driver bus pickup",        "bus pickup",                         1
            // "car truck driver bus pickup horse",  "car truck",                          0

            // A pipeline to convert the terms of the review_reverse column in 
            // making use of default settings.
            string defaultColumnName = "DefaultKeys";
            // REVIEW create through the catalog extension
            var default_pipeline = new WordTokenizeEstimator(ml, "ReviewReverse")
                .Append(new TermEstimator(ml, "ReviewReverse" , defaultColumnName));

            // Another pipeline, that customizes the advanced settings of the TermEstimator.
            // We can change the maxNumTerm to limit how many keys will get generated out of the set of words, 
            // and condition the order in which they get evaluated by changing sort from the default Occurence (order in which they get encountered) 
            // to value/alphabetically.
            string customizedColumnName = "CustomizedKeys";
            var customized_pipeline = new WordTokenizeEstimator(ml, "ReviewReverse", "ReviewReverse")
                .Append(new TermEstimator(ml, "ReviewReverse", customizedColumnName, maxNumTerms: 10, sort:TermTransform.SortOrder.Value));

            // The transformed data.
            var transformedData_default = default_pipeline.Fit(trainData).Transform(trainData);
            var transformedData_customized = customized_pipeline.Fit(trainData).Transform(trainData);

            // Small helper to print the text inside the columns, in the console. 
            Action<string, IEnumerable<VBuffer<uint>>> printHelper = (columnName, column) =>
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

            // Preview of the DefaultKeys column obtained after processing the input.
            var defaultColumn = transformedData_default.GetColumn<VBuffer<uint>>(ml, defaultColumnName);
            printHelper(defaultColumnName, defaultColumn);

            // DefaultKeys column obtained post-transformation.
            //
            // 1 2 3 4
            // 5 2 3 1
            // 6 7 3 1
            // 8 9 3 1

            // Previewing the CustomizedKeys column obtained after processing the input.
            var customizedColumn = transformedData_customized.GetColumn<VBuffer<uint>>(ml, customizedColumnName);
            printHelper(customizedColumnName, customizedColumn);

            // CustomizedKeys column obtained post-transformation.
            //
            // 6 4 9 3
            // 7 4 9 6
            // 1 5 9 6
            // 2 8 9 6

            // Retrieve the original values, by appending the KeyToValue etimator to the existing pipelines
            // to convert the keys back to the strings.
            var pipeline = default_pipeline.Append(new KeyToValueEstimator(ml, defaultColumnName));
            transformedData_default = pipeline.Fit(trainData).Transform(trainData);

            // Preview of the DefaultColumnName column obtained.
            var originalColumnBack = transformedData_default.GetColumn<VBuffer<ReadOnlyMemory<char>>>(ml, defaultColumnName);

            foreach (var row in originalColumnBack)
            {
                foreach (var value in row.Values)
                    Console.Write($"{value} ");
                Console.WriteLine("");
            }

            // DefaultKeys column obtained post-transformation.
            //
            // radiation galaxy universe duck
            // space galaxy universe radiation
            // bus pickup universe radiation
            // car truck universe radiation
        }
    }
}
