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
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleTopicsData> data = SamplesUtils.DatasetUtils.GetTopicsData();
            var trainData = ml.CreateStreamingDataView(data);

            // Preview of one of the columns of the the topics data. 
            // The Review column contains the keys associated with a particular body of text.  
            //
            // Review                               
            // "animals birds cats dogs fish horse" 
            // "horse birds house fish duck cats"   
            // "car truck driver bus pickup"       
            // "car truck driver bus pickup horse"

            // A pipeline to convert the terms of the 'Review' column in 
            // making use of default settings.
            string defaultColumnName = "DefaultKeys";
            // REVIEW create through the catalog extension
            var default_pipeline = new WordTokenizeEstimator(ml, "Review")
                .Append(new TermEstimator(ml, "Review", defaultColumnName));

            // Another pipeline, that customizes the advanced settings of the TermEstimator.
            // We can change the maxNumTerm to limit how many keys will get generated out of the set of words, 
            // and condition the order in which they get evaluated by changing sort from the default Occurence (order in which they get encountered) 
            // to value/alphabetically.
            string customizedColumnName = "CustomizedKeys";
            var customized_pipeline = new WordTokenizeEstimator(ml, "Review")
                .Append(new TermEstimator(ml, "Review", customizedColumnName, maxNumTerms: 10, sort:TermTransform.SortOrder.Value));

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
            // 1 2 3 4 5 6
            // 6 2 7 5 8 3
            // 9 10 11 12 13 3
            // 9 10 11 12 13 6

            // Previewing the CustomizedKeys column obtained after processing the input.
            var customizedColumn = transformedData_customized.GetColumn<VBuffer<uint>>(ml, customizedColumnName);
            printHelper(customizedColumnName, customizedColumn);

            // CustomizedKeys column obtained post-transformation.
            //
            // 1 2 4 5 7 8
            // 8 2 9 7 6 4
            // 3 10 0 0 0 4
            // 3 10 0 0 0 8

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
            // animals birds cats dogs fish horse
            // horse birds house fish duck cats
            // car truck driver bus pickup cats
            // car truck driver bus pickup horse
        }
    }
}
