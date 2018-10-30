// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

        // the alignment of the usings with the methods is intentional so they can display on the same level in the docs site.
        using Microsoft.ML.Runtime.Data;
        using Microsoft.ML.Runtime.Api;
        using System;
        using System.Linq;
        using System.Collections.Generic;
        using Microsoft.ML.Transforms;

namespace Microsoft.ML.Samples.Dynamic
{
    public partial class TransformSamples
    {
        class SampleInfertDataWithFeatures
        {
            public VBuffer<int> Features { get; set; }
        }

        public static void ConcatTransform()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = ml.CreateStreamingDataView(data);

            // Preview of the data.
            //
            // Age    Case  Education  induced     parity  pooled.stratum  row_num  ...
            // 26.0   1.0   0-5yrs      1.0         6.0       3.0      1.0  ...
            // 42.0   1.0   0-5yrs      1.0         1.0       1.0      2.0  ...
            // 39.0   1.0   0-5yrs      2.0         6.0       4.0      3.0  ...
            // 34.0   1.0   0-5yrs      2.0         4.0       2.0      4.0  ...
            // 35.0   1.0   6-11yrs     1.0         3.0       32.0     5.0  ...

            // A pipeline for concatenating the age, parity and induced columns together in the Features column.
            string outputColumnName = "Features";
            var pipeline = new ColumnConcatenatingEstimator(ml, outputColumnName, new[] { "Age", "Parity", "Induced"});

            // The transformed data.
            var transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the data of the newly created column as an IEnumerable of SampleInfertDataWithFeatures.
            var featuresColumn = transformedData.AsEnumerable<SampleInfertDataWithFeatures>(ml, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained post-transformation.");
            foreach (var featureRow in featuresColumn)
            {
                foreach (var value in featureRow.Features.Values)
                    Console.Write($"{value} ");
                Console.WriteLine("");
            }

            // Features column obtained post-transformation.
            //
            // 26 6 1
            // 42 1 1
            // 39 6 2
            // 34 4 2
            // 35 3 1
        }
    }
}
