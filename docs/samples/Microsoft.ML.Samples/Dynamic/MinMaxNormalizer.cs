// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

        // the alignment of the usings with the methods is intentional so they can display on the same level in the docs site.
        using Microsoft.ML.Runtime.Data;
        using Microsoft.ML.Runtime.Api;
        using Microsoft.ML.Data;
        using System;
        using System.Collections.Generic;

namespace Microsoft.ML.Samples.Dynamic
{
    public partial class TransformSamples
    {
        public static void MinMaxNormalizer()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = ml.CreateStreamingDataView(data);

            // Preview of the data.
            //
            // Age  Case  Education  Induced     Parity  PooledStratum  RowNum  ...
            // 26   1       0-5yrs      1         6         3             1  ...
            // 42   1       0-5yrs      1         1         1             2  ...
            // 39   1       0-5yrs      2         6         4             3  ...
            // 34   1       0-5yrs      2         4         2             4  ...
            // 35   1       6-11yrs     1         3         32            5  ...

            // A pipeline for normalizing the Induced column.
            var pipeline = ml.Transforms.Normalizer("Induced");
            // The transformed (normalized according to Normalizer.NormalizerMode.MinMax) data.
            var transformedData = pipeline.Fit(trainData).Transform(trainData);
            // Getting the data of the newly created column, so we can preview it.
            var normalizedColumn = transformedData.GetColumn<float>(ml, "Induced");

            // A small printing utility.
            Action<string, IEnumerable<float>> printHelper = (colName, column) =>
            {
                Console.WriteLine($"{colName} column obtained post-transformation.");
                foreach (var row in column)
                    Console.WriteLine($"{row} ");
            };

            printHelper("Induced", normalizedColumn);

            // Induced column obtained post-transformation.
            //
            // 0.5
            // 0.5
            // 1
            // 1
            // 0.5

            // Composing a different pipeline if we wanted to normalize more than one column at a time. 
            // Using log scale as the normalization mode. 
            var multiColPipeline = ml.Transforms.Normalizer(Normalizer.NormalizerMode.LogMeanVariance, new[] { ("Induced", "LogInduced"), ("Spontaneous", "LogSpontaneous") });
            // The transformed data.
            var multiColtransformedData = multiColPipeline.Fit(trainData).Transform(trainData);

            // Getting the newly created columns. 
            var normalizedInduced = multiColtransformedData.GetColumn<float>(ml, "LogInduced");
            var normalizedSpont = multiColtransformedData.GetColumn<float>(ml, "LogSpontaneous");

            printHelper("LogInduced", normalizedInduced);

            // LogInduced column obtained post-transformation.
            //
            // 0.2071445
            // 0.2071445
            // 0.889631
            // 0.889631
            // 0.2071445

            printHelper("LogSpontaneous", normalizedSpont);

            // LogSpontaneous column obtained post-transformation.
            //
            // 0.8413026
            // 0
            // 0
            // 0
            // 0.1586974
        }
    }
}
