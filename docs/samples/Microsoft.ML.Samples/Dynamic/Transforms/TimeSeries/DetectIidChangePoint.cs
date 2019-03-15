// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class DetectIidChangePoint
    {
        class ChangePointPrediction
        {
            [VectorType(4)]
            public double[] Prediction { get; set; }
        }

        class IidChangePointData
        {
            public float Value;

            public IidChangePointData(float value)
            {
                Value = value;
            }
        }

        // This example creates a time series (list of Data with the i-th element corresponding to the i-th time slot). 
        // The estimator is applied then to identify points where data distribution changed.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a change
            const int Size = 16;
            var data = new List<IidChangePointData>(Size);
            for (int i = 0; i < Size / 2; i++)
                data.Add(new IidChangePointData(5));
            // This is a change point
            for (int i = 0; i < Size / 2; i++)
                data.Add(new IidChangePointData(7));

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup estimator arguments
            string outputColumnName = nameof(ChangePointPrediction.Prediction);
            string inputColumnName = nameof(IidChangePointData.Value);

            // The transformed data.
            var transformedData = ml.Transforms.DetectIidChangePoint(outputColumnName, inputColumnName, 95, Size / 4).Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of ChangePointPrediction.
            var predictionColumn = ml.Data.CreateEnumerable<ChangePointPrediction>(transformedData, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained post-transformation.");
            Console.WriteLine("Data\tAlert\tScore\tP-Value\tMartingale value");
            int k = 0;
            foreach (var prediction in predictionColumn)
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t{4:0.00}", data[k++].Value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2], prediction.Prediction[3]);
            Console.WriteLine("");

            // Prediction column obtained post-transformation.
            // Data Alert      Score   P-Value Martingale value
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 7       1       7.00    0.00    10298.67   <-- alert is on, predicted changepoint
            // 7       0       7.00    0.13    33950.16
            // 7       0       7.00    0.26    60866.34
            // 7       0       7.00    0.38    78362.04
            // 7       0       7.00    0.50    0.01
            // 7       0       7.00    0.50    0.00
            // 7       0       7.00    0.50    0.00
            // 7       0       7.00    0.50    0.00
        }
    }
}
