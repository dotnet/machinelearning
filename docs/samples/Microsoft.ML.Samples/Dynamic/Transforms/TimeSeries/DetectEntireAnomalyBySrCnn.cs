// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic
{
    class DetectEntireAnomalyBySrCnn
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with an anomaly
            DateTime currentTime = new DateTime(2020, 01, 01);
            var data = new List<TimeSeriesData>();
            for (int index = 0; index < 20; index++)
            {
                currentTime = currentTime.AddDays(1);
                data.Add(new TimeSeriesData { Timestamp = currentTime, Value = 5 });
                
            }
            currentTime = currentTime.AddDays(1);
            data.Add(new TimeSeriesData { Timestamp = currentTime, Value = 10});
            for (int index = 0; index < 5; index++)
            {
                currentTime = currentTime.AddDays(1);
                data.Add(new TimeSeriesData { Timestamp = currentTime, Value = 5 });
            }

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup the estimator arguments
            string outputColumnName = nameof(AnomalyDetectionResult.Prediction);
            string timestampColumnName = nameof(TimeSeriesData.Timestamp);
            string valueColumnName = nameof(TimeSeriesData.Value);

            // The transformed data.
            var transformedData = ml.Transforms.DetectEntireAnomalyBySrCnn(outputColumnName, timestampColumnName, valueColumnName, 0.35, 512, SrCnnDetectMode.AnomalyAndMargin, 90.0)
                .Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of SrCnnAnomalyDetection.
            var predictionColumn = ml.Data.CreateEnumerable<AnomalyDetectionResult>(transformedData, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained post-transformation.");
            Console.WriteLine("Timestamp\t\tData\tAnomaly\tAnomalyScore\tMag\tExpectedValue\tBoundaryUnit\tUpperBoundary\tLowerBoundary");

            int k = 0;
            foreach (var prediction in predictionColumn)
            {
                PrintPrediction(data[k].Timestamp, data[k].Value, prediction);
                k++;
            }
            //Prediction column obtained post-transformation.
            //Timestamp Data    Anomaly AnomalyScore    Mag ExpectedValue   BoundaryUnit UpperBoundary   LowerBoundary
            //2020 / 1 / 2 0:00:00        5.00    0               0.00    0.21            5.00            5.00            5.01            4.99
            //2020 / 1 / 3 0:00:00        5.00    0               0.00    0.11            5.00            5.00            5.01            4.99
            //2020 / 1 / 4 0:00:00        5.00    0               0.00    0.03            5.00            5.00            5.01            4.99
            //2020 / 1 / 5 0:00:00        5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //2020 / 1 / 6 0:00:00        5.00    0               0.00    0.03            5.00            5.00            5.01            4.99
            //2020 / 1 / 7 0:00:00        5.00    0               0.00    0.06            5.00            5.00            5.01            4.99
            //2020 / 1 / 8 0:00:00        5.00    0               0.00    0.02            5.00            5.00            5.01            4.99
            //2020 / 1 / 9 0:00:00        5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //2020 / 1 / 10 0:00:00       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //2020 / 1 / 11 0:00:00       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //2020 / 1 / 12 0:00:00       5.00    0               0.00    0.00            5.00            5.00            5.01            4.99
            //2020 / 1 / 13 0:00:00       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //2020 / 1 / 14 0:00:00       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //2020 / 1 / 15 0:00:00       5.00    0               0.00    0.02            5.00            5.00            5.01            4.99
            //2020 / 1 / 16 0:00:00       5.00    0               0.00    0.07            5.00            5.00            5.01            4.99
            //2020 / 1 / 17 0:00:00       5.00    0               0.00    0.08            5.00            5.00            5.01            4.99
            //2020 / 1 / 18 0:00:00       5.00    0               0.00    0.02            5.00            5.00            5.01            4.99
            //2020 / 1 / 19 0:00:00       5.00    0               0.00    0.05            5.00            5.00            5.01            4.99
            //2020 / 1 / 20 0:00:00       5.00    0               0.00    0.12            5.00            5.00            5.01            4.99
            //2020 / 1 / 21 0:00:00       5.00    0               0.00    0.17            5.00            5.00            5.01            4.99
            //2020 / 1 / 22 0:00:00       10.00   1               0.50    0.80            5.00            5.00            5.01            4.99
            //2020 / 1 / 23 0:00:00       5.00    0               0.00    0.16            5.00            5.00            5.01            4.99
            //2020 / 1 / 24 0:00:00       5.00    0               0.00    0.11            5.00            5.00            5.01            4.99
            //2020 / 1 / 25 0:00:00       5.00    0               0.00    0.05            5.00            5.00            5.01            4.99
            //2020 / 1 / 26 0:00:00       5.00    0               0.00    0.11            5.00            5.00            5.01            4.99
            //2020 / 1 / 27 0:00:00       5.00    0               0.00    0.19            5.00            5.00            5.01            4.99

            var modelPath = "temp.zip";
            var fitModel = ml.Transforms.DetectEntireAnomalyBySrCnn(outputColumnName, timestampColumnName, valueColumnName, 0.35, 512, SrCnnDetectMode.AnomalyAndMargin, 90.0).Fit(dataView);
            ml.Model.Save(fitModel, dataView.Schema, modelPath);

            using (var file = File.OpenRead(modelPath))
            {
                ITransformer loadedModel = ml.Model.Load(file, out var schema);

                transformedData = loadedModel.Transform(dataView);
                predictionColumn = ml.Data.CreateEnumerable<AnomalyDetectionResult>(transformedData, reuseRowObject: false);

                Console.WriteLine($"{outputColumnName} column obtained post-transformation by saved model.");
                Console.WriteLine("Timestamp\t\tData\tAnomaly\tAnomalyScore\tMag\tExpectedValue\tBoundaryUnit\tUpperBoundary\tLowerBoundary");

                k = 0;
                foreach (var prediction in predictionColumn)
                {
                    PrintPrediction(data[k].Timestamp, data[k].Value, prediction);
                    k++;
                }

                //Prediction column obtained post-transformation by saved model.
                //Timestamp Data    Anomaly AnomalyScore    Mag ExpectedValue   BoundaryUnit UpperBoundary   LowerBoundary
                //2020 / 1 / 2 0:00:00        5.00    0               0.00    0.21            5.00            5.00            5.01            4.99
                //2020 / 1 / 3 0:00:00        5.00    0               0.00    0.11            5.00            5.00            5.01            4.99
                //2020 / 1 / 4 0:00:00        5.00    0               0.00    0.03            5.00            5.00            5.01            4.99
                //2020 / 1 / 5 0:00:00        5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
                //2020 / 1 / 6 0:00:00        5.00    0               0.00    0.03            5.00            5.00            5.01            4.99
                //2020 / 1 / 7 0:00:00        5.00    0               0.00    0.06            5.00            5.00            5.01            4.99
                //2020 / 1 / 8 0:00:00        5.00    0               0.00    0.02            5.00            5.00            5.01            4.99
                //2020 / 1 / 9 0:00:00        5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
                //2020 / 1 / 10 0:00:00       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
                //2020 / 1 / 11 0:00:00       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
                //2020 / 1 / 12 0:00:00       5.00    0               0.00    0.00            5.00            5.00            5.01            4.99
                //2020 / 1 / 13 0:00:00       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
                //2020 / 1 / 14 0:00:00       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
                //2020 / 1 / 15 0:00:00       5.00    0               0.00    0.02            5.00            5.00            5.01            4.99
                //2020 / 1 / 16 0:00:00       5.00    0               0.00    0.07            5.00            5.00            5.01            4.99
                //2020 / 1 / 17 0:00:00       5.00    0               0.00    0.08            5.00            5.00            5.01            4.99
                //2020 / 1 / 18 0:00:00       5.00    0               0.00    0.02            5.00            5.00            5.01            4.99
                //2020 / 1 / 19 0:00:00       5.00    0               0.00    0.05            5.00            5.00            5.01            4.99
                //2020 / 1 / 20 0:00:00       5.00    0               0.00    0.12            5.00            5.00            5.01            4.99
                //2020 / 1 / 21 0:00:00       5.00    0               0.00    0.17            5.00            5.00            5.01            4.99
                //2020 / 1 / 22 0:00:00       10.00   1               0.50    0.80            5.00            5.00            5.01            4.99
                //2020 / 1 / 23 0:00:00       5.00    0               0.00    0.16            5.00            5.00            5.01            4.99
                //2020 / 1 / 24 0:00:00       5.00    0               0.00    0.11            5.00            5.00            5.01            4.99
                //2020 / 1 / 25 0:00:00       5.00    0               0.00    0.05            5.00            5.00            5.01            4.99
                //2020 / 1 / 26 0:00:00       5.00    0               0.00    0.11            5.00            5.00            5.01            4.99
                //2020 / 1 / 27 0:00:00       5.00    0               0.00    0.19            5.00            5.00            5.01            4.99
            }
        }

        private static void PrintPrediction(DateTime timestamp, Double value, AnomalyDetectionResult prediction) =>
            Console.WriteLine("{0}\t{1:0.00}\t{2}\t\t{3:0.00}\t{4:0.00}\t\t{5:0.00}\t\t{6:0.00}\t\t{7:0.00}\t\t{8:0.00}",
                timestamp, value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2],
                prediction.Prediction[3], prediction.Prediction[4], prediction.Prediction[5], prediction.Prediction[6]);

        private class TimeSeriesData
        {
            public DateTime Timestamp { get; set; }
            public Double Value { get; set; }
        }

        public class AnomalyDetectionResult
        {
            [VectorType]
            public double[] Prediction { get; set; }
        }

    }
}
