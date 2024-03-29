﻿using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class DetectChangePointBySsaBatchPrediction
    {
        // This example creates a time series (list of Data with the i-th element
        // corresponding to the i-th time slot). The estimator is applied then to
        // identify points where data distribution changed. This estimator can
        // account for temporal seasonality in the data.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a recurring pattern and then a
            // change in trend
            const int SeasonalitySize = 5;
            const int TrainingSeasons = 3;
            const int TrainingSize = SeasonalitySize * TrainingSeasons;
            var data = new List<TimeSeriesData>()
            {
                new TimeSeriesData(0),
                new TimeSeriesData(1),
                new TimeSeriesData(2),
                new TimeSeriesData(3),
                new TimeSeriesData(4),

                new TimeSeriesData(0),
                new TimeSeriesData(1),
                new TimeSeriesData(2),
                new TimeSeriesData(3),
                new TimeSeriesData(4),

                new TimeSeriesData(0),
                new TimeSeriesData(1),
                new TimeSeriesData(2),
                new TimeSeriesData(3),
                new TimeSeriesData(4),

                //This is a change point
                new TimeSeriesData(0),
                new TimeSeriesData(100),
                new TimeSeriesData(200),
                new TimeSeriesData(300),
                new TimeSeriesData(400),
            };

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup estimator arguments
            var inputColumnName = nameof(TimeSeriesData.Value);
            var outputColumnName = nameof(ChangePointPrediction.Prediction);

            // The transformed data.
            var transformedData = ml.Transforms.DetectChangePointBySsa(
                outputColumnName, inputColumnName, 95.0d, 8, TrainingSize,
                SeasonalitySize + 1).Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of
            // ChangePointPrediction.
            var predictionColumn = ml.Data.CreateEnumerable<ChangePointPrediction>(
                transformedData, reuseRowObject: false);

            Console.WriteLine(outputColumnName + " column obtained " +
                "post-transformation.");

            Console.WriteLine("Data\tAlert\tScore\tP-Value\tMartingale value");
            int k = 0;
            foreach (var prediction in predictionColumn)
                PrintPrediction(data[k++].Value, prediction);

            // Prediction column obtained post-transformation.
            // Data    Alert   Score   P-Value Martingale value
            // 0       0      -2.53    0.50    0.00
            // 1       0      -0.01    0.01    0.00
            // 2       0       0.76    0.14    0.00
            // 3       0       0.69    0.28    0.00
            // 4       0       1.44    0.18    0.00
            // 0       0      -1.84    0.17    0.00
            // 1       0       0.22    0.44    0.00
            // 2       0       0.20    0.45    0.00
            // 3       0       0.16    0.47    0.00
            // 4       0       1.33    0.18    0.00
            // 0       0      -1.79    0.07    0.00
            // 1       0       0.16    0.50    0.00
            // 2       0       0.09    0.50    0.00
            // 3       0       0.08    0.45    0.00
            // 4       0       1.31    0.12    0.00
            // 0       0      -1.79    0.07    0.00
            // 100     1      99.16    0.00    4031.94     <-- alert is on, predicted changepoint
            // 200     0     185.23    0.00    731260.87
            // 300     0     270.40    0.01    3578470.47
            // 400     0     357.11    0.03    45298370.86
        }

        private static void PrintPrediction(float value, ChangePointPrediction
            prediction) =>
            Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t{4:0.00}", value,
            prediction.Prediction[0], prediction.Prediction[1],
            prediction.Prediction[2], prediction.Prediction[3]);

        class ChangePointPrediction
        {
            [VectorType(4)]
            public double[] Prediction { get; set; }
        }

        class TimeSeriesData
        {
            public float Value;

            public TimeSeriesData(float value)
            {
                Value = value;
            }
        }
    }
}
