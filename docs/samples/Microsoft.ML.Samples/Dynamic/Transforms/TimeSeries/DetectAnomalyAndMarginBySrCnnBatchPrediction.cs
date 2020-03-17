using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic.Transforms.TimeSeries
{
    class DetectAnomalyAndMarginBySrCnnBatchPrediction
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with an anomaly
            var data = new List<TimeSeriesData>();
            for (int index = 0; index < 20; index++)
            {
                data.Add(new TimeSeriesData(5));
            }
            data.Add(new TimeSeriesData(10));
            for (int index = 0; index < 5; index++)
            {
                data.Add(new TimeSeriesData(5));
            }

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup the estimator arguments
            string outputColumnName = nameof(SrCnnAnomalyDetection.Prediction);
            string inputColumnName = nameof(TimeSeriesData.Value);

            // The transformed data.
            var transformedData = ml.Transforms.DetectAnomalyBySrCnn(
                outputColumnName, inputColumnName, 16, 5, 5, 3, 8, 0.35, SrCnnDetectMode.AnomalyAndMargin, 90.0).Fit(
                dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of
            // SrCnnAnomalyDetection.
            var predictionColumn = ml.Data.CreateEnumerable<SrCnnAnomalyDetection>(
                transformedData, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained post-" +
                $"transformation.");

            Console.WriteLine("Data\tAlert\tAnomalyScore\tMag\tExpectedValue\tBoundaryUnit\tUpperBoundary\tLowerBoundary");

            int k = 0;
            foreach (var prediction in predictionColumn)
                PrintPrediction(data[k++].Value, prediction);

            //Prediction column obtained post-transformation.
            // Data	Alert	Score	            Mag ExpectedValue   BoundaryUnit    UpperBoundary   LowerBoundary
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.18    5.00            5.00            5.01            4.99
            //5       0       0.00            0.18    5.00            5.00            5.01            4.99
            //5       0       0.00            0.18    5.00            5.00            5.01            4.99
            //5       0       0.00            0.18    5.00            5.00            5.01            4.99
            //5       0       0.00            0.18    5.00            5.00            5.01            4.99
            //10      1       0.38            0.93    9.06            5.00            9.07            9.05
            //5       0       0.00            0.50    5.89            5.00            5.90            5.88
            //5       0       0.00            0.30    4.25            5.00            4.26            4.24
            //5       0       0.00            0.23    5.55            5.00            5.56            5.54
            //5       0       0.00            0.21    4.69            5.00            4.70            4.68
            //5       0       0.00            0.25    5.07            5.00            5.08            5.06
        }

        private static void PrintPrediction(float value, SrCnnAnomalyDetection
            prediction) =>
            Console.WriteLine("{0}\t{1}\t{2:0.00}\t\t{3:0.00}\t{4:0.00}\t\t{5:0.00}\t\t{6:0.00}\t\t{7:0.00}",
                value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2],
                prediction.Prediction[3], prediction.Prediction[4], prediction.Prediction[5], prediction.Prediction[6]);

        private class TimeSeriesData
        {
            public float Value;

            public TimeSeriesData(float value)
            {
                Value = value;
            }
        }

        private class SrCnnAnomalyDetection
        {
            [VectorType(7)]
            public double[] Prediction { get; set; }
        }
    }
}
