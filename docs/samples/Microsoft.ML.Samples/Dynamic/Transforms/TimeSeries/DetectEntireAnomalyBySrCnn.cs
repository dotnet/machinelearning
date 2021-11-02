using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;

namespace Samples.Dynamic
{
    public static class DetectEntireAnomalyBySrCnn
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
                data.Add(new TimeSeriesData { Value = 5 });
            }
            data.Add(new TimeSeriesData { Value = 10 });
            for (int index = 0; index < 5; index++)
            {
                data.Add(new TimeSeriesData { Value = 5 });
            }

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup the detection arguments
            string outputColumnName = nameof(SrCnnAnomalyDetection.Prediction);
            string inputColumnName = nameof(TimeSeriesData.Value);

            // Do batch anomaly detection
            var outputDataView = ml.AnomalyDetection.DetectEntireAnomalyBySrCnn(dataView, outputColumnName, inputColumnName,
                threshold: 0.35, batchSize: 512, sensitivity: 90.0, detectMode: SrCnnDetectMode.AnomalyAndMargin);

            // Getting the data of the newly created column as an IEnumerable of
            // SrCnnAnomalyDetection.
            var predictionColumn = ml.Data.CreateEnumerable<SrCnnAnomalyDetection>(
                outputDataView, reuseRowObject: false);

            Console.WriteLine("Index\tData\tAnomaly\tAnomalyScore\tMag\tExpectedValue\tBoundaryUnit\tUpperBoundary\tLowerBoundary");

            int k = 0;
            foreach (var prediction in predictionColumn)
            {
                PrintPrediction(k, data[k].Value, prediction);
                k++;
            }
            //Index Data    Anomaly AnomalyScore    Mag ExpectedValue   BoundaryUnit UpperBoundary   LowerBoundary
            //0       5.00    0               0.00    0.21            5.00            5.00            5.01            4.99
            //1       5.00    0               0.00    0.11            5.00            5.00            5.01            4.99
            //2       5.00    0               0.00    0.03            5.00            5.00            5.01            4.99
            //3       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //4       5.00    0               0.00    0.03            5.00            5.00            5.01            4.99
            //5       5.00    0               0.00    0.06            5.00            5.00            5.01            4.99
            //6       5.00    0               0.00    0.02            5.00            5.00            5.01            4.99
            //7       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //8       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //9       5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //10      5.00    0               0.00    0.00            5.00            5.00            5.01            4.99
            //11      5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //12      5.00    0               0.00    0.01            5.00            5.00            5.01            4.99
            //13      5.00    0               0.00    0.02            5.00            5.00            5.01            4.99
            //14      5.00    0               0.00    0.07            5.00            5.00            5.01            4.99
            //15      5.00    0               0.00    0.08            5.00            5.00            5.01            4.99
            //16      5.00    0               0.00    0.02            5.00            5.00            5.01            4.99
            //17      5.00    0               0.00    0.05            5.00            5.00            5.01            4.99
            //18      5.00    0               0.00    0.12            5.00            5.00            5.01            4.99
            //19      5.00    0               0.00    0.17            5.00            5.00            5.01            4.99
            //20      10.00   1               0.50    0.80            5.00            5.00            5.01            4.99
            //21      5.00    0               0.00    0.16            5.00            5.00            5.01            4.99
            //22      5.00    0               0.00    0.11            5.00            5.00            5.01            4.99
            //23      5.00    0               0.00    0.05            5.00            5.00            5.01            4.99
            //24      5.00    0               0.00    0.11            5.00            5.00            5.01            4.99
            //25      5.00    0               0.00    0.19            5.00            5.00            5.01            4.99
        }

        private static void PrintPrediction(int idx, double value, SrCnnAnomalyDetection prediction) =>
            Console.WriteLine("{0}\t{1:0.00}\t{2}\t\t{3:0.00}\t{4:0.00}\t\t{5:0.00}\t\t{6:0.00}\t\t{7:0.00}\t\t{8:0.00}",
                idx, value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2],
                prediction.Prediction[3], prediction.Prediction[4], prediction.Prediction[5], prediction.Prediction[6]);

        private class TimeSeriesData
        {
            public double Value { get; set; }
        }

        private class SrCnnAnomalyDetection
        {
            [VectorType]
            public double[] Prediction { get; set; }
        }
    }
}
