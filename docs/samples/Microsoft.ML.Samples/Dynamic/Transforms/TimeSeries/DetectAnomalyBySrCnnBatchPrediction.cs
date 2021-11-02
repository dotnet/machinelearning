using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class DetectAnomalyBySrCnnBatchPrediction
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
                outputColumnName, inputColumnName, 16, 5, 5, 3, 8, 0.35).Fit(
                dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of
            // SrCnnAnomalyDetection.
            var predictionColumn = ml.Data.CreateEnumerable<SrCnnAnomalyDetection>(
                transformedData, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained post-" +
                $"transformation.");

            Console.WriteLine("Data\tAlert\tScore\tMag");

            int k = 0;
            foreach (var prediction in predictionColumn)
                PrintPrediction(data[k++].Value, prediction);

            //Prediction column obtained post-transformation.
            //Data Alert   Score Mag
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.00    0.00
            //5       0       0.03    0.18
            //5       0       0.03    0.18
            //5       0       0.03    0.18
            //5       0       0.03    0.18
            //5       0       0.03    0.18
            //10      1       0.47    0.93
            //5       0       0.31    0.50
            //5       0       0.05    0.30
            //5       0       0.01    0.23
            //5       0       0.00    0.21
            //5       0       0.01    0.25
        }

        private static void PrintPrediction(float value, SrCnnAnomalyDetection
            prediction) =>
            Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", value, prediction
            .Prediction[0], prediction.Prediction[1], prediction.Prediction[2]);

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
            [VectorType(3)]
            public double[] Prediction { get; set; }
        }
    }
}
