using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;

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


            // Setup the detection arguments
            string outputColumnName = nameof(SrCnnAnomalyDetection.Prediction);
            string inputColumnName = nameof(TimeSeriesData.Value);

            // Do batch anomaly detection
            var outputDataView = ml.Data.BatchDetectAnomalyBySrCnn(dataView, outputColumnName, inputColumnName, batchSize: 512, sensitivity: 70, detectMode: SrCnnDetectMode.AnomalyAndMargin);

            // Getting the data of the newly created column as an IEnumerable of
            // SrCnnAnomalyDetection.
            var predictionColumn = ml.Data.CreateEnumerable<SrCnnAnomalyDetection>(
                outputDataView, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained post-" +
                $"transformation.");

            Console.WriteLine("Data\tAlert\tScore\tMag\tExpectedValue\tBoundaryUnit\tUpperBoundary\tLowerBoundary");

            int k = 0;
            foreach (var prediction in predictionColumn)
                PrintPrediction(data[k++].Value, prediction);

            //Prediction column obtained post-transformation.
            //Data Alert   Score Mag ExpectedValue BoundaryUnit UpperBoundary LowerBoundary
            // TODO: update with actual output from SrCnn
        }

        private static void PrintPrediction(double value, SrCnnAnomalyDetection
            prediction) =>
            Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t{4:0.00}\t{5:0.00}\t{6:0.00}", value,
                prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2],
                prediction.Prediction[3], prediction.Prediction[4], prediction.Prediction[5]);

        private class TimeSeriesData
        {
            public double Value;

            public TimeSeriesData(double value)
            {
                Value = value;
            }
        }

        private class SrCnnAnomalyDetection
        {
            [VectorType(6)]
            public double[] Prediction { get; set; }
        }
    }
}
