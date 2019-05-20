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
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a spike
            var data = new List<TimeSeriesData>();
            for (int index = 0; index < 100; index++)
            {
                data.Add(new TimeSeriesData(5));
            }
            for (int index = 0; index < 5; index++)
            {
                data.Add(new TimeSeriesData(15));
            }
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
            var transformedData = ml.Transforms.DetectAnomalyBySrCnn(outputColumnName, inputColumnName, 64, 5, 5, 3, 21, 0.25).Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of SrCnnAnomalyDetection.
            var predictionColumn = ml.Data.CreateEnumerable<SrCnnAnomalyDetection>(transformedData, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained post-transformation.");
            Console.WriteLine("Data\tAlert\tScore\tP-Value");

            int k = 0;
            foreach (var prediction in predictionColumn)
                PrintPrediction(data[k++].Value, prediction);

        }

        private static void PrintPrediction(float value, SrCnnAnomalyDetection prediction) =>
            Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", value, prediction.Prediction[0],
                prediction.Prediction[1], prediction.Prediction[2]);

        class TimeSeriesData
        {
            public float Value;

            public TimeSeriesData(float value)
            {
                Value = value;
            }
        }

        class SrCnnAnomalyDetection
        {
            [VectorType(3)]
            public double[] Prediction { get; set; }
        }
    }
}
