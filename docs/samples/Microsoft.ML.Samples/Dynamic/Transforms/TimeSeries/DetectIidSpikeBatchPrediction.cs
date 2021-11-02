using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class DetectIidSpikeBatchPrediction
    {
        // This example creates a time series (list of Data with the i-th element
        // corresponding to the i-th time slot). The estimator is applied then to
        // identify spiking points in the series.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a spike
            const int Size = 10;
            var data = new List<TimeSeriesData>(Size + 1)
            {
                new TimeSeriesData(5),
                new TimeSeriesData(5),
                new TimeSeriesData(5),
                new TimeSeriesData(5),
                new TimeSeriesData(5),

                // This is a spike.
                new TimeSeriesData(10),

                new TimeSeriesData(5),
                new TimeSeriesData(5),
                new TimeSeriesData(5),
                new TimeSeriesData(5),
                new TimeSeriesData(5),
            };

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup the estimator arguments
            string outputColumnName = nameof(IidSpikePrediction.Prediction);
            string inputColumnName = nameof(TimeSeriesData.Value);

            // The transformed data.
            var transformedData = ml.Transforms.DetectIidSpike(outputColumnName,
                inputColumnName, 95.0d, Size / 4).Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of
            // IidSpikePrediction.
            var predictionColumn = ml.Data.CreateEnumerable<IidSpikePrediction>(
                transformedData, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained " +
                $"post-transformation.");

            Console.WriteLine("Data\tAlert\tScore\tP-Value");

            int k = 0;
            foreach (var prediction in predictionColumn)
                PrintPrediction(data[k++].Value, prediction);

            // Prediction column obtained post-transformation.
            // Data    Alert   Score P-Value
            // 5       0       5.00    0.50
            // 5       0       5.00    0.50
            // 5       0       5.00    0.50
            // 5       0       5.00    0.50
            // 5       0       5.00    0.50
            // 10      1       10.00   0.00   <-- alert is on, predicted spike
            // 5       0       5.00    0.26
            // 5       0       5.00    0.26
            // 5       0       5.00    0.50
            // 5       0       5.00    0.50
            // 5       0       5.00    0.50
        }

        private static void PrintPrediction(float value, IidSpikePrediction
            prediction) =>
            Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", value,
            prediction.Prediction[0], prediction.Prediction[1],
            prediction.Prediction[2]);

        class TimeSeriesData
        {
            public float Value;

            public TimeSeriesData(float value)
            {
                Value = value;
            }
        }

        class IidSpikePrediction
        {
            [VectorType(3)]
            public double[] Prediction { get; set; }
        }
    }
}
