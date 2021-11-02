using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class DetectSpikeBySsaBatchPrediction
    {
        // This example creates a time series (list of Data with the i-th element
        // corresponding to the i-th time slot). The estimator is applied then to
        // identify spiking points in the series. This estimator can account for
        // temporal seasonality in the data.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a recurring pattern and a spike
            // within the pattern
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

                //This is a spike.
                new TimeSeriesData(100),

                new TimeSeriesData(0),
                new TimeSeriesData(1),
                new TimeSeriesData(2),
                new TimeSeriesData(3),
                new TimeSeriesData(4),
            };

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup estimator arguments
            var inputColumnName = nameof(TimeSeriesData.Value);
            var outputColumnName = nameof(SsaSpikePrediction.Prediction);

            // The transformed data.
            var transformedData = ml.Transforms.DetectSpikeBySsa(outputColumnName,
                inputColumnName, 95.0d, 8, TrainingSize, SeasonalitySize + 1).Fit(
                dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of
            // SsaSpikePrediction.
            var predictionColumn = ml.Data.CreateEnumerable<SsaSpikePrediction>(
                transformedData, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained " +
                $"post-transformation.");

            Console.WriteLine("Data\tAlert\tScore\tP-Value");
            int k = 0;
            foreach (var prediction in predictionColumn)
                PrintPrediction(data[k++].Value, prediction);

            // Prediction column obtained post-transformation.
            // Data    Alert   Score   P-Value
            // 0       0      -2.53    0.50
            // 1       0      -0.01    0.01
            // 2       0       0.76    0.14
            // 3       0       0.69    0.28
            // 4       0       1.44    0.18
            // 0       0      -1.84    0.17
            // 1       0       0.22    0.44
            // 2       0       0.20    0.45
            // 3       0       0.16    0.47
            // 4       0       1.33    0.18
            // 0       0      -1.79    0.07
            // 1       0       0.16    0.50
            // 2       0       0.09    0.50
            // 3       0       0.08    0.45
            // 4       0       1.31    0.12
            // 100     1      98.21    0.00   <-- alert is on, predicted spike
            // 0       0     -13.83    0.29
            // 1       0      -1.74    0.44
            // 2       0      -0.47    0.46
            // 3       0     -16.50    0.29
            // 4       0     -29.82    0.21
        }

        private static void PrintPrediction(float value, SsaSpikePrediction
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

        class SsaSpikePrediction
        {
            [VectorType(3)]
            public double[] Prediction { get; set; }
        }
    }
}
