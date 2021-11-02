using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic
{
    public static class DetectSpikeBySsa
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

            // Generate sample series data with a recurring pattern
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
            };

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup IidSpikeDetector arguments
            var inputColumnName = nameof(TimeSeriesData.Value);
            var outputColumnName = nameof(SsaSpikePrediction.Prediction);

            // Train the change point detector.
            ITransformer model = ml.Transforms.DetectSpikeBySsa(outputColumnName,
                inputColumnName, 95.0d, 8, TrainingSize, SeasonalitySize + 1).Fit(
                dataView);

            // Create a prediction engine from the model for feeding new data.
            var engine = model.CreateTimeSeriesEngine<TimeSeriesData,
                SsaSpikePrediction>(ml);

            // Start streaming new data points with no change point to the
            // prediction engine.
            Console.WriteLine($"Output from spike predictions on new data:");
            Console.WriteLine("Data\tAlert\tScore\tP-Value");

            // Output from spike predictions on new data:
            // Data    Alert   Score   P-Value

            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 5; i++)
                    PrintPrediction(i, engine.Predict(new TimeSeriesData(i)));

            // 0       0      -1.01    0.50
            // 1       0      -0.24    0.22
            // 2       0      -0.31    0.30
            // 3       0       0.44    0.01
            // 4       0       2.16    0.00
            // 0       0      -0.78    0.27
            // 1       0      -0.80    0.30
            // 2       0      -0.84    0.31
            // 3       0       0.33    0.31
            // 4       0       2.21    0.07

            // Now send a data point that reflects a spike.
            PrintPrediction(100, engine.Predict(new TimeSeriesData(100)));

            // 100     1      86.17    0.00   <-- alert is on, predicted spike

            // Now we demonstrate saving and loading the model.
            // Save the model that exists within the prediction engine.
            // The engine has been updating this model with every new data point.
            var modelPath = "model.zip";
            engine.CheckPoint(ml, modelPath);

            // Load the model.
            using (var file = File.OpenRead(modelPath))
                model = ml.Model.Load(file, out DataViewSchema schema);

            // We must create a new prediction engine from the persisted model.
            engine = model.CreateTimeSeriesEngine<TimeSeriesData,
                SsaSpikePrediction>(ml);

            // Run predictions on the loaded model.
            for (int i = 0; i < 5; i++)
                PrintPrediction(i, engine.Predict(new TimeSeriesData(i)));

            // 0       0      -2.74    0.40   <-- saved to disk, re-loaded, and running new predictions
            // 1       0      -1.47    0.42
            // 2       0     -17.50    0.24
            // 3       0     -30.82    0.16
            // 4       0     -23.24    0.28
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
