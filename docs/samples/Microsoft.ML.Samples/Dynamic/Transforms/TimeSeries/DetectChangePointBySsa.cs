using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic
{
    public static class DetectChangePointBySsa
    {
        // This example creates a time series (list of Data with the i-th element
        // corresponding to the i-th time slot). It demonstrates stateful prediction
        // engine that updates the state of the model and allows for
        // saving/reloading. The estimator is applied then to identify points where
        // data distribution changed. This estimator can account for temporal
        // seasonality in the data.
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

            // Setup SsaChangePointDetector arguments
            var inputColumnName = nameof(TimeSeriesData.Value);
            var outputColumnName = nameof(ChangePointPrediction.Prediction);
            double confidence = 95;
            int changeHistoryLength = 8;

            // Train the change point detector.
            ITransformer model = ml.Transforms.DetectChangePointBySsa(
                outputColumnName, inputColumnName, confidence, changeHistoryLength,
                TrainingSize, SeasonalitySize + 1).Fit(dataView);

            // Create a prediction engine from the model for feeding new data.
            var engine = model.CreateTimeSeriesEngine<TimeSeriesData,
                ChangePointPrediction>(ml);

            // Start streaming new data points with no change point to the
            // prediction engine.
            Console.WriteLine($"Output from ChangePoint predictions on new data:");
            Console.WriteLine("Data\tAlert\tScore\tP-Value\tMartingale value");

            // Output from ChangePoint predictions on new data:
            // Data    Alert   Score   P-Value Martingale value

            for (int i = 0; i < 5; i++)
                PrintPrediction(i, engine.Predict(new TimeSeriesData(i)));

            // 0       0      -1.01    0.50    0.00
            // 1       0      -0.24    0.22    0.00
            // 2       0      -0.31    0.30    0.00
            // 3       0       0.44    0.01    0.00
            // 4       0       2.16    0.00    0.24

            // Now stream data points that reflect a change in trend.
            for (int i = 0; i < 5; i++)
            {
                int value = (i + 1) * 100;
                PrintPrediction(value, engine.Predict(new TimeSeriesData(value)));
            }
            // 100     0      86.23    0.00    2076098.24
            // 200     0     171.38    0.00    809668524.21
            // 300     1     256.83    0.01    22130423541.93    <-- alert is on, note that delay is expected
            // 400     0     326.55    0.04    241162710263.29
            // 500     0     364.82    0.08    597660527041.45   <-- saved to disk

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
                ChangePointPrediction>(ml);

            // Run predictions on the loaded model.
            for (int i = 0; i < 5; i++)
            {
                int value = (i + 1) * 100;
                PrintPrediction(value, engine.Predict(new TimeSeriesData(value)));
            }

            // 100     0     -58.58    0.15    1096021098844.34  <-- loaded from disk and running new predictions
            // 200     0     -41.24    0.20    97579154688.98
            // 300     0     -30.61    0.24    95319753.87
            // 400     0      58.87    0.38    14.24
            // 500     0     219.28    0.36    0.05

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
