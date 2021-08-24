using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic
{
    public static class DetectIidSpike
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

            // Setup IidSpikeDetector arguments
            string outputColumnName = nameof(IidSpikePrediction.Prediction);
            string inputColumnName = nameof(TimeSeriesData.Value);

            // The transformed model.
            ITransformer model = ml.Transforms.DetectIidSpike(outputColumnName,
                inputColumnName, 95.0d, Size).Fit(dataView);

            // Create a time series prediction engine from the model.
            var engine = model.CreateTimeSeriesEngine<TimeSeriesData,
                IidSpikePrediction>(ml);

            Console.WriteLine($"{outputColumnName} column obtained " +
                $"post-transformation.");

            Console.WriteLine("Data\tAlert\tScore\tP-Value");

            // Prediction column obtained post-transformation.
            // Data Alert   Score   P-Value

            // Create non-anomalous data and check for anomaly.
            for (int index = 0; index < 5; index++)
            {
                // Anomaly spike detection.
                PrintPrediction(5, engine.Predict(new TimeSeriesData(5)));
            }

            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50

            // Spike.
            PrintPrediction(10, engine.Predict(new TimeSeriesData(10)));

            // 10     1      10.00    0.00  <-- alert is on, predicted spike (check-point model)

            // Checkpoint the model.
            var modelPath = "temp.zip";
            engine.CheckPoint(ml, modelPath);

            // Load the model.
            using (var file = File.OpenRead(modelPath))
                model = ml.Model.Load(file, out DataViewSchema schema);

            for (int index = 0; index < 5; index++)
            {
                // Anomaly spike detection.
                PrintPrediction(5, engine.Predict(new TimeSeriesData(5)));
            }

            // 5      0       5.00    0.26  <-- load model from disk.
            // 5      0       5.00    0.26
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50

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
