using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic
{
    public static class DetectAnomalyBySrCnn
    {
        // This example creates a time series (list of Data with the i-th element
        // corresponding to the i-th time slot). The estimator is applied then to
        // identify spiking points in the series.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
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

            // The transformed model.
            ITransformer model = ml.Transforms.DetectAnomalyBySrCnn(
                outputColumnName, inputColumnName, 16, 5, 5, 3, 8, 0.35).Fit(
                dataView);

            // Create a time series prediction engine from the model.
            var engine = model.CreateTimeSeriesEngine<TimeSeriesData,
                SrCnnAnomalyDetection>(ml);

            Console.WriteLine($"{outputColumnName} column obtained post-" +
                $"transformation.");

            Console.WriteLine("Data\tAlert\tScore\tMag");

            // Prediction column obtained post-transformation.
            // Data	Alert	Score	Mag

            // Create non-anomalous data and check for anomaly.
            for (int index = 0; index < 20; index++)
            {
                // Anomaly detection.
                PrintPrediction(5, engine.Predict(new TimeSeriesData(5)));
            }

            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.00    0.00
            //5   0   0.03    0.18
            //5   0   0.03    0.18
            //5   0   0.03    0.18
            //5   0   0.03    0.18
            //5   0   0.03    0.18

            // Anomaly.
            PrintPrediction(10, engine.Predict(new TimeSeriesData(10)));

            //10	1	0.47	0.93    <-- alert is on, predicted anomaly

            // Checkpoint the model.
            var modelPath = "temp.zip";
            engine.CheckPoint(ml, modelPath);

            // Load the model.
            using (var file = File.OpenRead(modelPath))
                model = ml.Model.Load(file, out DataViewSchema schema);

            for (int index = 0; index < 5; index++)
            {
                // Anomaly detection.
                PrintPrediction(5, engine.Predict(new TimeSeriesData(5)));
            }

            //5   0   0.31    0.50
            //5   0   0.05    0.30
            //5   0   0.01    0.23
            //5   0   0.00    0.21
            //5   0   0.01    0.25
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
