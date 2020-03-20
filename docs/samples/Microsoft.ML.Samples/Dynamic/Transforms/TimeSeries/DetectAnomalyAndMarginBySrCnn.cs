using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;


namespace Samples.Dynamic.Transforms.TimeSeries
{
    class DetectAnomalyAndMarginBySrCnn
    {
        // This example creates a time series (list of Data with the i-th element
        // corresponding to the i-th time slot). The estimator is applied then to
        // identify spiking points in the series, and also output the margin.
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
                outputColumnName, inputColumnName, 16, 5, 5, 3, 8, 0.35, SrCnnDetectMode.AnomalyAndMargin, 90.0).Fit(
                dataView);

            // Create a time series prediction engine from the model.
            var engine = model.CreateTimeSeriesEngine<TimeSeriesData,
                SrCnnAnomalyDetection>(ml);

            Console.WriteLine($"{outputColumnName} column obtained post-" +
                $"transformation.");

            Console.WriteLine("Data\tAlert\tAnomalyScore\tMag\tExpectedValue\tBoundaryUnit\tUpperBoundary\tLowerBoundary");

            // Prediction column obtained post-transformation.
            // Data	Alert	Score	        Mag ExpectedValue   BoundaryUnit    UpperBoundary   LowerBoundary

            // Create non-anomalous data and check for anomaly.
            for (int index = 0; index < 20; index++)
            {
                // Anomaly detection.
                PrintPrediction(5, engine.Predict(new TimeSeriesData(5)));
            }

            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.00    0.00            0.00            0.00            0.00
            //5       0       0.00            0.18    5.00            5.00            5.01            4.99
            //5       0       0.00            0.18    5.00            5.00            5.01            4.99
            //5       0       0.00            0.18    5.00            5.00            5.01            4.99
            //5       0       0.00            0.18    5.00            5.00            5.01            4.99
            //5       0       0.00            0.18    5.00            5.00            5.01            4.99

            // Anomaly.
            PrintPrediction(10, engine.Predict(new TimeSeriesData(10)));

            //10      1       0.50            0.93    5.00            5.00            5.01            4.99  < -- alert is on, predicted anomaly

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

            //5       0       0.00            0.50    5.00            5.00            5.01            4.99
            //5       0       0.00            0.30    5.00            5.00            5.01            4.99
            //5       0       0.00            0.23    5.00            5.00            5.01            4.99
            //5       0       0.00            0.21    5.00            5.00            5.01            4.99
            //5       0       0.00            0.25    5.00            5.00            5.01            4.99
        }

        private static void PrintPrediction(float value, SrCnnAnomalyDetection
            prediction) =>
            Console.WriteLine("{0}\t{1}\t{2:0.00}\t\t{3:0.00}\t{4:0.00}\t\t{5:0.00}\t\t{6:0.00}\t\t{7:0.00}", 
                value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2],
                prediction.Prediction[3], prediction.Prediction[4], prediction.Prediction[5], prediction.Prediction[6]);

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
            [VectorType(7)]
            public double[] Prediction { get; set; }
        }
    }
}
