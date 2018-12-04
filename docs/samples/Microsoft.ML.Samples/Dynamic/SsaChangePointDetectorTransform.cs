using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Microsoft.ML.TimeSeries;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic
{
    public partial class TransformSamples
    {
        class SsaChangePointData
        {
            public float Value;

            public SsaChangePointData(float value)
            {
                Value = value;
            }
        }

        // This example creates a time series (list of Data with the i-th element corresponding to the i-th time slot). 
        // SsaChangePointDetector is applied then to identify points where data distribution changed.
        // SsaChangePointDetector differs from IidChangePointDetector in that it can account for temporal seasonality
        // in the data.
        public static void SsaChangePointDetectorTransform()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a recurring pattern and then a change in trend
            const int SeasonalitySize = 5;
            const int TrainingSeasons = 3;
            const int TrainingSize = SeasonalitySize * TrainingSeasons;
            var data = new List<SsaChangePointData>();
            for (int i = 0; i < TrainingSeasons; i++)
                for (int j = 0; j < SeasonalitySize; j++)
                    data.Add(new SsaChangePointData(j));
            // This is a change point
            for (int i = 0; i < SeasonalitySize; i++)
                data.Add(new SsaChangePointData(i * 100));

            // Convert data to IDataView.
            var dataView = ml.CreateStreamingDataView(data);

            // Setup SsaChangePointDetector arguments
            var inputColumnName = nameof(SsaChangePointData.Value);
            var outputColumnName = nameof(ChangePointPrediction.Prediction);
            var args = new SsaChangePointDetector.Arguments()
            {
                Source = inputColumnName,
                Name = outputColumnName,
                Confidence = 95,                          // The confidence for spike detection in the range [0, 100]
                ChangeHistoryLength = 8,                  // The length of the window for detecting a change in trend; shorter windows are more sensitive to spikes.
                TrainingWindowSize = TrainingSize,        // The number of points from the beginning of the sequence used for training.
                SeasonalWindowSize = SeasonalitySize + 1  // An upper bound on the largest relevant seasonality in the input time series."

            };

            // The transformed data.
            var transformedData = new SsaChangePointEstimator(ml, args).Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of ChangePointPrediction.
            var predictionColumn = transformedData.AsEnumerable<ChangePointPrediction>(ml, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained post-transformation.");
            Console.WriteLine("Data\tAlert\tScore\tP-Value\tMartingale value");
            int k = 0;
            foreach (var prediction in predictionColumn)
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t{4:0.00}", data[k++].Value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2], prediction.Prediction[3]);
            Console.WriteLine("");

            // Prediction column obtained post-transformation.
            // Data    Alert   Score   P-Value Martingale value
            // 0       0     - 2.53    0.50    0.00
            // 1       0     - 0.01    0.01    0.00
            // 2       0       0.76    0.14    0.00
            // 3       0       0.69    0.28    0.00
            // 4       0       1.44    0.18    0.00
            // 0       0     - 1.84    0.17    0.00
            // 1       0       0.22    0.44    0.00
            // 2       0       0.20    0.45    0.00
            // 3       0       0.16    0.47    0.00
            // 4       0       1.33    0.18    0.00
            // 0       0     - 1.79    0.07    0.00
            // 1       0       0.16    0.50    0.00
            // 2       0       0.09    0.50    0.00
            // 3       0       0.08    0.45    0.00
            // 4       0       1.31    0.12    0.00
            // 0       0     - 1.79    0.07    0.00
            // 100     1      99.16    0.00    4031.94     <-- alert is on, predicted changepoint
            // 200     0     185.23    0.00    731260.87
            // 300     0     270.40    0.01    3578470.47
            // 400     0     357.11    0.03    45298370.86
        }

        // This example shows change point detection as above, but demonstrates how to train a model
        // that can run predictions on streaming data, and how to persist the trained model and then re-load it.
        public static void SsaChangePointDetectorPrediction()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a recurring pattern
            const int SeasonalitySize = 5;
            const int TrainingSeasons = 3;
            const int TrainingSize = SeasonalitySize * TrainingSeasons;
            var data = new List<SsaChangePointData>();
            for (int i = 0; i < TrainingSeasons; i++)
                for (int j = 0; j < SeasonalitySize; j++)
                    data.Add(new SsaChangePointData(j));

            // Convert data to IDataView.
            var dataView = ml.CreateStreamingDataView(data);

            // Setup SsaChangePointDetector arguments
            var inputColumnName = nameof(SsaChangePointData.Value);
            var outputColumnName = nameof(ChangePointPrediction.Prediction);
            var args = new SsaChangePointDetector.Arguments()
            {
                Source = inputColumnName,
                Name = outputColumnName,
                Confidence = 95,                         // The confidence for spike detection in the range [0, 100]
                ChangeHistoryLength = 8,                 // The length of the window for detecting a change in trend; shorter windows are more sensitive to spikes. 
                TrainingWindowSize = TrainingSize,       // The number of points from the beginning of the sequence used for training.
                SeasonalWindowSize = SeasonalitySize + 1 // An upper bound on the largest relevant seasonality in the input time series."

            };

            // Train the change point detector.
            ITransformer model = new SsaChangePointEstimator(ml, args).Fit(dataView);

            // Create a prediction engine from the model for feeding new data.
            var engine = model.CreateTimeSeriesPredictionFunction<SsaChangePointData, ChangePointPrediction>(ml);

            // Start streaming new data points with no change point to the prediction engine.
            Console.WriteLine($"Output from ChangePoint predictions on new data:");
            Console.WriteLine("Data\tAlert\tScore\tP-Value\tMartingale value");
            ChangePointPrediction prediction = null;
            for (int i = 0; i < 5; i++)
            {
                var value = i;
                prediction = engine.Predict(new SsaChangePointData(value));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t{4:0.00}", value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2], prediction.Prediction[3]);
            }

            // Now stream data points that reflect a change in trend.
            for (int i = 0; i < 5; i++)
            {
                var value = (i + 1) * 100;
                prediction = engine.Predict(new SsaChangePointData(value));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t{4:0.00}", value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2], prediction.Prediction[3]);
            }

            // Now we demonstrate saving and loading the model.

            // Save the model that exists within the prediction engine.
            // The engine has been updating this model with every new data point.
            var modelPath = "model.zip";
            engine.CheckPoint(ml, modelPath);

            // Load the model.
            using (var file = File.OpenRead(modelPath))
                model = TransformerChain.LoadFrom(ml, file);

            // We must create a new prediction engine from the persisted model.
            engine = model.CreateTimeSeriesPredictionFunction<SsaChangePointData, ChangePointPrediction>(ml);

            // Run predictions on the loaded model.
            for (int i = 0; i < 5; i++)
            {
                var value = (i + 1) * 100;
                prediction = engine.Predict(new SsaChangePointData(value));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t{4:0.00}", value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2], prediction.Prediction[3]);
            }

            // Output from ChangePoint predictions on new data:
            // Data    Alert   Score   P-Value Martingale value
            // 0       0     - 1.01    0.50    0.00
            // 1       0     - 0.24    0.22    0.00
            // 2       0     - 0.31    0.30    0.00
            // 3       0       0.44    0.01    0.00
            // 4       0       2.16    0.00    0.24
            // 100     0      86.23    0.00    2076098.24
            // 200     0     171.38    0.00    809668524.21
            // 300     1     256.83    0.01    22130423541.93    <-- alert is on, note that delay is expected
            // 400     0     326.55    0.04    241162710263.29
            // 500     0     364.82    0.08    597660527041.45   <-- saved to disk
            // 100     0    - 58.58    0.15    1096021098844.34  <-- loaded from disk and running new predictions
            // 200     0    - 41.24    0.20    97579154688.98
            // 300     0    - 30.61    0.24    95319753.87
            // 400     0      58.87    0.38    14.24
            // 500     0     219.28    0.36    0.05
        }
    }
}
