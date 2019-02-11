using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;
using Microsoft.ML.TimeSeriesProcessing;

namespace Microsoft.ML.Samples.Dynamic
{
    public partial class TransformSamples
    {
        class SsaSpikeData
        {
            public float Value;

            public SsaSpikeData(float value)
            {
                Value = value;
            }
        }

        class SsaSpikePrediction
        {
            [VectorType(3)]
            public double[] Prediction { get; set; }
        }

        // This example creates a time series (list of Data with the i-th element corresponding to the i-th time slot). 
        // SsaSpikeDetector is applied then to identify spiking points in the series.
        // SsaSpikeDetector differs from IidSpikeDetector in that it can account for temporal seasonality
        // in the data.
        public static void SsaSpikeDetectorTransform()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a recurring pattern and a spike within the pattern
            const int SeasonalitySize = 5;
            const int TrainingSeasons = 3;
            const int TrainingSize = SeasonalitySize * TrainingSeasons;
            var data = new List<SsaSpikeData>();
            for (int i = 0; i < TrainingSeasons; i++)
                for (int j = 0; j < SeasonalitySize; j++)
                    data.Add(new SsaSpikeData(j));
            // This is a spike
            data.Add(new SsaSpikeData(100));
            for (int i = 0; i < SeasonalitySize; i++)
                data.Add(new SsaSpikeData(i));

            // Convert data to IDataView.
            var dataView = ml.Data.ReadFromEnumerable(data);

            // Setup IidSpikeDetector arguments
            var inputColumnName = nameof(SsaSpikeData.Value);
            var outputColumnName = nameof(SsaSpikePrediction.Prediction);

            // The transformed data.
            var transformedData = ml.Transforms.SsaSpikeEstimator(outputColumnName, inputColumnName, 95, 8, TrainingSize, SeasonalitySize + 1).Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of SsaSpikePrediction.
            var predictionColumn = ml.CreateEnumerable<SsaSpikePrediction>(transformedData, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained post-transformation.");
            Console.WriteLine("Data\tAlert\tScore\tP-Value");
            int k = 0;
            foreach (var prediction in predictionColumn)
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", data[k++].Value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2]);
            Console.WriteLine("");

            // Prediction column obtained post-transformation.
            // Data    Alert   Score   P-Value
            // 0       0     - 2.53    0.50
            // 1       0     - 0.01    0.01
            // 2       0       0.76    0.14
            // 3       0       0.69    0.28
            // 4       0       1.44    0.18
            // 0       0     - 1.84    0.17
            // 1       0       0.22    0.44
            // 2       0       0.20    0.45
            // 3       0       0.16    0.47
            // 4       0       1.33    0.18
            // 0       0     - 1.79    0.07
            // 1       0       0.16    0.50
            // 2       0       0.09    0.50
            // 3       0       0.08    0.45
            // 4       0       1.31    0.12
            // 100     1      98.21    0.00   <-- alert is on, predicted spike
            // 0       0    - 13.83    0.29
            // 1       0     - 1.74    0.44
            // 2       0     - 0.47    0.46
            // 3       0    - 16.50    0.29
            // 4       0    - 29.82    0.21
        }

        // This example shows spike detection as above, but demonstrates how to train a model
        // that can run predictions on streaming data, and how to persist the trained model and then re-load it.
        public static void SsaSpikeDetectorPrediction()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a recurring pattern
            const int SeasonalitySize = 5;
            const int TrainingSeasons = 3;
            const int TrainingSize = SeasonalitySize * TrainingSeasons;
            var data = new List<SsaSpikeData>();
            for (int i = 0; i < TrainingSeasons; i++)
                for (int j = 0; j < SeasonalitySize; j++)
                    data.Add(new SsaSpikeData(j));

            // Convert data to IDataView.
            var dataView = ml.Data.ReadFromEnumerable(data);

            // Setup IidSpikeDetector arguments
            var inputColumnName = nameof(SsaSpikeData.Value);
            var outputColumnName = nameof(SsaSpikePrediction.Prediction);

            // Train the change point detector.
            ITransformer model = ml.Transforms.SsaChangePointEstimator(outputColumnName, inputColumnName, 95, 8, TrainingSize, SeasonalitySize + 1).Fit(dataView);

            // Create a prediction engine from the model for feeding new data.
            var engine = model.CreateTimeSeriesPredictionFunction<SsaSpikeData, SsaSpikePrediction>(ml);

            // Start streaming new data points with no change point to the prediction engine.
            Console.WriteLine($"Output from spike predictions on new data:");
            Console.WriteLine("Data\tAlert\tScore\tP-Value");
            SsaSpikePrediction prediction = null;
            for (int j = 0; j < 2; j++)
            {
                for (int i = 0; i < 5; i++)
                {
                    var value = i;
                    prediction = engine.Predict(new SsaSpikeData(value));
                    Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2]);
                }
            }

            // Now send a data point that reflects a spike.
            var newValue = 100;
            prediction = engine.Predict(new SsaSpikeData(newValue));
            Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", newValue, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2]);

            // Now we demonstrate saving and loading the model.

            // Save the model that exists within the prediction engine.
            // The engine has been updating this model with every new data point.
            var modelPath = "model.zip";
            engine.CheckPoint(ml, modelPath);

            // Load the model.
            using (var file = File.OpenRead(modelPath))
                model = TransformerChain.LoadFrom(ml, file);

            // We must create a new prediction engine from the persisted model.
            engine = model.CreateTimeSeriesPredictionFunction<SsaSpikeData, SsaSpikePrediction>(ml);

            // Run predictions on the loaded model.
            for (int i = 0; i < 5; i++)
            {
                var value = i;
                prediction = engine.Predict(new SsaSpikeData(value));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2]);
            }

            // Output from spike predictions on new data:
            // Data    Alert   Score   P-Value
            // 0       0     - 1.01    0.50
            // 1       0     - 0.24    0.22
            // 2       0     - 0.31    0.30
            // 3       0       0.44    0.01
            // 4       0       2.16    0.00
            // 0       0     - 0.78    0.27
            // 1       0     - 0.80    0.30
            // 2       0     - 0.84    0.31
            // 3       0       0.33    0.31
            // 4       0       2.21    0.07
            // 100     1      86.17    0.00   <-- alert is on, predicted spike
            // 0       0     - 2.74    0.40   <-- saved to disk, re-loaded, and running new predictions
            // 1       0     - 1.47    0.42
            // 2       0    - 17.50    0.24
            // 3       0    - 30.82    0.16
            // 4       0    - 23.24    0.28
        }
    }
}
