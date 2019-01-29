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
        class IidSpikeData
        {
            public float Value;

            public IidSpikeData(float value)
            {
                Value = value;
            }
        }

        class IidSpikePrediction
        {
            [VectorType(3)]
            public double[] Prediction { get; set; }
        }

        // This example creates a time series (list of Data with the i-th element corresponding to the i-th time slot). 
        // IidSpikeDetector is applied then to identify spiking points in the series.
        public static void IidSpikeDetectorTransform()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a spike
            const int Size = 10;
            var data = new List<IidSpikeData>(Size);
            for (int i = 0; i < Size / 2; i++)
                data.Add(new IidSpikeData(5));
            // This is a spike
            data.Add(new IidSpikeData(10));
            for (int i = 0; i < Size / 2; i++)
                data.Add(new IidSpikeData(5));

            // Convert data to IDataView.
            var dataView = ml.Data.ReadFromEnumerable(data);

            // Setup IidSpikeDetector arguments
            string outputColumnName = nameof(IidSpikePrediction.Prediction);
            string inputColumnName = nameof(IidSpikeData.Value);
            var args = new IidSpikeDetector.Arguments()
            {
                Source = inputColumnName,
                Name = outputColumnName,
                Confidence = 95,                // The confidence for spike detection in the range [0, 100]
                PvalueHistoryLength = Size / 4  // The size of the sliding window for computing the p-value; shorter windows are more sensitive to spikes.
            };

            // The transformed data.
            var transformedData = new IidSpikeEstimator(ml, args).Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of IidSpikePrediction.
            var predictionColumn = ml.CreateEnumerable<IidSpikePrediction>(transformedData, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained post-transformation.");
            Console.WriteLine("Alert\tScore\tP-Value");
            foreach (var prediction in predictionColumn)
                Console.WriteLine("{0}\t{1:0.00}\t{2:0.00}", prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2]);
            Console.WriteLine("");

            // Prediction column obtained post-transformation.
            // Alert   Score   P-Value
            // 0       5.00    0.50
            // 0       5.00    0.50
            // 0       5.00    0.50
            // 0       5.00    0.50
            // 0       5.00    0.50
            // 1       10.00   0.00   <-- alert is on, predicted spike
            // 0       5.00    0.26
            // 0       5.00    0.26
            // 0       5.00    0.50
            // 0       5.00    0.50
            // 0       5.00    0.50
        }

        public static void IidSpikeDetectorPrediction()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a spike
            const int Size = 10;
            var data = new List<IidSpikeData>(Size);
            for (int i = 0; i < Size / 2; i++)
                data.Add(new IidSpikeData(5));
            // This is a spike
            data.Add(new IidSpikeData(10));
            for (int i = 0; i < Size / 2; i++)
                data.Add(new IidSpikeData(5));

            // Convert data to IDataView.
            var dataView = ml.Data.ReadFromEnumerable(data);

            // Setup IidSpikeDetector arguments
            string outputColumnName = nameof(IidSpikePrediction.Prediction);
            string inputColumnName = nameof(IidSpikeData.Value);
            var args = new IidSpikeDetector.Arguments()
            {
                Source = inputColumnName,
                Name = outputColumnName,
                Confidence = 95,                // The confidence for spike detection in the range [0, 100]
                PvalueHistoryLength = Size / 4  // The size of the sliding window for computing the p-value; shorter windows are more sensitive to spikes.
            };

            // The transformed model.
            ITransformer model = new IidSpikeEstimator(ml, args).Fit(dataView);

            // Create a time series prediction engine from the model.
            var engine = model.CreateTimeSeriesPredictionFunction<IidSpikeData, IidSpikePrediction>(ml);
            for (int index = 0; index < 5; index++)
            {
                // Anomaly spike detection.
                var prediction = engine.Predict(new IidSpikeData(5));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", 5, prediction.Prediction[0],
                    prediction.Prediction[1], prediction.Prediction[2]);
            }

            // Spike.
            var spikePrediction = engine.Predict(new IidSpikeData(10));
            Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", 10, spikePrediction.Prediction[0],
                spikePrediction.Prediction[1], spikePrediction.Prediction[2]);

            // Checkpoint the model.
            var modelPath = "temp.zip";
            engine.CheckPoint(ml, modelPath);

            // Load the model.
            using (var file = File.OpenRead(modelPath))
                model = TransformerChain.LoadFrom(ml, file);

            for (int index = 0; index < 5; index++)
            {
                // Anomaly spike detection.
                var prediction = engine.Predict(new IidSpikeData(5));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", 5, prediction.Prediction[0],
                    prediction.Prediction[1], prediction.Prediction[2]);
            }

            // Data Alert   Score   P-Value
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 10     1      10.00    0.00  <-- alert is on, predicted spike (check-point model)
            // 5      0       5.00    0.26  <-- load model from disk.
            // 5      0       5.00    0.26
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
        }
    }
}
