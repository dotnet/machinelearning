    // the alignment of the usings with the methods is intentional so they can display on the same level in the docs site. 
    using System;
    using System.Linq;
    using System.Collections.Generic;
    using Microsoft.ML.Runtime.Data;
    using Microsoft.ML.Runtime.Api;
    using Microsoft.ML.Runtime.TimeSeriesProcessing;

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
        public static void SsaSpikeDetectorTransform()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a spike
            const int size = 16;
            var data = new List<SsaSpikeData>(size);
            for (int i = 0; i < size / 2; i++)
                data.Add(new SsaSpikeData(5));
            // This is a spike
            data.Add(new SsaSpikeData(10));
            for (int i = 0; i < size / 2; i++)
                data.Add(new SsaSpikeData(5));

            // Convert data to IDataView.
            var dataView = ml.CreateStreamingDataView(data);

            // Setup IidSpikeDetector arguments
            string outputColumnName = "Prediction";
            string inputColumnName = "Value";
            var args = new SsaSpikeDetector.Arguments()
            {
                Source = inputColumnName,
                Name = outputColumnName,
                Confidence = 95,                // The confidence for spike detection in the range [0, 100]
                PvalueHistoryLength = size / 4, // The size of the sliding window for computing the p-value
                TrainingWindowSize = size / 2,  // The number of points from the beginning of the sequence used for training.
                SeasonalWindowSize = size / 8,  // An upper bound on the largest relevant seasonality in the input time - series."
            };

            // The transformed data.
            var transformedData = new SsaSpikeEstimator(ml, args).Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of SsaSpikePrediction.
            var predictionColumn = transformedData.AsEnumerable<SsaSpikePrediction>(ml, reuseRowObject: false);

            Console.WriteLine($"{outputColumnName} column obtained post-transformation.");
            Console.WriteLine("Alert\tScore\tP-Value");
            foreach (var prediction in predictionColumn)
                Console.WriteLine("{0}\t{1:0.00}\t{2:0.00}", prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2]);
            Console.WriteLine("");

            // Prediction column obtained post-transformation.
            // Alert   Score   P-Value
            // 0       0.00    0.50
            // 0       0.00    0.50
            // 0       0.00    0.50
            // 0       0.00    0.50
            // 0       0.00    0.50
            // 0       0.00    0.50
            // 0       0.00    0.50
            // 0       0.00    0.50
            // 1       5.00    0.00   <-- alert is on, predicted spike
            // 0       -2.50   0.09
            // 0       -2.50   0.22
            // 0       0.00    0.47
            // 0       0.00    0.47
            // 0       0.00    0.26
            // 0       0.00    0.38
            // 0       0.00    0.50
            // 0       0.00    0.50
        }
    }
}
