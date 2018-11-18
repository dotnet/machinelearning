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
        public static void SsaChangePointDetectorTransform()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a change
            const int size = 16;
            var data = new List<SsaChangePointData>(size);
            for (int i = 0; i < size / 2; i++)
                data.Add(new SsaChangePointData(5));
            // This is a change point
            for (int i = 0; i < size / 2; i++)
                data.Add(new SsaChangePointData(7));

            // Convert data to IDataView.
            var dataView = ml.CreateStreamingDataView(data);

            // Setup IidSpikeDetector arguments
            string outputColumnName = "Prediction";
            string inputColumnName = "Value";
            var args = new SsaChangePointDetector.Arguments()
            {
                Source = inputColumnName,
                Name = outputColumnName,
                Confidence = 95,                // The confidence for spike detection in the range [0, 100]
                ChangeHistoryLength = size / 4, // The length of the sliding window on p-values for computing the martingale score. 
                TrainingWindowSize = size / 2,  // The number of points from the beginning of the sequence used for training.
                SeasonalWindowSize = size / 8,  // An upper bound on the largest relevant seasonality in the input time - series."
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
            // Data Alert      Score   P-Value Martingale value
            // 5       0       0.00    0.50    0.00
            // 5       0       0.00    0.50    0.00
            // 5       0       0.00    0.50    0.00
            // 5       0       0.00    0.50    0.00
            // 5       0       0.00    0.50    0.00
            // 5       0       0.00    0.50    0.00
            // 5       0       0.00    0.50    0.00
            // 5       0       0.00    0.50    0.00
            // 7       1       2.00    0.00    10298.67   <-- alert is on, predicted changepoint
            // 7       0       1.00    0.31    15741.58
            // 7       0       0.00    0.28    26487.48
            // 7       0       0.00    0.28    44569.02
            // 7       0       0.00    0.28    0.01
            // 7       0       0.00    0.38    0.01
            // 7       0       0.00    0.50    0.00
            // 7       0       0.00    0.50    0.00
        }
    }
}
