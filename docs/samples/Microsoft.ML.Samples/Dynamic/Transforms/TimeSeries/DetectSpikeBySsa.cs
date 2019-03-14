using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class DetectSpikeBySsa
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
        // The estimator is applied then to identify spiking points in the series.
        // This estimator can account for temporal seasonality in the data.
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
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup estimator arguments
            var inputColumnName = nameof(SsaSpikeData.Value);
            var outputColumnName = nameof(SsaSpikePrediction.Prediction);

            // The transformed data.
            var transformedData = ml.Transforms.DetectSpikeBySsa(outputColumnName, inputColumnName, 95, 8, TrainingSize, SeasonalitySize + 1).Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of SsaSpikePrediction.
            var predictionColumn = ml.Data.CreateEnumerable<SsaSpikePrediction>(transformedData, reuseRowObject: false);

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
    }
}
