using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class DetectIidSpike
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
        // The estimator is applied then to identify spiking points in the series.
        public static void Example()
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
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup the estimator arguments
            string outputColumnName = nameof(IidSpikePrediction.Prediction);
            string inputColumnName = nameof(IidSpikeData.Value);

            // The transformed data.
            var transformedData = ml.Transforms.DetectIidSpike(outputColumnName, inputColumnName, 95, Size / 4).Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of IidSpikePrediction.
            var predictionColumn = ml.Data.CreateEnumerable<IidSpikePrediction>(transformedData, reuseRowObject: false);

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
    }
}
