using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public class ValueMappingFloatToStringExample
    {
        /// <summary>
        /// Helper class for retrieving the resulting data
        /// </summary>
        class SampleTemperatureDataWithCategory
        {
            public DateTime Date = default;
            public float Temperature = 0.0f;
            public string TemperatureCategory = default;
        }

        /// This example demonstrates the use of ValueMappingEstimator by mapping float-to-string values. This is useful if the key
        /// data are floating point and need to be grouped into string values. In this example, the Induction value is mapped to 
        /// "T1", "T2", "T3", and "T4" groups.
        public static void Run()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleTemperatureData> data = SamplesUtils.DatasetUtils.GetSampleTemperatureData();
            IDataView trainData = mlContext.Data.ReadFromEnumerable(data);

            // If the list of keys and values are known, they can be passed to the API. The ValueMappingEstimator can also get the mapping through an IDataView
            // Creating a list of keys based on the induced value from the dataset
            var temperatureKeys = new List<float>()
            {
                39.0F,
                67.0F,
                75.0F,
                82.0F,
            };

            // Creating a list of values, these strings will map accordingly to each key.
            var classificationValues = new List<string>()
            {
                "T1",
                "T2",
                "T3", 
                "T4"
            };

            // Constructs the ValueMappingEstimator making the ML.net pipeline
            var pipeline = mlContext.Transforms.Conversion.ValueMap(temperatureKeys, classificationValues, ("TemperatureCategory", "Temperature"));

            // Fits the ValueMappingEstimator and transforms the data adding the TemperatureCategory column.
            IDataView transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the resulting data as an IEnumerable of SampleTemperatureDataWithCategory. This will contain the newly created column TemperatureCategory
            IEnumerable<SampleTemperatureDataWithCategory> featureRows = mlContext.CreateEnumerable<SampleTemperatureDataWithCategory>(transformedData, reuseRowObject: false);

            Console.WriteLine($"Example of mapping float->string");
            Console.WriteLine($"Date\t\tTemperature\tTemperatureCategory");
            foreach (var featureRow in featureRows)
            {
                Console.WriteLine($"{featureRow.Date.ToString("d")}\t{featureRow.Temperature}\t\t{featureRow.TemperatureCategory}");
            }

            // Features column obtained post-transformation.
            //
            // Example of mapping float->string
            // Date         Temperature TemperatureCategory
            // 1/1/2012     39          T1
            // 1/2/2012     82          T4
            // 1/3/2012     75          T3
            // 1/4/2012     67          T2
            // 1/5/2012     75          T3
        }
    }
}