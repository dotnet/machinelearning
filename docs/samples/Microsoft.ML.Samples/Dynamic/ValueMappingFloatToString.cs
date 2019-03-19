using System;
using System.Collections.Generic;
namespace Microsoft.ML.Samples.Dynamic
{
    public static class ValueMappingFloatToString
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
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleTemperatureData> data = SamplesUtils.DatasetUtils.GetSampleTemperatureData(5);
            IDataView trainData = mlContext.Data.LoadFromEnumerable(data);

            // If the list of keys and values are known, they can be passed to the API. The ValueMappingEstimator can also get the mapping through an IDataView
            // Creating a list of key-value pairs based on the induced value from the dataset
            var temperatureMap = new Dictionary<float, string>();
            temperatureMap[36.0f] = "T1";
            temperatureMap[35.0f] = "T2";
            temperatureMap[34.0f] = "T3";

            // Constructs the ValueMappingEstimator making the ML.net pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValue("TemperatureCategory", temperatureMap, "Temperature");

            // Fits the ValueMappingEstimator and transforms the data adding the TemperatureCategory column.
            IDataView transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the resulting data as an IEnumerable of SampleTemperatureDataWithCategory. This will contain the newly created column TemperatureCategory
            IEnumerable<SampleTemperatureDataWithCategory> featureRows = mlContext.Data.CreateEnumerable<SampleTemperatureDataWithCategory>(transformedData, reuseRowObject: false);

            Console.WriteLine($"Example of mapping float->string");
            Console.WriteLine($"Date\t\tTemperature\tTemperatureCategory");
            foreach (var featureRow in featureRows)
            {
                Console.WriteLine($"{featureRow.Date.ToString("d")}\t{featureRow.Temperature}\t\t{featureRow.TemperatureCategory}");
            }

            // Expected output:
            //  Features column obtained post-transformation.
            //  Example of mapping float->string
            //  Date         Temperature TemperatureCategory
            //  1/2/2012        36              T1
            //  1/3/2012        36              T1
            //  1/4/2012        34              T3
            //  1/5/2012        35              T2
            //  1/6/2012        35              T2
        }
    }
}