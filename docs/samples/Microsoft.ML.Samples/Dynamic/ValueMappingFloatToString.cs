using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Conversions;

namespace Microsoft.ML.Samples.Dynamic
{
    public class ValueMappingFloatToStringExample
    {
        /// <summary>
        /// Helper class for retrieving the resulting data
        /// </summary>
        class SampleInfertDataWithInducedCategory
        {
            public float Age = 0;
            public float Induced = 0.0f;
            public string InducedCategory = default;
        }

        ///<summary>
        /// This example demonstrates the use of floating types as the key type for ValueMappingEstimator by mapping a float-to-string value.
        /// The mapping looks like the following:
        /// <list>
        ///     <item>1.0 -> Cat1</item>
        ///     <item>2.0 -> Cat2</item>
        /// </list>
        /// </summary>
        public static void Run()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = ml.Data.ReadFromEnumerable(data);

            // Creating a list of keys based on the induced value from the dataset
            // These lists are created by hand for the demonstration, but the ValueMappingEstimator does take an IEnumerable.
            var inducedKeys = new List<float>()
            {
                1.0f,
                2.0f
            };

            // Creating a list of values, these strings will map accordingly to each key.
            var inducedValues = new List<string>()
            {
                "Cat1",
                "Cat2"
            };

            // Constructs the ValueMappingEstimator making the ML.net pipeline
            var pipeline = new ValueMappingEstimator<float, string>(ml, inducedKeys, inducedValues, ("InducedCategory", "Induced"));

            // Fits the ValueMappingEstimator and transforms the data adding the InducedCategory column.
            IDataView transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the resulting data as an IEnumerable of SampleInfertDataWithInducedCategory. This will contain the newly created column InducedCategory
            IEnumerable<SampleInfertDataWithInducedCategory> featureRows = ml.CreateEnumerable<SampleInfertDataWithInducedCategory>(transformedData, reuseRowObject: false);

            Console.WriteLine($"Example of mapping float->string");
            Console.WriteLine($"Age\tInduced\tInducedCategory");
            foreach (var featureRow in featureRows)
            {
                Console.WriteLine($"{featureRow.Age}\t{featureRow.Induced}\t{featureRow.InducedCategory}");
            }

            // Features column obtained post-transformation.
            //
            // Example of mapping float->string
            // Age     Induced InducedCategory
            // 26      1       Cat1
            // 42      1       Cat1
            // 39      2       Cat2
            // 34      2       Cat2
            // 35      1       Cat1
        }
    }
}