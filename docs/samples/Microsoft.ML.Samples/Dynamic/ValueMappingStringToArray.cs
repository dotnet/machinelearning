using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Conversions;

namespace Microsoft.ML.Samples.Dynamic
{
    public class ValueMappingStringToArrayExample
    {
        /// <summary>
        /// Helper class for retrieving the resulting data
        /// </summary>
        class SampleInfertDataWithIntArray
        {
            public float Age = 0;
            public string Education = default;
            public int[] EducationFeature = default;
        }

        /// This example demonstrates the use of the ValueMappingEstimator by mapping string-to-array values which allows for mapping string data
        /// to numeric arrays that can then be used as a feature set for a trainer. In this example, we are mapping the education data to
        /// arbitrary integer arrays with the following association:
        ///     0-5yrs  -> 1, 2, 3
        ///     6-11yrs -> 5, 6, 7
        ///     12+yrs  -> 42,32,64
        public static void Run()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            IDataView trainData = mlContext.Data.ReadFromEnumerable(data);

            // If the list of keys and values are known, they can be passed to the API. The ValueMappingEstimator can also get the mapping through an IDataView
            // Creating a list of keys based on the Education values from the dataset
            var educationKeys = new List<string>()
            {
                "0-5yrs",
                "6-11yrs",
                "12+yrs"
            };

            // Sample list of associated array values
            var educationValues = new List<int[]>()
            {
                new int[] { 1,2,3 },
                new int[] { 5,6,7 },
                new int[] { 42,32,64 }
            };

            // Constructs the ValueMappingEstimator making the ML.net pipeline
            var pipeline = mlContext.Transforms.Conversion.ValueMap<string, int>(educationKeys, educationValues, ("EducationFeature", "Education"));

            // Fits the ValueMappingEstimator and transforms the data adding the EducationFeature column.
            IDataView transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the resulting data as an IEnumerable of SampleInfertDataWithIntArray. This will contain the newly created column EducationCategory
            IEnumerable<SampleInfertDataWithIntArray> featuresColumn = mlContext.CreateEnumerable<SampleInfertDataWithIntArray>(transformedData, reuseRowObject: false);

            Console.WriteLine($"Example of mapping string->array");
            Console.WriteLine($"Age\tEducation\tEducationFeature");
            foreach (var featureRow in featuresColumn)
            {
                Console.WriteLine($"{featureRow.Age}\t{featureRow.Education}  \t{string.Join(",", featureRow.EducationFeature)}");
            }

            // Features column obtained post-transformation.
            //
            // Example of mapping string->array
            // Age     Education   EducationFeature
            // 26      0 - 5yrs    1,2,3
            // 42      0 - 5yrs    1,2,3
            // 39      12 + yrs    42,32,64
            // 34      0 - 5yrs    1,2,3
            // 35      6 - 11yrs   5,6,7
        }
    }
}