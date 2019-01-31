using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

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
            public int[] EducationCategory = default;
        }

        /// This example demonstrates the use arrays as the values for the ValueMappingEstimator. It maps a set of keys that are type string
        /// to a integer arrays of variable length.
        /// The mapping looks like the following:
        ///     0-5yrs  -> 1,2,3,4
        ///     6-11yrs -> 5,6,7
        ///     12+yrs  -> 42, 32
        public static void Run()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            IDataView trainData = mlContext.Data.ReadFromEnumerable(data);

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
                new int[] { 1,2,3,4 },
                new int[] { 5,6,7 },
                new int[] { 42, 32 }
            };

            // Constructs the ValueMappingEstimator making the ML.net pipeline
            var pipeline = mlContext.Transforms.Conversion.ValueMap(educationKeys, educationValues, ("EducationCategory", "Education"));

            // Fits the ValueMappingEstimator and transforms the data adding the EducationCategory column.
            IDataView transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the resulting data as an IEnumerable of SampleInfertDataWithIntArray. This will contain the newly created column EducationCategory
            IEnumerable<SampleInfertDataWithIntArray> featuresColumn = mlContext.CreateEnumerable<SampleInfertDataWithIntArray>(transformedData, reuseRowObject: false);

            Console.WriteLine($"Example of mapping string->array");
            Console.WriteLine($"Age\tEducation\tEducationCategory");
            foreach (var featureRow in featuresColumn)
            {
                Console.WriteLine($"{featureRow.Age}\t{featureRow.Education}  \t{string.Join(",", featureRow.EducationCategory)}");
            }

            // Features column obtained post-transformation.
            //
            // Example of mapping string->array
            // Age     Education   EducationCategory
            // 26      0 - 5yrs    1,2,3,4
            // 42      0 - 5yrs    1,2,3,4
            // 39      12 + yrs    42,32
            // 34      0 - 5yrs    1,2,3,4
            // 35      6 - 11yrs   5,6,7
        }
    }
}