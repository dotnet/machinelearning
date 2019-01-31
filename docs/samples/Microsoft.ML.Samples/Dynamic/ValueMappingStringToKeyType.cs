using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Conversions;

namespace Microsoft.ML.Samples.Dynamic
{
    public class ValueMappingStringToKeyTypeExample
    {
        /// <summary>
        /// Helper class for retrieving the resulting data
        /// </summary>
        class SampleInfertDataWithFeatures

        {
            public float Age = 0;
            public string Education = default;
            public string EducationCategory = default;
        }

        /// This example demonstrates the use of KeyTypes in the ValueMappingEstimator by setting treatValuesAsKeyTypes to true, 
        /// This is useful in cases where you want the output to be integer based rather than the actual value.
        ///
        /// When using KeyTypes as a Value, the ValueMappingEstimator will do one of the following:
        /// 1) If the Value type is an unsigned int or unsigned long, the specified values are used directly as the KeyType values. 
        /// 2) If the Value type is not an unsigned int or unsigned long, new KeyType values are generated for each unique value.
        /// 
        /// In this example, the Value type is a string. Since we are setting treatValueAsKeyTypes to true, 
        /// the ValueMappingEstimator will generate its own KeyType values for each unique string.
        /// As with KeyTypes, they contain the actual Value information as part of the metadata, therefore
        /// we can convert a KeyType back to the actual value the KeyType represents. To demonstrate
        /// the reverse lookup and to confirm the correct value is mapped, a KeyToValueEstimator is added
        /// to the pipeline to convert back to the original value.
        public static void Run()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            IDataView trainData = mlContext.Data.ReadFromEnumerable(data);

            // Creating a list of keys based on the Education values from the dataset
            // These lists are created by hand for the demonstration, but the ValueMappingEstimator does take an IEnumerable.
            var educationKeys = new List<string>()
            {
                "0-5yrs",
                "6-11yrs",
                "12+yrs"
            };

            // Creating a list of values that are sample strings. These will be converted to KeyTypes
            var educationValues = new List<string>()
            {
                "Cat1",
                "Cat2",
                "Cat3"
            };

            // Generate the ValueMappingEstimator that will output KeyTypes even though our values are strings.
            // The KeyToValueMappingEstimator is added to provide a reverse lookup of the KeyType, converting the KeyType value back
            // to the original value.
            var pipeline = new ValueMappingEstimator<string, string>(mlContext, educationKeys, educationValues, true, ("EducationKeyType", "Education"))
                              .Append(new KeyToValueMappingEstimator(mlContext, ("EducationCategory", "EducationKeyType")));

            // Fits the ValueMappingEstimator and transforms the data adding the EducationKeyType column.
            IDataView transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the resulting data as an IEnumerable of SampleInfertDataWithFeatures.
            IEnumerable<SampleInfertDataWithFeatures> featureRows = mlContext.CreateEnumerable<SampleInfertDataWithFeatures>(transformedData, reuseRowObject: false);

            Console.WriteLine($"Example of mapping string->keytype");
            Console.WriteLine($"Age\tEducation\tEducationCategory");
            foreach (var featureRow in featureRows)
            {
                Console.WriteLine($"{featureRow.Age}\t{featureRow.Education}  \t{featureRow.EducationCategory}");
            }

            // Features column obtained post-transformation.
            //
            // Age Education    EducationCategory
            // 26  0-5yrs       Cat1
            // 42  0-5yrs       Cat1
            // 39  12+yrs       Cat3
            // 34  0-5yrs       Cat1
            // 35  6-11yrs      Cat2
        }
    }
}