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

        /// This example demonstrates the use of KeyTypes using both the ValueMappingEstimator and KeyToValueEstimator. Using a KeyType
        /// instead of the actual value provides a unique integer representation of the value. When the treatValueAsKeyTypes is true, 
        /// the ValueMappingEstimator will generate a KeyType for each unique value.
        /// 
        /// In this example, the education data is mapped to a grouping of 'Undergraduate' and 'Postgraduate'. Because KeyTypes are used, the
        /// ValueMappingEstimator will output the KeyType value rather than string value of 'Undergraduate' or 'Postgraduate'.
        /// 
        /// The KeyToValueEstimator is added to the pipeline to convert the KeyType back to the original value. Therefore the output of this example 
        /// results in the string value of 'Undergraduate' and 'Postgraduate'.
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
                "Undergraduate",
                "Postgraduate",
                "Postgraduate"
            };

            // Generate the ValueMappingEstimator that will output KeyTypes even though our values are strings.
            // The KeyToValueMappingEstimator is added to provide a reverse lookup of the KeyType, converting the KeyType value back
            // to the original value.
            var pipeline = new ValueMappingEstimator<string, string>(mlContext, educationKeys, educationValues, true, ("EducationKeyType", "Education"))
                              .Append(mlContext.Transforms.Conversion.MapKeyToValue(("EducationCategory", "EducationKeyType")));

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
            // 26  0-5yrs       Undergraduate 
            // 42  0-5yrs       Undergraduate 
            // 39  12+yrs       Postgraduate 
            // 34  0-5yrs       Undergraduate 
            // 35  6-11yrs      Postgraduate 
        }
    }
}