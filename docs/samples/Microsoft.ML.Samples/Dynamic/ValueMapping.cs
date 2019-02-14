using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static partial class ValueMapping
    {
        class SampleInfertDataWithFeatures
        {
            public float Age = 0;
            public string Education = default;
            public string EducationCategory = default;
        }

        /// This example demonstrates the use of the ValueMappingEstimator by mapping string-to-string values. This is useful
        /// to map strings to a grouping. In this example, the education data maps to the groups Undergraduate and Postgraduate:
        ///   0-5yrs  -> Undergraduate 
        ///   6-11yrs -> Postgraduate
        ///   12+yrs  -> Postgraduate
        /// Its possible to have multiple keys map to the same value.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            IDataView trainData = mlContext.Data.ReadFromEnumerable(data);

            // Preview of the data.
            //
            // Age    Case  Education  induced     parity  pooled.stratum  row_num  ...
            // 26.0   1.0   0-5yrs      1.0         6.0       3.0      1.0  ...
            // 42.0   1.0   0-5yrs      1.0         1.0       1.0      2.0  ...
            // 39.0   1.0   12+yrs      2.0         6.0       4.0      3.0  ...
            // 34.0   1.0   0-5yrs      2.0         4.0       2.0      4.0  ...
            // 35.0   1.0   6-11yrs     1.0         3.0       32.0     5.0  ...

            // If the list of keys and values are known, they can be passed to the API. The ValueMappingEstimator can also get the mapping through an IDataView
            // Creating a list of keys based on the Education values from the dataset. 
            var educationKeys = new List<string>()
            {
                "0-5yrs",
                "6-11yrs",
                "12+yrs"
            };

            // Creating a list of associated values that will map respectively to each educationKey
            var educationValues = new List<string>()
            {
                "Undergraduate",
                "Postgraduate",
                "Postgraduate"
            };

            // Constructs the ValueMappingEstimator making the ML.net pipeline
            var pipeline = mlContext.Transforms.Conversion.ValueMap(educationKeys, educationValues, ("EducationCategory", "Education"));

            // Fits the ValueMappingEstimator and transforms the data converting the Education to EducationCategory.
            IDataView transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the resulting data as an IEnumerable of SampleInfertDataWithFeatures. This will contain the newly created column EducationCategory
            IEnumerable<SampleInfertDataWithFeatures> featureRows = mlContext.CreateEnumerable<SampleInfertDataWithFeatures>(transformedData, reuseRowObject: false);

            Console.WriteLine($"Example of mapping string->string");
            Console.WriteLine($"Age\tEducation\tEducationCategory");
            foreach (var featureRow in featureRows)
            {
                Console.WriteLine($"{featureRow.Age}\t{featureRow.Education}  \t{featureRow.EducationCategory}");
            }

            // Features column obtained post-transformation.
            //
            // Age Education    EducationCategory
            // 26  0-5yrs       Undergraduate 
            // 42  0-5yrs       Undergraudate 
            // 39  12+yrs       Postgraduate 
            // 34  0-5yrs       Undergraduate 
            // 35  6-11yrs      Postgraduate 
        }
    }
}
