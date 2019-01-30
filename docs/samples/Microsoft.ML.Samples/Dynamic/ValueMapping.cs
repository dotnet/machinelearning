using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Conversions;

namespace Microsoft.ML.Samples.Dynamic
{
    public class ValueMappingExample
    {
        class SampleInfertDataWithFeatures
        {
            public float Age = 0;
            public string Education = default;
            public string EducationCategory = default;
        }

        ///<summary>
        /// This example demonstrates the use of the ValueMappingEstimator by mapping string-to-string values. The ValueMappingEstimator uses
        /// level of education as keys to a respective string label which is the value.
        /// The mapping looks like the following:
        /// <list>
        ///     <item>0-5yrs -> Cat1</item>
        ///     <item>6-11yrs -> Cat2</item>
        ///     <item>12+yrs -> Cat3</item>
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

            // Preview of the data.
            //
            // Age    Case  Education  induced     parity  pooled.stratum  row_num  ...
            // 26.0   1.0   0-5yrs      1.0         6.0       3.0      1.0  ...
            // 42.0   1.0   0-5yrs      1.0         1.0       1.0      2.0  ...
            // 39.0   1.0   12+yrs      2.0         6.0       4.0      3.0  ...
            // 34.0   1.0   0-5yrs      2.0         4.0       2.0      4.0  ...
            // 35.0   1.0   6-11yrs     1.0         3.0       32.0     5.0  ...

            // Creating a list of keys based on the Education values from the dataset
            // These lists are created by hand for the demonstration, but the ValueMappingEstimator does take an IEnumerable.
            var educationKeys = new List<string>()
            {
                "0-5yrs",
                "6-11yrs",
                "12+yrs"
            };

            // Creating a list of associated values that will map respectively to each educationKey
            var educationValues = new List<string>()
            {
                "Cat1",
                "Cat2", 
                "Cat3"
            };

            // Constructs the ValueMappingEstimator making the ML.net pipeline
            var pipeline = new ValueMappingEstimator<string, string>(ml, educationKeys, educationValues, ("EducationCategory", "Education"));

            // Fits the ValueMappingEstimator and transforms the data converting the Education to EducationCategory.
            IDataView transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the resulting data as an IEnumerable of SampleInfertDataWithFeatures. This will contain the newly created column EducationCategory
            IEnumerable<SampleInfertDataWithFeatures> featureRows = ml.CreateEnumerable<SampleInfertDataWithFeatures>(transformedData, reuseRowObject: false);
            
            Console.WriteLine($"Example of mapping string->string");
            Console.WriteLine($"Age\tEducation\tEducationLabel");
            foreach (var featureRow in featureRows)
            {
                Console.WriteLine($"{featureRow.Age}\t{featureRow.Education}  \t{featureRow.EducationCategory}");
            }

            // Features column obtained post-transformation.
            //
            // Age Education    EducationLabel
            // 26  0-5yrs       Cat1
            // 42  0-5yrs       Cat1
            // 39  12+yrs       Cat3
            // 34  0-5yrs       Cat1
            // 35  6-11yrs      Cat2
        }
    }
}
