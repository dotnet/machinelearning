using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
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

        class SampleInfertDataWithInducedCategory
        {
            public float Age = 0;
            public float Induced = 0.0f;
            public string InducedCategory = default;
        }

        class SampleInfertDataWithIntArray
        {
            public float Age = 0;
            public string Education = default;
            public int[] EducationCategory = default;
        }


        public static void ValueMappingTransform()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = ml.CreateStreamingDataView(data);

            // Preview of the data.
            //
            // Age    Case  Education  induced     parity  pooled.stratum  row_num  ...
            // 26.0   1.0   0-5yrs      1.0         6.0       3.0      1.0  ...
            // 42.0   1.0   0-5yrs      1.0         1.0       1.0      2.0  ...
            // 39.0   1.0   12+yrs      2.0         6.0       4.0      3.0  ...
            // 34.0   1.0   0-5yrs      2.0         4.0       2.0      4.0  ...
            // 35.0   1.0   6-11yrs     1.0         3.0       32.0     5.0  ...

            StringToStringMappingExample(ml, trainData);
            FloatToStringMappingExample(ml, trainData);
            StringToKeyTypeMappingExample(ml, trainData);
            StringToArrayMappingExample(ml, trainData);
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
        public static void StringToStringMappingExample(MLContext ml, IDataView trainData)
        {
            // Creating a list of keys based on the Education values from the dataset
            // These lists are created by hand for the demonstration, but the ValueMappingEstimator does take an IEnumerable.
            var educationKeys = new List<string>()
            {
                "0-5yrs",
                "6-11yrs",
                "12+yrs"
            };

            var educationValues = new List<string>()
            {
                "Cat1",
                "Cat2", 
                "Cat3"
            };

            var pipeline = new ValueMappingEstimator<string, string>(ml, educationKeys, educationValues, ("Education", "EducationCategory"));

            // The transformed data.
            var transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the data of the newly created column as an IEnumerable of SampleInfertDataWithFeatures.
            var featuresColumn = ml.CreateEnumerable<SampleInfertDataWithFeatures>(transformedData, reuseRowObject: false);
            
            Console.WriteLine($"Example of mapping string->string");
            Console.WriteLine($"Age\tEducation\tEducationLabel");
            foreach (var featureRow in featuresColumn)
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

        ///<summary>
        /// This example demonstrates the use of KeyTypes by setting treatValuesAsKeyTypes to true, 
        /// <see cref="ValueMappingEstimator.ValueMappingEstimator(IHostEnvironment, IEnumerable{TKey}, IEnumerable{TValue}, bool, (string input, string output)[])")/> to true.
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
        /// </summary>
        public static void StringToKeyTypeMappingExample(MLContext ml, IDataView trainData)
        {
            // Creating a list of keys based on the Education values from the dataset
            // These lists are created by hand for the demonstration, but the ValueMappingEstimator does take an IEnumerable.
            var educationKeys = new List<string>()
            {
                "0-5yrs",
                "6-11yrs",
                "12+yrs"
            };

            // Sample string values
            var educationValues = new List<string>()
            {
                "Cat1",
                "Cat2", 
                "Cat3"
            };

            // Generate the ValueMappingEstimator that will output KeyTypes even though our values are strings.
            // The KeyToValueMappingEstimator is added to provide a reverse lookup of the KeyType, converting the KeyType value back
            // to the original value.
            var pipeline = new ValueMappingEstimator<string, string>(ml, educationKeys, educationValues, true, ("Education", "EducationKeyType"))
                              .Append(new KeyToValueMappingEstimator(ml, ("EducationKeyType", "EducationCategory")));

            // The transformed data.
            var transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the data of the newly created column as an IEnumerable of SampleInfertDataWithFeatures.
            var featuresColumn = ml.CreateEnumerable<SampleInfertDataWithFeatures>(transformedData, reuseRowObject: false);
            
            Console.WriteLine($"Example of mapping string->keytype");
            Console.WriteLine($"Age\tEducation\tEducationLabel");
            foreach (var featureRow in featuresColumn)
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

        ///<summary>
        /// This example demonstrates the use of floating types as the key type for ValueMappingEstimator by mapping a float-to-string value.
        /// The mapping looks like the following:
        /// <list>
        ///     <item>1.0 -> Cat1</item>
        ///     <item>2.0 -> Cat2</item>
        /// </list>
        /// </summary>
        public static void FloatToStringMappingExample(MLContext ml, IDataView trainData)
        {
            // Creating a list of keys based on the induced value from the dataset
            // These lists are created by hand for the demonstration, but the ValueMappingEstimator does take an IEnumerable.
            var inducedKeys = new List<float>()
            {
                1.0f, 
                2.0f
            };

            // Sample list of associated string values
            var inducedValues = new List<string>()
            {
                "Cat1",  
                "Cat2"
            };

            var pipeline = new ValueMappingEstimator<float, string>(ml, inducedKeys, inducedValues, ("Induced", "InducedCategory"));

            // The transformed data.
            var transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the data of the newly created column as an IEnumerable of SampleInfertDataWithFeatures.
            var featuresColumn = ml.CreateEnumerable<SampleInfertDataWithInducedCategory>(transformedData, reuseRowObject: false);
            
            Console.WriteLine($"Example of mapping float->string");
            Console.WriteLine($"Age\tInduced\tInducedCategory");
            foreach (var featureRow in featuresColumn)
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

        ///<summary>
        /// This example demonstrates the use arrays as the values for the ValueMappingEstimator. It maps a set of keys that are type string
        /// to a integer arrays of variable length.
        /// The mapping looks like the following:
        /// <list>
        ///     <item>0-5yrs -> 1,2,3,4</item>
        ///     <item>6-11yrs -> 5,6,7</item>
        ///     <item>12+yrs -> 42, 32</item>
        /// </list>
        /// </summary>
        public static void StringToArrayMappingExample(MLContext ml, IDataView trainData)
        {
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

            var pipeline = new ValueMappingEstimator<string, int>(ml, educationKeys, educationValues, ("Education", "EducationCategory"));

            // The transformed data.
            var transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Getting the data of the newly created column as an IEnumerable of SampleInfertDataWithFeatures.
            var featuresColumn = ml.CreateEnumerable<SampleInfertDataWithIntArray>(transformedData, reuseRowObject: false);
            
            Console.WriteLine($"Example of mapping string->array");
            Console.WriteLine($"Age\tEducation\tEducationLabel");
            foreach (var featureRow in featuresColumn)
            {
                Console.WriteLine($"{featureRow.Age}\t{featureRow.Education}  \t{string.Join(",", featureRow.EducationCategory)}");
            }

            // Features column obtained post-transformation.
            //
            // Example of mapping string->array
            // Age     Education   EducationLabel
            // 26      0 - 5yrs    1,2,3,4
            // 42      0 - 5yrs    1,2,3,4
            // 39      12 + yrs    42,32
            // 34      0 - 5yrs    1,2,3,4
            // 35      6 - 11yrs   5,6,7
        }
    }
}
