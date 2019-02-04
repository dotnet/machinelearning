using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public class SelectColumns
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and them read it as ML.NET's data type.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = mlContext.Data.ReadFromEnumerable(data);

            // Preview of the data.
            //
            // Age    Case  Education  induced     parity  pooled.stratum  row_num  ...
            // 26.0   1.0   0-5yrs      1.0         6.0       3.0      1.0  ...
            // 42.0   1.0   0-5yrs      1.0         1.0       1.0      2.0  ...
            // 39.0   1.0   0-5yrs      2.0         6.0       4.0      3.0  ...
            // 34.0   1.0   0-5yrs      2.0         4.0       2.0      4.0  ...
            // 35.0   1.0   6-11yrs     1.0         3.0       32.0     5.0  ...

            // Select a subset of columns to keep.
            var pipeline = mlContext.Transforms.SelectColumns(new string[] { "Age", "Education" });

            // Now we can transform the data and look at the output to confirm the behavior of CopyColumns.
            // Don't forget that this operation doesn't actually evaluate data until we read the data below,
            // as transformations are lazy in ML.NET.
            var transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Print the number of columns in the schema
            Console.WriteLine($"There are {transformedData.Schema.Count} columns in the dataset.");

            // Expected output:
            //  There are 2 columns in the dataset.

            // We can extract the newly created column as an IEnumerable of SampleInfertDataTransformed, the class we define below.
            var rowEnumerable = mlContext.CreateEnumerable<SampleInfertDataTransformed>(transformedData, reuseRowObject: false);

            // And finally, we can write out the rows of the dataset, looking at the columns of interest.
            Console.WriteLine($"Age and Educations columns obtained post-transformation.");
            foreach (var row in rowEnumerable)
            {
                Console.WriteLine($"Age: {row.Age} Education: {row.Education}");
            }

            // Expected output:
            //  Age and Education columns obtained post-transformation.
            //  Age: 26 Education: 0 - 5yrs
            //  Age: 42 Education: 0 - 5yrs
            //  Age: 39 Education: 0 - 5yrs
            //  Age: 34 Education: 0 - 5yrs
            //  Age: 35 Education: 6 - 11yrs
        }

        private class SampleInfertDataTransformed
        {
            public float Age { get; set; }
            public string Education { get; set; }
        }
    }
}
