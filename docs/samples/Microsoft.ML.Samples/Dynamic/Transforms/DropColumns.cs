using System;
using System.Collections.Generic;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class DropColumns
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and them read it as ML.NET's data type.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = mlContext.Data.LoadFromEnumerable(data);

            // Preview of the data.
            //
            // Age    Case  Education  Induced     Parity  Pooled.stratum  Row_num  ...
            // 26.0   1.0   0-5yrs      1.0         6.0       3.0      1.0  ...
            // 42.0   1.0   0-5yrs      1.0         1.0       1.0      2.0  ...
            // 39.0   1.0   0-5yrs      2.0         6.0       4.0      3.0  ...
            // 34.0   1.0   0-5yrs      2.0         4.0       2.0      4.0  ...
            // 35.0   1.0   6-11yrs     1.0         3.0       32.0     5.0  ...

            // Drop the Age and Education columns from the dataset.
            var pipeline = mlContext.Transforms.DropColumns("Age", "Education");

            // Now we can transform the data and look at the output.
            // Don't forget that this operation doesn't actually operate on data until we perform an action that requires 
            // the data to be materialized.
            var transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Now let's take a look at what the DropColumns operations did.
            // We can extract the transformed data as an IEnumerable of SampleInfertDataNonExistentColumns, the class we define below.
            // When we try to pull out the Age and Education columns, ML.NET will raise an exception on the first non-existent column
            // that it tries to access. 
            try
            {
                var failingRowEnumerable = mlContext.Data.CreateEnumerable<SampleInfertDataNonExistentColumns>(transformedData, reuseRowObject: false);
            } catch(ArgumentOutOfRangeException exception)
            {
                Console.WriteLine($"Age and Education were not available, so an exception was thrown: {exception.Message}.");
            }

            // Expected output:
            //  Age and Education were not available, so an exception was thrown: Could not find  column 'Age'.
            //  Parameter name: Schema

            // And we can write a few columns out to see that the rest of the data is still available.
            var rowEnumerable = mlContext.Data.CreateEnumerable<SampleInfertDataTransformed>(transformedData, reuseRowObject: false);
            Console.WriteLine($"The columns we didn't drop are still available.");
            foreach (var row in rowEnumerable)
            {
                Console.WriteLine($"Case: {row.Case} Induced: {row.Induced} Parity: {row.Parity}");
            }

            // Expected output:
            //  The columns we didn't drop are still available.
            //  Case: 1 Induced: 1 Parity: 6
            //  Case: 1 Induced: 1 Parity: 1
            //  Case: 1 Induced: 2 Parity: 6
            //  Case: 1 Induced: 2 Parity: 4
            //  Case: 1 Induced: 1 Parity: 3
        }

        private class SampleInfertDataNonExistentColumns
        {
            public float Age { get; set; }
            public float Education { get; set; }
        }

        private class SampleInfertDataTransformed
        {
            public float Case { get; set; }
            public float Induced { get; set; }
            public float Parity { get; set; }
        }
    }
}
