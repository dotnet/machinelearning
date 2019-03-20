using System;
namespace Microsoft.ML.Samples.Dynamic
{
    public static class SelectColumns
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and them read it as ML.NET's data type.
            var enumerableData = SamplesUtils.DatasetUtils.GetInfertData();
            var data = mlContext.Data.LoadFromEnumerable(enumerableData);

            // Before transformation, take a look at the dataset
            Console.WriteLine($"Age\tCase\tEducation\tInduced\tParity\tPooledStratum");
            foreach (var row in enumerableData)
            {
                Console.WriteLine($"{row.Age}\t{row.Case}\t{row.Education}\t{row.Induced}\t{row.Parity}\t{row.PooledStratum}");
            }
            Console.WriteLine();
            // Expected output:
            //  Age     Case    Education       Induced Parity  PooledStratum
            //  26      1       0 - 5yrs        1       6       3
            //  42      1       0 - 5yrs        1       1       1
            //  39      1       12 + yrs        2       6       4
            //  34      1       0 - 5yrs        2       4       2
            //  35      1       6 - 11yrs       1       3       32

            // Select a subset of columns to keep.
            var pipeline = mlContext.Transforms.SelectColumns("Age", "Education");

            // Now we can transform the data and look at the output to confirm the behavior of CopyColumns.
            // Don't forget that this operation doesn't actually evaluate data until we read the data below,
            // as transformations are lazy in ML.NET.
            var transformedData = pipeline.Fit(data).Transform(data);

            // Print the number of columns in the schema
            Console.WriteLine($"There are {transformedData.Schema.Count} columns in the dataset.");

            // Expected output:
            //  There are 2 columns in the dataset.

            // We can extract the newly created column as an IEnumerable of SampleInfertDataTransformed, the class we define below.
            var rowEnumerable = mlContext.Data.CreateEnumerable<SampleInfertDataTransformed>(transformedData, reuseRowObject: false);

            // And finally, we can write out the rows of the dataset, looking at the columns of interest.
            Console.WriteLine($"Age and Educations columns obtained post-transformation.");
            foreach (var row in rowEnumerable)
            {
                Console.WriteLine($"Age: {row.Age} Education: {row.Education}");
            }

            // Expected output:
            //  Age and Education columns obtained post-transformation.
            //  Age: 26 Education: 0-5yrs
            //  Age: 42 Education: 0-5yrs
            //  Age: 39 Education: 12+yrs
            //  Age: 34 Education: 0-5yrs
            //  Age: 35 Education: 6-11yrs
        }

        private class SampleInfertDataTransformed
        {
            public float Age { get; set; }
            public string Education { get; set; }
        }
    }
}
