using System;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class FilterRowsByCustomPredicate
    {
        // Sample class showing how to filter out some rows in IDataView using a custom filter function.
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available
            // operations and as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var enumerableOfData = new[]
            {
                new InputData() { Name = "Joey" },
                new InputData() { Name = "Chandler" },
                new InputData() { Name = "Ross" },
                new InputData() { Name = "Monica" },
                new InputData() { Name = "Rachel" },
                new InputData() { Name = "Phoebe" },
            };

            var data = mlContext.Data.LoadFromEnumerable(enumerableOfData);

            // Before we apply a filter, examine all the records in the dataset.
            Console.WriteLine("Name");
            foreach (var row in enumerableOfData)
            {
                Console.WriteLine(row.Name);
            }
            Console.WriteLine();

            // Expected output:

            //  Name
            //  Joey
            //  Chandler
            //  Ross
            //  Monica
            //  Rachel
            //  Phoebe

            // Filter the data by using a custom filter.
            var filteredData = mlContext.Data.FilterByCustomPredicate<InputData>(
                data, input => input.Name.StartsWith("r", StringComparison.OrdinalIgnoreCase));

            // Look at the filtered data and observe that names starting with "R" have been dropped.
            var enumerable = mlContext.Data
                .CreateEnumerable<InputData>(filteredData,
                reuseRowObject: true);

            Console.WriteLine("Name");
            foreach (var row in enumerable)
            {
                Console.WriteLine(row.Name);
            }

            // Expected output:

            //  Name
            //  Joey
            //  Chandler
            //  Monica
            //  Phoebe
        }

        private class InputData
        {
            public string Name { get; set; }
        }
    }
}

