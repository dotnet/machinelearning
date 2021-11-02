using System;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class FilterRowsByStatefulCustomPredicate
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
                new InputData() { Name = "Joey", FilterNext = false },
                new InputData() { Name = "Chandler", FilterNext = false },
                new InputData() { Name = "Ross", FilterNext = false },
                new InputData() { Name = "Monica", FilterNext = true },
                new InputData() { Name = "Rachel", FilterNext = true },
                new InputData() { Name = "Phoebe", FilterNext = false },
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
            var filteredData = mlContext.Data.FilterByStatefulCustomPredicate<InputData, State>(
                data, (input, state) =>
                {
                    var filter = state.Filter;
                    state.Filter = input.FilterNext;
                    return (filter && input.Name.StartsWith("r", StringComparison.OrdinalIgnoreCase));
                }, state => state.Filter = false);

            // Look at the filtered data and observe that names starting with "R" have been dropped,
            // but only those where the FilterNext field in the previous example is true.
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
            //  Ross
            //  Monica
            //  Phoebe
        }

        private class InputData
        {
            public string Name { get; set; }
            public bool FilterNext { get; set; }
        }

        private class State
        {
            public bool Filter { get; set; }
        }
    }
}
