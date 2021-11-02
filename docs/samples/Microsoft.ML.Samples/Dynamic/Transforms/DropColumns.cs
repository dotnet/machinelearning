using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class DropColumns
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<InputData>()
            {
                new InputData(){ Age = 21, Gender = "Male", Education = "BS",
                    ExtraColumn = 1 },

                new InputData(){ Age = 23, Gender = "Female", Education = "MBA",
                    ExtraColumn = 2 },

                new InputData(){ Age = 28, Gender = "Male", Education = "PhD",
                    ExtraColumn = 3 },

                new InputData(){ Age = 22, Gender = "Male", Education = "BS",
                    ExtraColumn = 4 },

                new InputData(){ Age = 23, Gender = "Female", Education = "MS",
                    ExtraColumn = 5 },

                new InputData(){ Age = 27, Gender = "Female", Education = "PhD",
                    ExtraColumn = 6 },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // Drop the ExtraColumn from the dataset.
            var pipeline = mlContext.Transforms.DropColumns("ExtraColumn");

            // Now we can transform the data and look at the output.
            // Don't forget that this operation doesn't actually operate on data
            // until we perform an action that requires 
            // the data to be materialized.
            var transformedData = pipeline.Fit(dataview).Transform(dataview);

            // Now let's take a look at what the DropColumns operations did.
            // We can extract the transformed data as an IEnumerable of InputData,
            // the class we define below. When we try to pull out the Age, Gender,
            // Education and ExtraColumn columns, ML.NET will raise an exception on
            // the ExtraColumn
            try
            {
                var failingRowEnumerable = mlContext.Data.CreateEnumerable<
                    InputData>(transformedData, reuseRowObject: false);
            }
            catch (ArgumentOutOfRangeException exception)
            {
                Console.WriteLine($"ExtraColumn is not available, so an exception" +
                    $" is thrown: {exception.Message}.");
            }

            // Expected output:
            //  ExtraColumn is not available, so an exception is thrown: Could not find  column 'ExtraColumn'.
            //  Parameter name: Schema

            // And we can write a few columns out to see that the rest of the data
            // is still available.
            var rowEnumerable = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false);

            Console.WriteLine($"The columns we didn't drop are still available.");
            foreach (var row in rowEnumerable)
                Console.WriteLine($"Age: {row.Age} Gender: {row.Gender} " +
                    $"Education: {row.Education}");

            // Expected output:
            //  The columns we didn't drop are still available.
            //  Age: 21 Gender: Male Education: BS
            //  Age: 23 Gender: Female Education: MBA
            //  Age: 28 Gender: Male Education: PhD
            //  Age: 22 Gender: Male Education: BS
            //  Age: 23 Gender: Female Education: MS
            //  Age: 27 Gender: Female Education: PhD
        }

        private class InputData
        {
            public int Age { get; set; }
            public string Gender { get; set; }
            public string Education { get; set; }
            public float ExtraColumn { get; set; }
        }

        private class TransformedData
        {
            public int Age { get; set; }
            public string Gender { get; set; }
            public string Education { get; set; }
        }
    }
}
