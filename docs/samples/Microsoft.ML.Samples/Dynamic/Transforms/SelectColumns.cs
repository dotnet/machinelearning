using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class SelectColumns
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

            // Select a subset of columns to keep.
            var pipeline = mlContext.Transforms.SelectColumns("Age", "Education");

            // Now we can transform the data and look at the output to confirm the
            // behavior of SelectColumns. Don't forget that this operation doesn't
            // actually evaluate data until we read the data below, as
            // transformations are lazy in ML.NET.
            var transformedData = pipeline.Fit(dataview).Transform(dataview);

            // Print the number of columns in the schema
            Console.WriteLine($"There are {transformedData.Schema.Count} columns" +
                $" in the dataset.");

            // Expected output:
            //  There are 2 columns in the dataset.

            // We can extract the newly created column as an IEnumerable of
            // TransformedData, the class we define below.
            var rowEnumerable = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false);

            // And finally, we can write out the rows of the dataset, looking at the
            // columns of interest.
            Console.WriteLine($"Age and Educations columns obtained " +
                $"post-transformation.");

            foreach (var row in rowEnumerable)
                Console.WriteLine($"Age: {row.Age} Education: {row.Education}");

            // Expected output:
            //  Age and Educations columns obtained post-transformation.
            //  Age: 21 Education: BS
            //  Age: 23 Education: MBA
            //  Age: 28 Education: PhD
            //  Age: 22 Education: BS
            //  Age: 23 Education: MS
            //  Age: 27 Education: PhD
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
            public string Education { get; set; }
        }
    }
}
