using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    // This example illustrates how to convert multiple columns of different types
    // to one type, in this case System.Single. 
    // This is often a useful data transformation before concatenating the features
    // together and passing them to a particular estimator.
    public static class ConvertTypeMultiColumn
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext(seed: 1);

            var rawData = new[] {
                new InputData() { Feature1 = true, Feature2 = "0.4",
                    Feature3 = DateTime.Now, Feature4 = 0.145},

                new InputData() { Feature1 = false, Feature2 = "0.5",
                    Feature3 = DateTime.Today, Feature4 = 3.14},

                new InputData() { Feature1 = false, Feature2 = "14",
                    Feature3 = DateTime.Today, Feature4 = 0.2046},

                new InputData() { Feature1 = false, Feature2 = "23",
                    Feature3 = DateTime.Now, Feature4 = 0.1206},

                new InputData() { Feature1 = true, Feature2 = "8904",
                    Feature3 = DateTime.UtcNow, Feature4 = 8.09},
            };

            // Convert the data to an IDataView.
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Construct the pipeline.
            var pipeline = mlContext.Transforms.Conversion.ConvertType(new[]
            {
                    new InputOutputColumnPair("Converted1", "Feature1"),
                    new InputOutputColumnPair("Converted2", "Feature2"),
                    new InputOutputColumnPair("Converted3", "Feature3"),
                    new InputOutputColumnPair("Converted4", "Feature4"),
             },
             DataKind.Single);

            // Let's fit our pipeline to the data.
            var transformer = pipeline.Fit(data);
            // Transforming the same data. This will add the 4 columns defined in
            // the pipeline, containing the converted
            // values of the initial columns. 
            var transformedData = transformer.Transform(data);

            // Shape the transformed data as a strongly typed IEnumerable.
            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, true);

            // Printing the results.
            Console.WriteLine("Converted1\t Converted2\t Converted3\t Converted4");
            foreach (var item in convertedData)
                Console.WriteLine($"\t{item.Converted1}\t {item.Converted2}\t\t  " +
                    $"{item.Converted3}\t {item.Converted4}");

            // Transformed data.
            //
            // Converted1   Converted2    Converted3     Converted4
            //      1        0.4        6.368921E+17        0.145
            //      0        0.5        6.368916E+17        3.14
            //      0        14         6.368916E+17        0.2046
            //      0        23         6.368921E+17        0.1206
            //      1       8904        6.368924E+17        8.09

        }

        // The initial data type
        private class InputData
        {
            public bool Feature1;
            public string Feature2;
            public DateTime Feature3;
            public double Feature4;
        }

        // The resulting data type after the transformation
        private class TransformedData : InputData
        {
            public float Converted1 { get; set; }
            public float Converted2 { get; set; }
            public float Converted3 { get; set; }
            public float Converted4 { get; set; }
        }
    }
}
