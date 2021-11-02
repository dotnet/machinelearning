using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class SelectFeaturesBasedOnCountMultiColumn
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var rawData = GetData();

            // Printing the columns of the input data. 
            Console.WriteLine($"NumericVector             StringVector");
            foreach (var item in rawData)
                Console.WriteLine("{0,-25} {1,-25}", string.Join(",", item.
                    NumericVector), string.Join(",", item.StringVector));

            // NumericVector             StringVector
            // 4,NaN,6                   A,WA,Male
            // 4,5,6                     A,,Female
            // 4,5,6                     A,NY,
            // 4,NaN,NaN                 A,,Male

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // We will use the SelectFeaturesBasedOnCount transform estimator, to
            // retain only those slots which have at least 'count' non-default
            // values per slot.

            // Multi column example. This pipeline transform two columns using the
            // provided parameters.
            var pipeline = mlContext.Transforms.FeatureSelection
                .SelectFeaturesBasedOnCount(new InputOutputColumnPair[] { new
                InputOutputColumnPair("NumericVector"), new InputOutputColumnPair(
                "StringVector") }, count: 3);

            var transformedData = pipeline.Fit(data).Transform(data);

            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, true);

            // Printing the columns of the transformed data. 
            Console.WriteLine($"NumericVector             StringVector");
            foreach (var item in convertedData)
                Console.WriteLine("{0,-25} {1,-25}", string.Join(",", item
                    .NumericVector), string.Join(",", item.StringVector));

            // NumericVector             StringVector
            // 4,6                       A,Male
            // 4,6                       A,Female
            // 4,6                       A,
            // 4,NaN                     A,Male
        }

        private class TransformedData
        {
            public float[] NumericVector { get; set; }

            public string[] StringVector { get; set; }
        }

        public class InputData
        {
            [VectorType(3)]
            public float[] NumericVector { get; set; }

            [VectorType(3)]
            public string[] StringVector { get; set; }
        }

        /// <summary>
        /// Returns a few rows of data.
        /// </summary>
        public static IEnumerable<InputData> GetData()
        {
            var data = new List<InputData>
            {
                new InputData
                {
                    NumericVector = new float[] { 4, float.NaN, 6 },
                    StringVector = new string[] { "A", "WA", "Male"}
                },
                new InputData
                {
                    NumericVector = new float[] { 4, 5, 6 },
                    StringVector = new string[] { "A", "", "Female"}
                },
                new InputData
                {
                    NumericVector = new float[] { 4, 5, 6 },
                    StringVector = new string[] { "A", "NY", null}
                },
                new InputData
                {
                    NumericVector = new float[] { 4, float.NaN, float.NaN },
                    StringVector = new string[] { "A", null, "Male"}
                }
            };
            return data;
        }
    }
}
