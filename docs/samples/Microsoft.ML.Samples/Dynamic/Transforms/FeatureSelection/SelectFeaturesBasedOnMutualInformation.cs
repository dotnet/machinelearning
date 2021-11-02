using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class SelectFeaturesBasedOnMutualInformation
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var rawData = GetData();

            // Printing the columns of the input data. 
            Console.WriteLine($"Label             NumericVector");
            foreach (var item in rawData)
                Console.WriteLine("{0,-25} {1,-25}", item.Label, string.Join(",",
                    item.NumericVector));

            // Label                       NumericVector
            // True                        4,0,6
            // False                       0,5,7
            // True                        4,0,6
            // False                       0,5,7

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // We define a MutualInformationFeatureSelectingEstimator that selects
            // the top k slots in a feature vector based on highest mutual
            // information between that slot and a specified label. 
            var pipeline = mlContext.Transforms.FeatureSelection
                .SelectFeaturesBasedOnMutualInformation(outputColumnName:
                "NumericVector", labelColumnName: "Label", slotsInOutput: 2);

            // The pipeline can then be trained, using .Fit(), and the resulting
            // transformer can be used to transform data. 
            var transformedData = pipeline.Fit(data).Transform(data);

            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, true);

            // Printing the columns of the transformed data. 
            Console.WriteLine($"NumericVector");
            foreach (var item in convertedData)
                Console.WriteLine("{0,-25}", string.Join(",", item.NumericVector));

            // NumericVector
            // 4,0
            // 0,5
            // 4,0
            // 0,5
        }

        public class TransformedData
        {
            public float[] NumericVector { get; set; }
        }

        public class NumericData
        {
            public bool Label;

            [VectorType(3)]
            public float[] NumericVector { get; set; }
        }

        /// <summary>
        /// Returns a few rows of numeric data.
        /// </summary>
        public static IEnumerable<NumericData> GetData()
        {
            var data = new List<NumericData>
            {
                new NumericData
                {
                    Label = true,
                    NumericVector = new float[] { 4, 0, 6 },
                },
                new NumericData
                {
                    Label = false,
                    NumericVector = new float[] { 0, 5, 7 },
                },
                new NumericData
                {
                    Label = true,
                    NumericVector = new float[] { 4, 0, 6 },
                },
                new NumericData
                {
                    Label = false,
                    NumericVector = new float[] { 0, 5, 7 },
                }
            };
            return data;
        }
    }
}
