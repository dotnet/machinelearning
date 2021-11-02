using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class SelectFeaturesBasedOnMutualInformationMultiColumn
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var rawData = GetData();

            // Printing the columns of the input data. 
            Console.WriteLine($"NumericVectorA            NumericVectorB");
            foreach (var item in rawData)
                Console.WriteLine("{0,-25} {1,-25}", string.Join(",", item
                    .NumericVectorA), string.Join(",", item.NumericVectorB));

            // NumericVectorA              NumericVectorB
            // 4,0,6                       7,8,9
            // 0,5,7                       7,9,0
            // 4,0,6                       7,8,9
            // 0,5,7                       7,8,0

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // We define a MutualInformationFeatureSelectingEstimator that selects
            // the top k slots in a feature vector based on highest mutual
            // information between that slot and a specified label. 

            // Multi column example : This pipeline transform two columns using the
            // provided parameters.
            var pipeline = mlContext.Transforms.FeatureSelection
                .SelectFeaturesBasedOnMutualInformation(new InputOutputColumnPair[]
                { new InputOutputColumnPair("NumericVectorA"), new
                InputOutputColumnPair("NumericVectorB") }, labelColumnName: "Label",
                slotsInOutput: 4);

            var transformedData = pipeline.Fit(data).Transform(data);

            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, true);

            // Printing the columns of the transformed data. 
            Console.WriteLine($"NumericVectorA            NumericVectorB");
            foreach (var item in convertedData)
                Console.WriteLine("{0,-25} {1,-25}", string.Join(",", item
                    .NumericVectorA), string.Join(",", item.NumericVectorB));

            // NumericVectorA              NumericVectorB
            // 4,0,6                       9
            // 0,5,7                       0
            // 4,0,6                       9
            // 0,5,7                       0
        }

        private class TransformedData
        {
            public float[] NumericVectorA { get; set; }

            public float[] NumericVectorB { get; set; }
        }

        public class NumericData
        {
            public bool Label;

            [VectorType(3)]
            public float[] NumericVectorA { get; set; }

            [VectorType(3)]
            public float[] NumericVectorB { get; set; }
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
                    NumericVectorA = new float[] { 4, 0, 6 },
                    NumericVectorB = new float[] { 7, 8, 9 },
                },
                new NumericData
                {
                    Label = false,
                    NumericVectorA = new float[] { 0, 5, 7 },
                    NumericVectorB = new float[] { 7, 9, 0 },
                },
                new NumericData
                {
                    Label = true,
                    NumericVectorA = new float[] { 4, 0, 6 },
                    NumericVectorB = new float[] { 7, 8, 9 },
                },
                new NumericData
                {
                    Label = false,
                    NumericVectorA = new float[] { 0, 5, 7 },
                    NumericVectorB = new float[] { 7, 8, 0 },
                }
            };
            return data;
        }
    }
}
