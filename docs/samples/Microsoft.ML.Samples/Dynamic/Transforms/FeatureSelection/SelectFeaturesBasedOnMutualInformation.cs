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
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var rawData = GetData();

            Console.WriteLine("Contents of two columns 'Label' and 'GroupA'.");
            foreach (var item in rawData)
                Console.WriteLine("{0}\t\t{1}", item.Label, string.Join(" ", item.GroupA));
            // True            4 0 6
            // False           0 5 7
            // True            4 0 6
            // False           0 5 7

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // We define a MutualInformationFeatureSelectingEstimator that selects the top k slots in a feature 
            // vector based on highest mutual information between that slot and a specified label. 

            var pipeline = mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation(
                outputColumnName: "FeaturesSelectedGroupA", inputColumnName: "GroupA", labelColumnName: "Label",
                slotsInOutput:2);

            // The pipeline can then be trained, using .Fit(), and the resulting transformer can be used to transform data. 
            var transformedData = pipeline.Fit(data).Transform(data);

            Console.WriteLine("Contents of column 'FeaturesSelectedGroupA'");
            PrintDataColumn(transformedData, "FeaturesSelectedGroupA");
            // 4 0
            // 0 5
            // 4 0
            // 0 5
        }

        private static void PrintDataColumn(IDataView transformedData, string columnName)
        {
            var countSelectColumn = transformedData.GetColumn<float[]>(transformedData.Schema[columnName]);

            foreach (var row in countSelectColumn)
            {
                for (var i = 0; i < row.Length; i++)
                    Console.Write($"{row[i]} ");
                Console.WriteLine();
            }
        }

        public class NumericData
        {
            public bool Label;

            [VectorType(3)]
            public float[] GroupA { get; set; }
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
                    GroupA = new float[] { 4, 0, 6 },
                },
                new NumericData
                {
                    Label = false,
                    GroupA = new float[] { 0, 5, 7 },
                },
                new NumericData
                {
                    Label = true,
                    GroupA = new float[] { 4, 0, 6 },
                },
                new NumericData
                {
                    Label = false,
                    GroupA = new float[] { 0, 5, 7 },
                }
            };
            return data;
        }
    }
}
