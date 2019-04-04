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
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var rawData = GetData();

            Console.WriteLine("Contents of columns 'Label', 'GroupA' and 'GroupB'.");
            foreach (var item in rawData)
                Console.WriteLine("{0}\t\t{1}\t\t{2}", item.Label, string.Join(" ", item.GroupA), string.Join(" ", item.GroupB));
            // True            4 0 6           7 8 9
            // False           0 5 7           7 9 0
            // True            4 0 6           7 8 9
            // False           0 5 7           7 8 0

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // We define a MutualInformationFeatureSelectingEstimator that selects the top k slots in a feature 
            // vector based on highest mutual information between that slot and a specified label. 

            // Multi column example : This pipeline transform two columns using the provided parameters.
            var pipeline = mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation(
                new InputOutputColumnPair[] { new InputOutputColumnPair("GroupA"), new InputOutputColumnPair("GroupB") },
                labelColumnName: "Label",
                slotsInOutput: 4);

            var transformedData = pipeline.Fit(data).Transform(data);

            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, true);
            Console.WriteLine("Contents of two columns 'GroupA' and 'GroupB'.");
            foreach (var item in convertedData)
                Console.WriteLine("{0}\t\t{1}", string.Join(" ", item.GroupA), string.Join(" ", item.GroupB));

            // Here, we see SelectFeaturesBasedOnMutualInformation selected 4 slots. (3 slots from the 'GroupB' column and 1 slot from the 'GroupC' column.)
            // 4 0 6           9
            // 0 5 7           0
            // 4 0 6           9
            // 0 5 7           0
        }

        private class TransformedData
        {
            public float[] GroupA { get; set; }

            public float[] GroupB { get; set; }
        }

        public class NumericData
        {
            public bool Label;

            [VectorType(3)]
            public float[] GroupA { get; set; }

            [VectorType(3)]
            public float[] GroupB { get; set; }
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
                    GroupB = new float[] { 7, 8, 9 },
                },
                new NumericData
                {
                    Label = false,
                    GroupA = new float[] { 0, 5, 7 },
                    GroupB = new float[] { 7, 9, 0 },
                },
                new NumericData
                {
                    Label = true,
                    GroupA = new float[] { 4, 0, 6 },
                    GroupB = new float[] { 7, 8, 9 },
                },
                new NumericData
                {
                    Label = false,
                    GroupA = new float[] { 0, 5, 7 },
                    GroupB = new float[] { 7, 8, 0 },
                }
            };
            return data;
        }
    }
}
