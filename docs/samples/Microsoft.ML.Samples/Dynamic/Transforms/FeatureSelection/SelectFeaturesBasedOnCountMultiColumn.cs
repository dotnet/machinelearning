using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class SelectFeaturesBasedOnCountMultiColumn
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var rawData = GetData();
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            Console.WriteLine("Contents of column 'GroupB'");
            PrintDataColumn(data, "GroupB");
            // 4       NaN     6
            // 4       5       6
            // 4       5       6
            // 4       NaN     NaN

            Console.WriteLine("Contents of column 'GroupC'");
            PrintDataColumn(data, "GroupC");
            // NaN     8       9
            // NaN     8       9
            // NaN     8       9
            // 7       8       9

            // Second, we define the transformations that we apply on the data. Remember that an Estimator does not transform data
            // directly, but it needs to be trained on data using .Fit(), and it will output a Transformer, which can transform data.

            // We will use the SelectFeaturesBasedOnCount transform estimator, to retain only those slots which have 
            // at least 'count' non-default values per slot.

            // Multi column example : This pipeline uses two columns for transformation
            var pipeline = mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(
                new InputOutputColumnPair[] { new InputOutputColumnPair("GroupB"), new InputOutputColumnPair("GroupC") },
                count: 3);

            var transformedData = pipeline.Fit(data).Transform(data);

            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, true);
            Console.WriteLine("Contents of two columns 'GroupB' and 'GroupC'.");
            foreach (var item in convertedData)
                Console.WriteLine("{0}\t\t{1}", string.Join(" ", item.GroupB), string.Join(" ", item.GroupC));
            // 4 6             8 9
            // 4 6             8 9
            // 4 6             8 9
            // 4 NaN           8 9
        }

        private static void PrintDataColumn(IDataView transformedData, string columnName)
        {
            var countSelectColumn = transformedData.GetColumn<float[]>(transformedData.Schema[columnName]);

            foreach (var row in countSelectColumn)
            {
                for (var i = 0; i < row.Length; i++)
                    Console.Write($"{row[i]}\t");
                Console.WriteLine();
            }
        }

        private class TransformedData
        {
            public float[] GroupB { get; set; }

            public float[] GroupC { get; set; }
        }

        public class NumericData
        {
            public bool Label;

            [VectorType(3)]
            public float[] GroupA { get; set; }

            [VectorType(3)]
            public float[] GroupB { get; set; }

            [VectorType(3)]
            public float[] GroupC { get; set; }
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
                    GroupA = new float[] { 1, 2, 3 },
                    GroupB = new float[] { 4, float.NaN, 6 },
                    GroupC = new float[] { float.NaN, 8, 9 },
                },
                new NumericData
                {
                    Label = false,
                    GroupA = new float[] { 1, 2, 3 },
                    GroupB = new float[] { 4, 5, 6 },
                    GroupC = new float[] { float.NaN, 8, 9 },
                },
                new NumericData
                {
                    Label = true,
                    GroupA = new float[] { 1, 2, 3 },
                    GroupB = new float[] { 4, 5, 6 },
                    GroupC = new float[] { float.NaN, 8, 9 },
                },
                new NumericData
                {
                    Label = false,
                    GroupA = new float[] { 1, 2, 3 },
                    GroupB = new float[] { 4, float.NaN, float.NaN },
                    GroupC = new float[] { 7, 8, 9 },
                }
            };
            return data;
        }
    }
}
