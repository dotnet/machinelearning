using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class SelectFeaturesBasedOnCount
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var rawData = GetData();
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            var convertedData = mlContext.Data.CreateEnumerable<InputData>(data, true);

            Console.WriteLine("Contents of two columns 'GroupA' and 'Info'.");
            foreach (var item in convertedData)
                Console.WriteLine("{0}\t\t\t{1}", string.Join("\t", item.GroupA), string.Join("\t", item.Info));
            // 4       NaN     6                       A       WA   Male
            // 4       5       6                       A            Female
            // 4       5       6                       A       NY
            // 4       NaN     NaN                     A            Male

            // We will use the SelectFeaturesBasedOnCount to retain only those slots which have at least 'count' non-default values per slot.

            // Usage on numeric column.
            var pipeline = mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(
                outputColumnName: "FeaturesSelectedGroupA", inputColumnName: "GroupA", count: 3);

            // The pipeline can then be trained, using .Fit(), and the resulting transformer can be used to transform data. 
            var transformedData = pipeline.Fit(data).Transform(data);

            Console.WriteLine("Contents of column 'FeaturesSelectedGroupA'");
            var featuresSelectedGroupA = transformedData.GetColumn<float[]>(transformedData.Schema["FeaturesSelectedGroupA"]);
            foreach (var row in featuresSelectedGroupA)
            {
                for (var i = 0; i < row.Length; i++)
                    Console.Write($"{row[i]}\t");
                Console.WriteLine();
            }
            // 4       6
            // 4       6
            // 4       6
            // 4       NaN

            // Usage on text column.
            pipeline = mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(
                outputColumnName: "FeaturesSelectedInfo", inputColumnName: "Info", count: 3);

            transformedData = pipeline.Fit(data).Transform(data);

            Console.WriteLine("Contents of column 'FeaturesSelectedInfo'");
            var featuresSelectedInfo = transformedData.GetColumn<string[]>(transformedData.Schema["FeaturesSelectedInfo"]);
            foreach (var row in featuresSelectedInfo)
            {
                for (var i = 0; i < row.Length; i++)
                    Console.Write($"{row[i]}\t");
                Console.WriteLine();
            }
            // A       Male
            // A       Female
            // A
            // A       Male
        }

        public class InputData
        {
            [VectorType(3)]
            public float[] GroupA { get; set; }

            [VectorType(3)]
            public string[] Info { get; set; }
        }

        /// <summary>
        /// Return a few rows of data.
        /// </summary>
        public static IEnumerable<InputData> GetData()
        {
            var data = new List<InputData>
            {
                new InputData
                {
                    GroupA = new float[] { 4, float.NaN, 6 },
                    Info = new string[] { "A", "WA", "Male"}
                },
                new InputData
                {
                    GroupA = new float[] { 4, 5, 6 },
                    Info = new string[] { "A", "", "Female"}
                },
                new InputData
                {
                    GroupA = new float[] { 4, 5, 6 },
                    Info = new string[] { "A", "NY", null}
                },
                new InputData
                {
                    GroupA = new float[] { 4, float.NaN, float.NaN },
                    Info = new string[] { "A", null, "Male"}
                }
            };
            return data;
        }
    }
}
