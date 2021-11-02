using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    /// This example demonstrates the use of the ValueToKeyMappingEstimator, by
    /// mapping KeyType values to the original strings. For more on ML.NET KeyTypes
    /// see: https://github.com/dotnet/machinelearning/blob/main/docs/code/IDataViewTypeSystem.md#key-types
    public class MapKeyToValueMultiColumn
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);
            // Get a small dataset as an IEnumerable.

            // Create a list of data examples.
            var examples = GenerateRandomDataPoints(1000, 10);

            // Convert the examples list to an IDataView object, which is consumable
            // by ML.NET API.
            var dataView = mlContext.Data.LoadFromEnumerable(examples);

            // Create a pipeline.
            var pipeline =
                    // Convert the string labels into key types.
                    mlContext.Transforms.Conversion.MapValueToKey("Label")
                    // Apply StochasticDualCoordinateAscent multiclass trainer.
                    .Append(mlContext.MulticlassClassification.Trainers.
                    SdcaMaximumEntropy());

            // Train the model and do predictions on same data set.
            // Typically predictions would be in a different, validation set.
            var dataWithPredictions = pipeline.Fit(dataView).Transform(dataView);

            // At this point, the Label column is transformed from strings, to
            // DataViewKeyType and the transformation has added the PredictedLabel
            // column, with same DataViewKeyType as transformed Label column.
            // MapKeyToValue would take columns with DataViewKeyType and convert
            // them back to their original values.
            var newPipeline = mlContext.Transforms.Conversion.MapKeyToValue(new[]
            {
                new InputOutputColumnPair("LabelOriginalValue","Label"),
                new InputOutputColumnPair("PredictedLabelOriginalValue",
                "PredictedLabel")

            });

            var transformedData = newPipeline.Fit(dataWithPredictions).Transform(
                dataWithPredictions);

            // Let's iterate over first 5 items.
            transformedData = mlContext.Data.TakeRows(transformedData, 5);
            var values = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false);

            // Printing the column names of the transformed data.
            Console.WriteLine($"Label   LabelOriginalValue   PredictedLabel   " +
                $"PredictedLabelOriginalValue");

            foreach (var row in values)
                Console.WriteLine($"{row.Label}\t\t{row.LabelOriginalValue}\t\t\t" +
                    $"{row.PredictedLabel}\t\t\t{row.PredictedLabelOriginalValue}");

            // Expected output:
            //  Label   LabelOriginalValue   PredictedLabel   PredictedLabelOriginalValue
            //  1               AA                      1                       AA
            //  2               BB                      2                       BB
            //  3               CC                      4                       DD
            //  4               DD                      4                       DD
            //  1               AA                      1                       AA

        }

        private class DataPoint
        {
            public string Label { get; set; }
            [VectorType(10)]
            public float[] Features { get; set; }
        }

        private static List<DataPoint> GenerateRandomDataPoints(int count,
            int featureVectorLenght)
        {
            var examples = new List<DataPoint>();
            var rnd = new Random(0);
            for (int i = 0; i < count; ++i)
            {
                var example = new DataPoint();
                example.Features = new float[featureVectorLenght];
                var res = i % 4;
                // Generate random float feature values.
                for (int j = 0; j < featureVectorLenght; ++j)
                {
                    var value = (float)rnd.NextDouble() + res * 0.2f;
                    example.Features[j] = value;
                }

                // Generate label based on feature sum.
                if (res == 0)
                    example.Label = "AA";
                else if (res == 1)
                    example.Label = "BB";
                else if (res == 2)
                    example.Label = "CC";
                else
                    example.Label = "DD";
                examples.Add(example);
            }
            return examples;
        }
        private class TransformedData
        {
            public uint Label { get; set; }
            public uint PredictedLabel { get; set; }
            public string LabelOriginalValue { get; set; }
            public string PredictedLabelOriginalValue { get; set; }
        }
    }
}
