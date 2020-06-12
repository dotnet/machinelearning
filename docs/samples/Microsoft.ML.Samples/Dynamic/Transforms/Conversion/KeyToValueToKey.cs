using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.Transforms;

namespace Samples.Dynamic
{
    public class KeyToValueToKey
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Review = "animals birds cats dogs fish horse"},
                new DataPoint() { Review = "horse birds house fish duck cats"},
                new DataPoint() { Review = "car truck driver bus pickup"},
                new DataPoint() { Review = "car truck driver bus pickup horse"},
            };

            var trainData = mlContext.Data.LoadFromEnumerable(rawData);

            // A pipeline to convert the terms of the 'Review' column in 
            // making use of default settings.
            var defaultPipeline = mlContext.Transforms.Text.TokenizeIntoWords(
                "TokenizedText", nameof(DataPoint.Review)).Append(mlContext
                .Transforms.Conversion.MapValueToKey(nameof(TransformedData.Keys),
                "TokenizedText"));

            // Another pipeline, that customizes the advanced settings of the
            // ValueToKeyMappingEstimator. We can change the maximumNumberOfKeys to
            // limit how many keys will get generated out of the set of words, and
            // condition the order in which they get evaluated by changing
            // keyOrdinality from the default ByOccurence (order in which they get
            // encountered) to value/alphabetically.
            var customizedPipeline = mlContext.Transforms.Text.TokenizeIntoWords(
                "TokenizedText", nameof(DataPoint.Review)).Append(mlContext
                .Transforms.Conversion.MapValueToKey(nameof(TransformedData.Keys),
                "TokenizedText", maximumNumberOfKeys: 10, keyOrdinality:
                ValueToKeyMappingEstimator.KeyOrdinality.ByValue));

            // The transformed data.
            var transformedDataDefault = defaultPipeline.Fit(trainData).Transform(
                trainData);

            var transformedDataCustomized = customizedPipeline.Fit(trainData)
                .Transform(trainData);

            // Getting the resulting data as an IEnumerable.
            // This will contain the newly created columns.
            IEnumerable<TransformedData> defaultData = mlContext.Data.
                CreateEnumerable<TransformedData>(transformedDataDefault,
                reuseRowObject: false);

            IEnumerable<TransformedData> customizedData = mlContext.Data.
                CreateEnumerable<TransformedData>(transformedDataCustomized,
                reuseRowObject: false);

            Console.WriteLine($"Keys");
            foreach (var dataRow in defaultData)
                Console.WriteLine($"{string.Join(',', dataRow.Keys)}");
            // Expected output:
            //  Keys
            //  1,2,3,4,5,6
            //  6,2,7,5,8,3
            //  9,10,11,12,13
            //  9,10,11,12,13,6

            Console.WriteLine($"Keys");
            foreach (var dataRow in customizedData)
                Console.WriteLine($"{string.Join(',', dataRow.Keys)}");
            // Expected output:
            //  Keys
            //  1,2,4,5,7,8
            //  8,2,9,7,6,4
            //  3,10,0,0,0
            //  3,10,0,0,0,8
            // Retrieve the original values, by appending the KeyToValue estimator to
            // the existing pipelines to convert the keys back to the strings.
            var pipeline = defaultPipeline.Append(mlContext.Transforms.Conversion
                .MapKeyToValue(nameof(TransformedData.Keys)));

            transformedDataDefault = pipeline.Fit(trainData).Transform(trainData);

            // Preview of the DefaultColumnName column obtained.
            var originalColumnBack = transformedDataDefault.GetColumn<VBuffer<
                ReadOnlyMemory<char>>>(transformedDataDefault.Schema[nameof(
                TransformedData.Keys)]);

            foreach (var row in originalColumnBack)
            {
                foreach (var value in row.GetValues())
                    Console.Write($"{value} ");
                Console.WriteLine("");
            }

            // Expected output:
            //  animals birds cats dogs fish horse
            //  horse birds house fish duck cats
            //  car truck driver bus pickup
            //  car truck driver bus pickup horse
        }

        private class DataPoint
        {
            public string Review { get; set; }
        }

        private class TransformedData : DataPoint
        {
            public uint[] Keys { get; set; }
        }
    }
}
