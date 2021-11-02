using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Samples.Dynamic
{
    // This example demonstrates hashing of categorical string and integer data types by using Hash transform's 
    // advanced options API.
    public static class HashWithOptions
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext(seed: 1);

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Category = "MLB" , Age = 18 },
                new DataPoint() { Category = "NFL" , Age = 14 },
                new DataPoint() { Category = "NFL" , Age = 15 },
                new DataPoint() { Category = "MLB" , Age = 18 },
                new DataPoint() { Category = "MLS" , Age = 14 },
            };

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Construct the pipeline that would hash the two columns and store the
            // results in new columns. The first transform hashes the string column
            // and the second transform hashes the integer column.
            //
            // Hashing is not a reversible operation, so there is no way to retrieve
            // the original value from the hashed value. Sometimes, for debugging,
            // or model explainability, users will need to know what values in the
            // original columns generated the values in the hashed columns, since
            // the algorithms will mostly use the hashed values for further
            // computations. The Hash method will preserve the mapping from the
            // original values to the hashed values in the Annotations of the newly
            // created column (column populated with the hashed values). 
            //
            // Setting the maximumNumberOfInverts parameters to -1 will preserve the
            // full map. If that parameter is left to the default 0 value, the
            // mapping is not preserved.
            var pipeline = mlContext.Transforms.Conversion.Hash(
                    new[]
                    {
                            new HashingEstimator.ColumnOptions(
                                "CategoryHashed",
                                "Category",
                                16,
                                useOrderedHashing: false,
                                maximumNumberOfInverts: -1),

                            new HashingEstimator.ColumnOptions(
                                "AgeHashed",
                                "Age",
                                8,
                                useOrderedHashing: false)
                    });

            // Let's fit our pipeline, and then apply it to the same data.
            var transformer = pipeline.Fit(data);
            var transformedData = transformer.Transform(data);

            // Convert the post transformation from the IDataView format to an
            // IEnumerable <TransformedData> for easy consumption.
            var convertedData = mlContext.Data.CreateEnumerable<
                TransformedDataPoint>(transformedData, true);

            Console.WriteLine("Category CategoryHashed\t Age\t AgeHashed");
            foreach (var item in convertedData)
                Console.WriteLine($"{item.Category}\t {item.CategoryHashed}\t\t  " +
                    $"{item.Age}\t {item.AgeHashed}");

            // Expected data after the transformation.
            //
            // Category CategoryHashed   Age     AgeHashed
            // MLB      36206            18      127
            // NFL      19015            14      62
            // NFL      19015            15      43
            // MLB      36206            18      127
            // MLS      6013             14      62

            // For the Category column, where we set the maximumNumberOfInverts
            // parameter, the names of the original categories, and their
            // correspondence with the generated hash values is preserved in the
            // Annotations in the format of indices and values.the indices array
            // will have the hashed values, and the corresponding element,
            // position -wise, in the values array will contain the original value. 
            //
            // See below for an example on how to retrieve the mapping. 
            var slotNames = new VBuffer<ReadOnlyMemory<char>>();
            transformedData.Schema["CategoryHashed"].Annotations.GetValue(
                "KeyValues", ref slotNames);

            var indices = slotNames.GetIndices();
            var categoryNames = slotNames.GetValues();

            for (int i = 0; i < indices.Length; i++)
                Console.WriteLine($"The original value of the {indices[i]} " +
                    $"category is {categoryNames[i]}");

            // Output Data
            // 
            // The original value of the 6012 category is MLS
            // The original value of the 19014 category is NFL
            // The original value of the 36205 category is MLB
        }

        public class DataPoint
        {
            public string Category { get; set; }
            public uint Age { get; set; }
        }

        public class TransformedDataPoint : DataPoint
        {
            public uint CategoryHashed { get; set; }
            public uint AgeHashed { get; set; }
        }

    }
}
