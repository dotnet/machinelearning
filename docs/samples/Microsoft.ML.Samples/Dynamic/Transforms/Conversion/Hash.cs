using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class Hash
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext(seed: 1);

            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new InputData() { Category = "MLB" , Age = 18 },
                new InputData() { Category = "NFL" , Age = 14 },
                new InputData() { Category = "NFL" , Age = 15 },
                new InputData() { Category = "MLB" , Age = 18 },
                new InputData() { Category = "MLS" , Age = 14 },
            };

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Construct the pipeline that would hash the two columns and store the results in new columns.
            var pipeline = mlContext.Transforms.Conversion.Hash("CategoryHashed", "Category", numberOfBits: 16, maximumNumberOfInverts: 2)
                          .Append(mlContext.Transforms.Conversion.Hash("AgeHashed", "Age", numberOfBits: 8));

            // Let's fit our pipeline, and then apply it to the same data.
            var transformer = pipeline.Fit(data);
            var transformedData = transformer.Transform(data);

            // Store the post transformation from the IDataView format, to an IEnumerable<TransformedData> for easy consumption
            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, true);
            foreach (var item in convertedData)
            {
                Console.WriteLine($"Category: {item.Category} - CategoryHashed: {item.CategoryHashed}. Age: {item.Age} - AgeHashed {item.AgeHashed}");
            }

            // Output
            //
            // Category: MLB - CategoryHashed: 36206.Age: 18 - AgeHashed 127
            // Category: NFL - CategoryHashed: 19015.Age: 14 - AgeHashed 62
            // Category: NFL - CategoryHashed: 19015.Age: 15 - AgeHashed 43
            // Category: MLB - CategoryHashed: 36206.Age: 18 - AgeHashed 127
            // Category: MLS - CategoryHashed: 6013.Age: 14 - AgeHashed 62

            // for the Category column, where we set the maximumNumberOfInvertsparameter, the names of the original categories, 
            // and their correspondance with the generated hash values is preserved. 
            var slotNames = new VBuffer<ReadOnlyMemory<char>>();
            transformedData.Schema["CategoryHashed"].Annotations.GetValue("KeyValues", ref slotNames);

            var indices = slotNames.GetIndices();
            var categoryNames = slotNames.GetValues();

            for (int i = 0; i < indices.Length; i++)
                Console.WriteLine($"The original value of the {indices[i]} category is {categoryNames[i]}");

            // Output Data
            // 
            // The original value of the 6012 category is MLS
            // The original value of the 19014 category is NFL
            // The original value of the 36205 category is MLB

        }

        private class InputData
        {
            public string Category;
            public uint Age;
        }

        private class TransformedData : InputData
        {
            public uint CategoryHashed;
            public uint AgeHashed;
        }

    }
}