using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class ConvertType
    {
        private sealed class InputData
        {
            public bool Survived;
        }

        private sealed class TransformedData
        {
            public bool Survived { get; set; }

            public Int32 SurvivedInt32 { get; set; }
        }

        public static void Example()
        {
            var mlContext = new MLContext(seed: 1);
            var rawData = new[] {
                new InputData() { Survived = true },
                new InputData() { Survived = false },
                new InputData() { Survived = true },
                new InputData() { Survived = false },
                new InputData() { Survived = false },
            };

            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Construct the pipeline.
            var pipeline = mlContext.Transforms.Conversion.ConvertType("SurvivedInt32", "Survived", DataKind.Int32);

            // Let's train our pipeline, and then apply it to the same data.
            var transformer = pipeline.Fit(data);
            var transformedData = transformer.Transform(data);

            // Display original column 'Survived' (boolean) and converted column 'SurvivedInt32' (Int32)
            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, true);
            foreach (var item in convertedData)
            {
                Console.WriteLine("A:{0,-10}  Aconv:{1}", item.Survived, item.SurvivedInt32);
            }

            // Output
            // A: True     Aconv:1
            // A: False    Aconv:0
            // A: True     Aconv:1
            // A: False    Aconv:0
            // A: False    Aconv:0
        }
    }
}