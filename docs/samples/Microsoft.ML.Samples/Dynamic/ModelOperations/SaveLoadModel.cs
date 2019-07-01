using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;

namespace Samples.Dynamic.ModelOperations
{
    public class SaveLoadModel
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Generate sample data.
            var data = new List<Data>()
            {
                new Data() { Value="abc" }
            };

            // Convert data to IDataView.
            var dataView = mlContext.Data.LoadFromEnumerable(data);
            var inputColumnName = nameof(Data.Value);
            var outputColumnName = nameof(Transformation.Key);

            // Transform.
            ITransformer model = mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName, inputColumnName).Fit(dataView);

            // Save model.
            mlContext.Model.Save(model, dataView.Schema, "model.zip");

            // Load model.
            using (var file = File.OpenRead("model.zip"))
                model = mlContext.Model.Load(file, out DataViewSchema schema);

            // Create a prediction engine from the model for feeding new data.
            var engine = mlContext.Model
                .CreatePredictionEngine<Data, Transformation>(model);

            var transformation = engine.Predict(new Data() { Value = "abc" });

            // Print transformation to console.
            Console.WriteLine("Value: {0}\t Key:{1}", transformation.Value,
                transformation.Key);

            // Value: abc       Key:1

        }

        private class Data
        {
            public string Value { get; set; }
        }

        private class Transformation
        {
            public string Value { get; set; }
            public uint Key { get; set; }
        }
    }
}
