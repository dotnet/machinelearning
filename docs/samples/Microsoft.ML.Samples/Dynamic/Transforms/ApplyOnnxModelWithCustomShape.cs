using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class ApplyOnnxModelWithCustomShape
    {
        public static void Example()
        {
            // The model comes from the Microsoft.ML.Onnx.TestModels package.
            // Its input "input" has shape [-1, 3], so the first dimension is
            // unknown. It applies ArgMax along the second axis, which returns
            // the index of the largest value in each row as "argmax".
            var modelPath = Path.Combine(AppContext.BaseDirectory,
                "unknowndimensions", "test_unknowndimensions_float.onnx");

            var mlContext = new MLContext();

            // Each sample is a 3-by-3 matrix stored as 9 values in a row.
            var samples = new[]
            {
                new TensorData { input = new float[] {
                    1.1f, 1.3f, 1.2f,
                    1.9f, 1.3f, 1.2f,
                    1.1f, 1.3f, 1.8f } },
                new TensorData { input = new float[] {
                    0f, 0f, 1f,
                    1f, 0f, 0f,
                    1f, 0f, 0f } }
            };
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // The model leaves the first input dimension unknown. Override it
            // so ML.NET treats each sample as a 3-by-3 tensor. The keys are
            // the model's input names; a shape that conflicts with a known
            // dimension in the model, such as [3, 4], causes an exception.
            var shapeDictionary = new Dictionary<string, int[]>
            {
                { "input", new[] { 3, 3 } }
            };

            var pipeline = mlContext.Transforms.ApplyOnnxModel(modelPath,
                shapeDictionary);

            var model = pipeline.Fit(data);
            var transformedValues = model.Transform(data);
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(
                transformedValues, reuseRowObject: false);

            foreach (var prediction in predictions)
                Console.WriteLine(string.Join(" ", prediction.argmax));

            // The ONNX transformer holds native resources, so dispose it.
            (model as IDisposable)?.Dispose();

            // Expected output:
            // 1 0 2
            // 2 0 0
        }

        // The member name must match the model's input name.
        private class TensorData
        {
            [VectorType(3, 3)]
            public float[] input { get; set; }
        }

        // The member name must match the model's output name.
        private class Prediction
        {
            [VectorType(3)]
            public long[] argmax { get; set; }
        }
    }
}
