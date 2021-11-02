using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class ApplyOnnxModel
    {
        public static void Example()
        {
            // Download the squeeznet image model from ONNX model zoo, version 1.2
            // https://github.com/onnx/models/tree/master/squeezenet or use
            // Microsoft.ML.Onnx.TestModels nuget.
            var modelPath = @"squeezenet\00000001\model.onnx";

            // Create ML pipeline to score the data using OnnxScoringEstimator
            var mlContext = new MLContext();

            // Generate sample test data.
            var samples = GetTensorData();
            // Convert training data to IDataView, the general data type used in
            // ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);
            // Create the pipeline to score using provided onnx model.
            var pipeline = mlContext.Transforms.ApplyOnnxModel(modelPath);
            // Fit the pipeline and get the transformed values
            var transformedValues = pipeline.Fit(data).Transform(data);
            // Retrieve model scores into Prediction class
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(
                transformedValues, reuseRowObject: false);

            // Iterate rows
            foreach (var prediction in predictions)
            {
                int numClasses = 0;
                foreach (var classScore in prediction.softmaxout_1.Take(3))
                {
                    Console.WriteLine("Class #" + numClasses++ + " score = " +
                        classScore);
                }
                Console.WriteLine(new string('-', 10));
            }

            // Results look like below...
            // Class #0 score = 4.544065E-05
            // Class #1 score = 0.003845858
            // Class #2 score = 0.0001249467
            // ----------
            // Class #0 score = 4.491953E-05
            // Class #1 score = 0.003848222
            // Class #2 score = 0.0001245592
            // ----------
        }

        // inputSize is the overall dimensions of the model input tensor.
        private const int inputSize = 224 * 224 * 3;

        // A class to hold sample tensor data. Member name should match  
        // the inputs that the model expects (in this case, data_0)
        public class TensorData
        {
            [VectorType(inputSize)]
            public float[] data_0 { get; set; }
        }

        // Method to generate sample test data. Returns 2 sample rows.
        public static TensorData[] GetTensorData()
        {
            // This can be any numerical data. Assume image pixel values.
            var image1 = Enumerable.Range(0, inputSize).Select(x => (float)x /
                inputSize).ToArray();

            var image2 = Enumerable.Range(0, inputSize).Select(x => (float)(x +
                10000) / inputSize).ToArray();

            return new TensorData[] { new TensorData() { data_0 = image1 }, new
                TensorData() { data_0 = image2 } };
        }

        // Class to contain the output values from the transformation.
        // This model generates a vector of 1000 floats.
        class Prediction
        {
            [VectorType(1000)]
            public float[] softmaxout_1 { get; set; }
        }
    }
}
