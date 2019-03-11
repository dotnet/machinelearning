using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class OnnxTransformExample
    {
        /// <summary>
        /// Example use of OnnxEstimator in an ML.NET pipeline
        /// </summary>
        public static void Example()
        {
            // Download the squeeznet image model from ONNX model zoo, version 1.2
            // https://github.com/onnx/models/tree/master/squeezenet
            var modelPath = @"squeezenet\model.onnx";

            // Inspect the model's inputs and outputs
            var session = new InferenceSession(modelPath);
            var inputInfo = session.InputMetadata.First();
            var outputInfo = session.OutputMetadata.First();
            Console.WriteLine($"Input Name is {String.Join(",", inputInfo.Key)}");
            Console.WriteLine($"Input Dimensions are {String.Join(",", inputInfo.Value.Dimensions)}");
            Console.WriteLine($"Output Name is {String.Join(",", outputInfo.Key)}");
            Console.WriteLine($"Output Dimensions are {String.Join(",", outputInfo.Value.Dimensions)}");
            // Results..
            // Input Name is data_0
            // Input Dimensions are 1,3,224,224
            // Output Name is softmaxout_1
            // Output Dimensions are 1,1000,1,1

            // Create ML pipeline to score the data using OnnxScoringEstimator
            var mlContext = new MLContext();
            var data = GetTensorData();
            var idv = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ApplyOnnxModel(modelPath, new[] { outputInfo.Key }, new[] { inputInfo.Key });

            // Run the pipeline and get the transformed values
            var transformedValues = pipeline.Fit(idv).Transform(idv);

            // Retrieve model scores into Prediction class
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedValues, reuseRowObject: false);

            // Iterate rows
            foreach (var prediction in predictions)
            {
                int numClasses = 0;
                foreach (var classScore in prediction.softmaxout_1.Take(3))
                {
                    Console.WriteLine($"Class #{numClasses++} score = {classScore}");
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

        /// <summary>
        /// inputSize is the overall dimensions of the model input tensor.
        /// </summary>
        private const int inputSize = 224 * 224 * 3;

        /// <summary>
        /// A class to hold sample tensor data. Member name should match  
        /// the inputs that the model expects (in this case, data_0)
        /// </summary>
        public class TensorData
        {
            [VectorType(inputSize)]
            public float[] data_0 { get; set; }
        }

        /// <summary>
        /// Method to generate sample test data. Returns 2 sample rows.
        /// </summary>
        /// <returns></returns>
        public static TensorData[] GetTensorData()
        {
            // This can be any numerical data. Assume image pixel values.
            var image1 = Enumerable.Range(0, inputSize).Select(x => (float)x / inputSize).ToArray();
            var image2 = Enumerable.Range(0, inputSize).Select(x => (float)(x + 10000) / inputSize).ToArray();
            return new TensorData[] { new TensorData() { data_0 = image1 }, new TensorData() { data_0 = image2 } };
        }

        /// <summary>
        /// Class to contain the output values from the transformation.
        /// This model generates a vector of 1000 floats.
        /// </summary>
        class Prediction
        {
            [VectorType(1000)]
            public float[] softmaxout_1 { get; set; }
        }
    }
}