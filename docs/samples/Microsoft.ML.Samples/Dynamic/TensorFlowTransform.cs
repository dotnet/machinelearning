// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using System;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic
{
    class TensorFlowTransformExample
    {
        /// <summary>
        /// Example use of TensorFlowEstimator in an ML.NET pipeline
        /// </summary>
        public static void TFTransformSample()
        {
            Console.WriteLine("Hello World!");
            // var model_location = @"D:\machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Tests\netcoreapp2.1\cifar_model\frozen_model.pb";
            var model_location = @"C:\Users\agoswami\Downloads\resnet_v2_101_299_frozen.pb";

            // Create ML pipeline to score the data using OnnxScoringEstimator
            var mlContext = new MLContext();
            var data = GetTensorData();

            var idv = mlContext.CreateStreamingDataView(data);
            // var pipeline = new TensorFlowEstimator(mlContext, model_location, new[] { "Input" }, new[] { "Output" });
            var pipeline = new TensorFlowEstimator(mlContext, model_location, new[] { "input" }, new[] { "output" });

            // Run the pipeline and get the transformed values
            var estimator = pipeline.Fit(idv);
            var transformedValues = estimator.Transform(idv);

            // Retrieve model scores into OutputScores class
            var outScores = transformedValues.AsEnumerable<OutputScores>(mlContext, reuseRowObject: false);

            // Iterate rows
            foreach (var prediction in outScores)
            {
                int numClasses = 0;
                //foreach (var classScore in prediction.Output.Take(3))
                foreach (var classScore in prediction.output.Take(3))
                {
                    Console.WriteLine($"Class #{numClasses++} score = {classScore}");
                }
                Console.WriteLine(new string('-', 10));
            }
        }

        /// <summary>
        /// inputSize is the overall dimensions of the model input tensor.
        /// </summary>
        private const int imageHeight = 324;
        private const int imageWidth = 324;
        private const int numChannels = 3;
        private const int inputSize = imageHeight * imageWidth * numChannels;

        /// <summary>
        /// A class to hold sample tensor data. Member name should match  
        /// the inputs that the model expects (in this case, Input)
        /// </summary>
        public class TensorData
        {
            [VectorType(imageHeight, imageWidth, numChannels)]
            // public float[] Input { get; set; }
            public float[] input { get; set; }
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
            //return new TensorData[] { new TensorData() { Input = image1 }, new TensorData() { Input = image2 } };
            return new TensorData[] { new TensorData() { input = image1 }, new TensorData() { input = image2 } };
        }

        /// <summary>
        /// Class to contain the output values from the transformation.
        /// This model generates a vector of 1000 floats.
        /// </summary>
        class OutputScores
        {
            [VectorType(1000)]
            //public float[] Output { get; set; }
            public float[] output { get; set; }
        }
    }
}