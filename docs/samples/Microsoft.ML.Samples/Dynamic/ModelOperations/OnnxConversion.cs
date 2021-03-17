using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic.ModelOperations
{
    public static class OnnxConversion
    {
        private class ScoreValue
        {
            public float Score { get; set; }
        }

        private class OnnxScoreValue
        {
            public VBuffer<float> Score { get; set; }
        }

        private static void PrintScore(IEnumerable<ScoreValue> values, int numRows)
        {
            foreach (var value in values.Take(numRows))
                Console.WriteLine("{0, -10} {1, -10}", "Score", value.Score);
        }

        private static void PrintScore(IEnumerable<OnnxScoreValue> values, int numRows)
        {
            foreach (var value in values.Take(numRows))
                Console.WriteLine("{0, -10} {1, -10}", "Score", value.Score.GetItemOrDefault(0));
        }

        public static void Example()
        {
            var mlContext = new MLContext(seed: 0);

            //Get dataset
            // Download the raw dataset.
            var originalData = Microsoft.ML.SamplesUtils.DatasetUtils
                .LoadRawAdultDataset(mlContext);

            //Dataset partition
            // Partition the original dataset. Leave out 10% of data for testing.
            var trainTestOriginalData = mlContext.Data
                .TrainTestSplit(originalData, testFraction: 0.3);

            // Define training pielines(wholePipeline = featurizationPipeline + binaryRegressionpipeline)
            var wholePipeline = mlContext.Transforms.CopyColumns("Label", "IsOver50K")
                        // Convert categorical features to one-hot vectors
                        .Append(mlContext.Transforms.Categorical.OneHotEncoding("workclass"))
                        .Append(mlContext.Transforms.Categorical.OneHotEncoding("education"))
                        .Append(mlContext.Transforms.Categorical.OneHotEncoding("marital-status"))
                        .Append(mlContext.Transforms.Categorical.OneHotEncoding("occupation"))
                        .Append(mlContext.Transforms.Categorical.OneHotEncoding("relationship"))
                        .Append(mlContext.Transforms.Categorical.OneHotEncoding("ethnicity"))
                        .Append(mlContext.Transforms.Categorical.OneHotEncoding("native-country"))
                        // Combine all features into one feature vector
                        .Append(mlContext.Transforms.Concatenate("Features", "workclass", "education", "marital-status",
                        "occupation", "relationship", "ethnicity", "native-country", "age", "education-num",
                        "capital-gain", "capital-loss", "hours-per-week"))
                        // Min-max normalize all the features
                        .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                        .Append(mlContext.BinaryClassification.Trainers.AveragedPerceptron());

            // Fit the pipeline, and get a transformer that knows how to score new data
            var transformer = wholePipeline.Fit(trainTestOriginalData.TrainSet);

            //What you need to convert an ML.NET model to an onnx model is a transformer and input data
            //By default, the onnx conversion will generate the onnx file with the latest OpSet version
            using (var stream = File.Create("sample_onnx_conversion_1.onnx"))
                mlContext.Model.ConvertToOnnx(transformer, originalData, stream);

            //However, you can also specify a custom OpSet version by using the following code
            //Currently, we support OpSet versions 9 for most transformers, but there are certain transformers that require a higher OpSet version
            //Please refer to the following link for most update information of what OpSet version we support
            //https://github.com/dotnet/machinelearning/blob/main/src/Microsoft.ML.OnnxConverter/OnnxExportExtensions.cs
            int customOpSetVersion = 9;
            using (var stream = File.Create("sample_onnx_conversion_2.onnx"))
                mlContext.Model.ConvertToOnnx(transformer, originalData, customOpSetVersion, stream);

            //Create the pipeline using onnx file.
            var onnxModelPath = "your_path_to_sample_onnx_conversion_1.onnx";
            var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(onnxModelPath);
            //Make sure to either use the 'using' clause or explicitly dispose the returned onnxTransformer to prevent memory leaks
            using var onnxTransformer = onnxEstimator.Fit(trainTestOriginalData.TrainSet);

            //Inference the testset
            var output = transformer.Transform(trainTestOriginalData.TestSet);
            var onnxOutput = onnxTransformer.Transform(trainTestOriginalData.TestSet);

            //Get the outScores
            var outScores = mlContext.Data.CreateEnumerable<ScoreValue>(output, reuseRowObject: false);
            var onnxOutScores = mlContext.Data.CreateEnumerable<OnnxScoreValue>(onnxOutput, reuseRowObject: false);

            //Print
            PrintScore(outScores, 5);
            PrintScore(onnxOutScores, 5);
            //Expected same results for the above 4 methods
            //Score - 0.09044361
            //Score - 9.105377
            //Score - 11.049
            //Score - 3.061928
            //Score - 6.375817
        }
    }
}
