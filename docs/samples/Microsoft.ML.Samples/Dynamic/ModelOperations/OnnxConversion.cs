using System.IO;
using Microsoft.ML;

namespace Samples.Dynamic.ModelOperations
{
    public static class OnnxConversion
    {
        private class ScoreValue
        {
            public float Score { get; set; }
        }

        public static void Example()
        {
            var mlContext = new MLContext(seed: 0);

            // Download the raw dataset.
            var rawData = Microsoft.ML.SamplesUtils.DatasetUtils
                .LoadRawAdultDataset(mlContext);

            // Leave out 10% of data for testing.
            var trainTestData = mlContext.Data
                .TrainTestSplit(rawData, testFraction: 0.3);

            // Create data training pipeline for non calibrated trainer and train
            // Naive calibrator on top of it.
            var pipeline = mlContext.Transforms.CopyColumns("Label", "IsOver50K")
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
            var transformer = pipeline.Fit(trainTestData.TrainSet);

            //What you need to convert an ML.NET model to an onnx model is a transformer and input data
            //By default, the onnx conversion will generate the onnx file with the latest OpSet version
            using (var stream = File.Create("sample_onnx_conversion_1.onnx"))
                mlContext.Model.ConvertToOnnx(transformer, rawData, stream);

            //However, you can also specify a custom OpSet version by using the following code
            //Currently, we support OpSet versions 9 for most transformers, but there are certain transformers that require a higher OpSet version
            int customOpSetVersion = 9;
            using (var stream = File.Create("sample_onnx_conversion_2.onnx"))
                mlContext.Model.ConvertToOnnx(transformer, rawData, customOpSetVersion, stream);

            //Inference using the onnx model
            var onnxModelPath = "your_path_to_onnx_file";
            var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(onnxModelPath);
            var onnxTransformer = onnxEstimator.Fit(trainTestData.TrainSet);

            var onnxResult = onnxTransformer.Transform(trainTestData.TestSet);
        }
    }
}
