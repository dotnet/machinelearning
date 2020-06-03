using System.Collections.Generic;
using System.IO;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class OnnxConversion
    {
        

        public static void Example()
        {
            var mlContext = new MLContext(seed: 0);

            // Download and featurize the dataset.
            var data = Microsoft.ML.SamplesUtils.DatasetUtils
                .LoadFeaturizedAdultDataset(mlContext);

            // Leave out 10% of data for testing.
            var trainTestData = mlContext.Data
                .TrainTestSplit(data, testFraction: 0.3);

            // Create data training pipeline for non calibrated trainer and train
            // Naive calibrator on top of it.
            var pipeline = mlContext.BinaryClassification.Trainers.LinearSvm();

            // Fit the pipeline, and get a transformer that knows how to score new
            // data.  
            var transformer = pipeline.Fit(data);

            //What you need to convert an ML.NET model to an onnx model is a transformer and input data
            //By default, the onnx conversion will generate the onnx file with the latest OpSet version
            using (var stream = File.Create("sample_onnx_conversion_1.onnx"))
                mlContext.Model.ConvertToOnnx(transformer, data, stream);

            //However, you can also specify a custom OpSet version by using the following code
            //Currently, we support OpSet versions 9 for most transformers, but there are certain transformers that require a higher OpSet version
            /*int customOpSetVersion = 9;
            using (var stream = File.Create("sample_onnx_conversion_2.onnx"))
                mlContext.Model.ConvertToOnnx(transformer, data, customOpSetVersion, stream);*/

            var onnxModelPath = "C:\\Users\\Administrator\\Desktop\\MLNET\\sample\\bin\\AnyCPU.Debug\\Microsoft.ML.Samples\\netcoreapp2.1\\sample_onnx_conversion_1.onnx";
            var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(onnxModelPath);
            var onnxTransformer = onnxEstimator.Fit(data);

            var onnxResult = onnxTransformer.Transform(data);
            var mlnetResult = transformer.Transform(data);
        }
    }
}
