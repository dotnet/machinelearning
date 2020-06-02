using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Transforms.Onnx;

namespace Samples.Dynamic.ModelOperations
{
    public static class OnnxConversion
    {
        class PlattModelInput
        {
            public bool Label { get; set; }
            public float Score { get; set; }
        }

        static IEnumerable<PlattModelInput> PlattGetData()
        {
            for (int i = 0; i < 100; i++)
            {
                yield return new PlattModelInput { Score = i, Label = i % 2 == 0 };
            }
        }
        public static void Example()
        {
            
            var mlContext = new MLContext(seed: 0);

            //Get input data
            IDataView data = mlContext.Data.LoadFromEnumerable(PlattGetData());

            //Get estimator
            var calibratorEstimator = mlContext.BinaryClassification.Calibrators.Platt();

            //Get pipeline(calibratorTransformer) by calling Fit()
            var calibratorTransformer = calibratorEstimator.Fit(data);

            //What you need to convert ML.NET model to onnx model are pipeline and input data
            //In default, the onnx conversion will generate the onnx file with the latest OpSet version
            using (var stream = File.Create("sample_onnx_conversion_1.onnx"))
                mlContext.Model.ConvertToOnnx(calibratorTransformer, data, stream);

            //However, you can also specify custom OpSet version by the following code
            //Currently we support OpSet version from 9-12 for most transformers, but there are certain transformers that require higher version of OpSet
            int customOpSetVersion = 9;
            using (var stream = File.Create("sample_onnx_conversion_2.onnx"))
                mlContext.Model.ConvertToOnnx(calibratorTransformer, data, customOpSetVersion, stream);

        }
    }
}
