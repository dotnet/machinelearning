using System.Linq;
using Google.Protobuf;
using Microsoft.ML.Data;
using Microsoft.ML.Model.Onnx;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class OnnxConversionTest : BaseTestBaseline
    {
        private class AdultData
        {
            [LoadColumn(0, 10), ColumnName("FeatureVector")]
            public float Features { get; set; }

            [LoadColumn(11)]
            public float Target { get; set; }
        }

        public OnnxConversionTest(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// In this test, we convert a trained <see cref="TransformerChain"/> into ONNX <see cref="UniversalModelFormat.Onnx.ModelProto"/> file and then
        /// call <see cref="OnnxScoringEstimator"/> to evaluate that file. The outputs of <see cref="OnnxScoringEstimator"/> are checked against the original
        /// ML.NET model's outputs. 
        /// </summary>
        [Fact]
        public void SimpleEndToEndOnnxConversionTest()
        {
            // Step 1: Create and train a ML.NET pipeline.
            var trainDataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var mlContext = new MLContext();
            var data = mlContext.Data.ReadFromTextFile<AdultData>(trainDataPath,
                hasHeader: true,
                separatorChar: ';'
            );
            var cachedTrainData = mlContext.Data.Cache(data);
            var dynamicPipeline =
                mlContext.Transforms.Normalize("FeatureVector")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.StochasticDualCoordinateAscent(labelColumn: "Target", featureColumn: "FeatureVector"));
            var model = dynamicPipeline.Fit(data);
            var transformedData = model.Transform(data);

            // Step 2: Convert ML.NET model to ONNX format and save it as a file.
            var onnxModel = TransformerChainOnnxConverter.Convert(model, data);
            var onnxFileName = "model.onnx";
            var onnxFilePath = GetOutputPath(onnxFileName);
            using (var file = (mlContext as IHostEnvironment).CreateOutputFile(onnxFilePath))
            using (var stream = file.CreateWriteStream())
                onnxModel.WriteTo(stream);

            // Step 3: Evaluate the saved ONNX model using the data used to train the ML.NET pipeline.
            string[] inputNames = onnxModel.Graph.Input.Select(valueInfoProto => valueInfoProto.Name).ToArray();
            string[] outputNames = onnxModel.Graph.Output.Select(valueInfoProto => valueInfoProto.Name).ToArray();
            var onnxEstimator = new OnnxScoringEstimator(mlContext, onnxFilePath, inputNames, outputNames);
            var onnxTransformer = onnxEstimator.Fit(data);
            var onnxResult = onnxTransformer.Transform(data);

            // Step 4: Compare ONNX and ML.NET results.
            using (var expectedCursor = transformedData.GetRowCursor(columnIndex => columnIndex == transformedData.Schema["Score"].Index))
            using (var actualCursor = onnxResult.GetRowCursor(columnIndex => columnIndex == onnxResult.Schema["Score0"].Index))
            {
                float expected = default;
                VBuffer<float> actual = default;
                var expectedGetter = expectedCursor.GetGetter<float>(transformedData.Schema["Score"].Index);
                var actualGetter = actualCursor.GetGetter<VBuffer<float>>(onnxResult.Schema["Score0"].Index);
                while(expectedCursor.MoveNext() && actualCursor.MoveNext())
                {
                    expectedGetter(ref expected);
                    actualGetter(ref actual);

                    Assert.Equal(expected, actual.GetValues()[0], 1);
                }
            }
        }
    }
}
