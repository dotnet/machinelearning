using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework.Attributes;
using TorchSharp.Tensor;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Torch
{
    public partial class TorchTests : TestDataPipeBase
    {
        public TorchTests(ITestOutputHelper helper) : base(helper)
        {
        }

        private class TestReLUModelData
        {
            [VectorType(5)]
            public float[] Features { get; set; }
        }

        [TorchFact]
        public void TorchScoringReLUTest()
        {
            var mlContext = new MLContext();
            var tensor = new float[] { -1, -1, 0, 1, 1 }.ToTorchTensor(dimensions: new long[] { 5 });
            var data = new TestReLUModelData
            {
                Features = tensor.Data<float>().ToArray()
            };
            var dataPoint = new List<TestReLUModelData>() { data };

            var dataView = mlContext.Data.LoadFromEnumerable(dataPoint);

            var output = mlContext.Model
                .LoadTorchModel(GetDataPath("Torch/relu.pt"))
                .ScoreTorchModel("Features", new long[] { 5 })
                .Fit(dataView)
                .Transform(dataView);

             var transformedData = mlContext.Data.CreateEnumerable<TestReLUModelData>(output, false).ToArray()[0].Features;
            Assert.True(transformedData.Length == 5);
            Assert.Equal(transformedData, new float[] { 0, 0, 0, 1, 1 });

        }

        [TorchFact]
        public void TorchTransformerWorkoutTest()
        {
            var mlContext = new MLContext();
            var tensorData = FloatTensor.Random(new long[] { 5 });
            var datapoint = new TestReLUModelData
            {
                Features = tensorData.Data<float>().ToArray()
            };
            var data = new List<TestReLUModelData>() { datapoint, datapoint, datapoint, datapoint, datapoint };

            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var estimator = mlContext.Model.LoadTorchModel(GetDataPath("Torch/relu.pt"))
                .ScoreTorchModel("TorchOutput", new long[] { 5 }, "Features");

            TestEstimatorCore(estimator, dataView);

            var output = estimator.Fit(dataView)
                .Transform(dataView);

            var transformedData = mlContext.Data.CreateEnumerable<TestReLUModelData>(output, false).ToArray()[0].Features;
            Assert.True(transformedData.Length == 5);
            foreach (var elt in transformedData)
                Assert.True(elt >= 0);
        }
    }
}
