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

        private class MNISTInputData
        {
            [VectorType(1, 3, 224, 224)]
            public float[] Features { get; set; }
        }

        private class MINSTOutputData
        {
            [VectorType(1000)]
            public float[] TorchOutput { get; set; }
        }

        [TorchFact]
        public void TorchMNISTScoringTest()
        {
            var mlContext = new MLContext();
            var ones = FloatTensor.Ones(new long[] { 1, 3, 224, 224 });
            var data = new MNISTInputData
            {
                Features = ones.Data<float>().ToArray()
            };
            var dataPoint = new List<MNISTInputData>() { data };

            var dataView = mlContext.Data.LoadFromEnumerable(dataPoint);

            var output = mlContext.Model
                .LoadTorchModel(GetDataPath("Torch/MnistModel.pt"))
                .ScoreTorchModel("TorchOutput", new long[] { 1, 3, 224, 224 }, "Features")
                .Fit(dataView)
                .Transform(dataView);

             var count = mlContext.Data.CreateEnumerable<MINSTOutputData>(output, false).ToArray()[0].TorchOutput.Length;
            Assert.True(count == 1000);
        }

        [TorchFact]
        public void TorchMNISTWorkoutTest()
        {
            var mlContext = new MLContext();
            var ones = FloatTensor.Ones(new long[] { 1, 3, 224, 224 });
            var datapoint = new MNISTInputData
            {
                Features = ones.Data<float>().ToArray()
            };
            var data = new List<MNISTInputData>() { datapoint, datapoint, datapoint, datapoint, datapoint };

            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var estimator = mlContext.Model.LoadTorchModel(GetDataPath("Torch/MnistModel.pt"))
                .ScoreTorchModel("TorchOutput", new long[] { 1, 3, 224, 224 }, "Features");

            TestEstimatorCore(estimator, dataView);

            var output = estimator.Fit(dataView)
                .Transform(dataView);

            var count = mlContext.Data.CreateEnumerable<MINSTOutputData>(output, false).ToArray()[0].TorchOutput.Length;
            Assert.True(count == 1000);
        }
    }
}
