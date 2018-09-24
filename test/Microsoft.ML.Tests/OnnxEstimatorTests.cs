using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using Xunit;
using Xunit.Abstractions;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxScoring;

namespace Microsoft.ML.Tests
{
    public class OnnxEstimatorTests : TestDataPipeBase
    {

        private const int inputsize = 150528;

        private class TestData
        {
            [VectorType(inputsize)]
            public float[] data_0;
        }
        private class TestDataSize
        {
            [VectorType(2)]
            public float[] data_0;
        }
        private class TestDataXY
        {
            [VectorType(inputsize)]
            public float[] A;
        }
        private class TestDataDifferntType
        {
            [VectorType(inputsize)]
            public string[] data_0;
        }

        private float[] getSampleArrayData()
        {
            var samplevector = new float[inputsize];
            for (int i = 0; i < inputsize; i++)
                samplevector[i] = (i / (inputsize *1.01f));
            return samplevector;
        }

        public OnnxEstimatorTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        void TestSimpleCase()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return;
            var modelFile = "squeezenet/00000001/model.onnx";

            var samplevector = getSampleArrayData();

            var dataView = ComponentCreation.CreateDataView(Env,
                new List<TestData>(new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    },
                     new TestData()
                     {
                    data_0 = samplevector
                     }
                }));

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[inputsize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[inputsize] } };
            var sizeData = new List<TestDataSize> { new TestDataSize() { data_0 = new float[2] } };
            var pipe = new OnnxEstimator(Env, modelFile, new[] { "data_0" }, new[] { "softmaxout_1" });

            var invalidDataWrongNames = ComponentCreation.CreateDataView(Env, xyData);
            var invalidDataWrongTypes = ComponentCreation.CreateDataView(Env, stringData);
            var invalidDataWrongVectorSize = ComponentCreation.CreateDataView(Env, sizeData);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongNames);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongTypes);

            pipe.GetOutputSchema(SchemaShape.Create(invalidDataWrongVectorSize.Schema));
            try
            {
                pipe.Fit(invalidDataWrongVectorSize);
                Assert.False(true);
            }
            catch (ArgumentOutOfRangeException) { }
            catch (InvalidOperationException) { }
        }

        [Fact]
        void TestModelLoadAndScore()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return;

            var modelFile = "squeezenet/00000001/model.onnx";

            var samplevector = getSampleArrayData();

            var dataView = ComponentCreation.CreateDataView(Env,
                new List<TestData>(new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    }
                }));

            var inputNames = new[] { "data_0" };
            var outputNames = new[] { "softmaxout_1" };
            var est = new OnnxEstimator(Env, modelFile, inputNames, outputNames);
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
                //ValidateTensorFlowTransformer(loadedView);

                loadedView.Schema.TryGetColumnIndex(outputNames[0], out int softMaxOut1);
                using (var cursor = loadedView.GetRowCursor(col => col == softMaxOut1))
                {
                    VBuffer<float> softMaxValue = default;
                    var softMaxGetter = cursor.GetGetter<VBuffer<float>>(softMaxOut1);
                    while (cursor.MoveNext())
                    {
                        softMaxGetter(ref softMaxValue);
                        Console.WriteLine(softMaxValue.Values[0]);
                        Console.WriteLine(softMaxValue.Values[1]);
                    }
                }
            }
        }

        [Fact]
        public void OnnxStatic()
        {
            var modelFile = "squeezenet/00000001/model.onnx";

            using (var env = new ConsoleEnvironment(null, false, 0, 1, null, null) )
            {
                var imageHeight = 224;
                var imageWidth = 224;
                var dataFile = GetDataPath("images/images.tsv");
                var imageFolder = Path.GetDirectoryName(dataFile);

                var data = TextLoader.CreateReader(env, ctx => (
                    imagePath: ctx.LoadText(0),
                    name: ctx.LoadText(1)))
                    .Read(new MultiFileSource(dataFile));

                // Note that CamelCase column names are there to match the TF graph node names.
                var pipe = data.MakeNewEstimator()
                    .Append(row => (
                        row.name,
                        data_0: row.imagePath.LoadAsImage(imageFolder).Resize(imageHeight, imageWidth).ExtractPixels(interleaveArgb: true)))
                    .Append(row => (row.name, softmaxout_1: row.data_0.ApplyOnnxModel(modelFile)));

                TestEstimatorCore(pipe.AsDynamic, data.AsDynamic);

                var result = pipe.Fit(data).Transform(data).AsDynamic;
                result.Schema.TryGetColumnIndex("Output", out int output);
                using (var cursor = result.GetRowCursor(col => col == output))
                {
                    var buffer = default(VBuffer<float>);
                    var getter = cursor.GetGetter<VBuffer<float>>(output);
                    var numRows = 0;
                    while (cursor.MoveNext())
                    {
                        getter(ref buffer);
                        Assert.Equal(10, buffer.Length);
                        numRows += 1;
                    }
                    Assert.Equal(3, numRows);
                }
            }
        }
    }
}
