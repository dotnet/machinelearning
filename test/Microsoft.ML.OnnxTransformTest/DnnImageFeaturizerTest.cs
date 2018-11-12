// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Xunit;
using Xunit.Abstractions;
using System.Reflection;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Tests
{
    public class DnnImageFeaturizerTests : TestDataPipeBase
    {
        private const int inputSize = 3 * 224 * 224;

        private class TestData
        {
            [VectorType(inputSize)]
            public float[] data_0;
        }
        private class TestDataSize
        {
            [VectorType(2)]
            public float[] data_0;
        }
        private class TestDataXY
        {
            [VectorType(inputSize)]
            public float[] A;
        }
        private class TestDataDifferntType
        {
            [VectorType(inputSize)]
            public string[] data_0;
        }

        private float[] getSampleArrayData()
        {
            var samplevector = new float[inputSize];
            for (int i = 0; i < inputSize; i++)
                samplevector[i] = (i / ((float) inputSize));
            return samplevector;
        }

        public DnnImageFeaturizerTests(ITestOutputHelper helper) : base(helper)
        {
        }

        // Onnx is only supported on x64 Windows
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))]
        void TestDnnImageFeaturizer()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;
            

            var samplevector = getSampleArrayData();

            var dataView = ComponentCreation.CreateDataView(Env,
                new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    },
                });

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[inputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[inputSize] } };
            var sizeData = new List<TestDataSize> { new TestDataSize() { data_0 = new float[2] } };
            var pipe = new DnnImageFeaturizerEstimator(Env, m => m.ModelSelector.ResNet18(m.Env, m.InputColumn, m.OutputColumn), "data_0", "output_1");

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

        // Onnx is only supported on x64 Windows
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))]
        public void OnnxStatic()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;

            var env = new MLContext(null, 1);
            var imageHeight = 224;
            var imageWidth = 224;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var data = TextLoader.CreateReader(env, ctx => (
                imagePath: ctx.LoadText(0),
                name: ctx.LoadText(1)))
                .Read(dataFile);
          
            var pipe = data.MakeNewEstimator()
                .Append(row => (
                    row.name,
                    data_0: row.imagePath.LoadAsImage(imageFolder).Resize(imageHeight, imageWidth).ExtractPixels(interleaveArgb: true)))
                .Append(row => (row.name, output_1: row.data_0.DnnImageFeaturizer(m => m.ModelSelector.ResNet18(m.Env, m.InputColumn, m.OutputColumn))));

            TestEstimatorCore(pipe.AsDynamic, data.AsDynamic);

            var result = pipe.Fit(data).Transform(data).AsDynamic;
            result.Schema.TryGetColumnIndex("output_1", out int output);
            using (var cursor = result.GetRowCursor(col => col == output))
            {
                var buffer = default(VBuffer<float>);
                var getter = cursor.GetGetter<VBuffer<float>>(output);
                var numRows = 0;
                while (cursor.MoveNext())
                {
                    getter(ref buffer);
                    Assert.Equal(512, buffer.Length);
                    numRows += 1;
                }
                Assert.Equal(3, numRows);
            }
        }

        // Onnx is only supported on x64 Windows
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))]
        public void TestOldSavingAndLoading()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;
            

            var samplevector = getSampleArrayData();

            var dataView = ComponentCreation.CreateDataView(Env,
                new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    }
                });

            var inputNames = "data_0";
            var outputNames = "output_1";
            var est = new DnnImageFeaturizerEstimator(Env, m => m.ModelSelector.ResNet18(m.Env, m.InputColumn, m.OutputColumn), inputNames, outputNames);
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);

                loadedView.Schema.TryGetColumnIndex(outputNames, out int softMaxOut1);
                using (var cursor = loadedView.GetRowCursor(col => col == softMaxOut1))
                {
                    VBuffer<float> softMaxValue = default;
                    var softMaxGetter = cursor.GetGetter<VBuffer<float>>(softMaxOut1);
                    float sum = 0f;
                    int i = 0;
                    while (cursor.MoveNext())
                    {
                        softMaxGetter(ref softMaxValue);
                        var values = softMaxValue.DenseValues();
                        foreach (var val in values)
                        {
                            sum += val;
                            if (i == 0)
                                Assert.InRange(val, 0.0, 0.00001);
                            if (i == 7)
                                Assert.InRange(val, 0.62935, 0.62940);
                            if (i == 500)
                                Assert.InRange(val, 0.15521, 0.155225);
                            i++;
                        }
                    }
                    Assert.InRange(sum, 83.50, 84.50);
                }
            }
        }
    }
}
