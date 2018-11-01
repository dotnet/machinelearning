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
                samplevector[i] = (i / (inputSize * 1.01f));
            return samplevector;
        }

        public DnnImageFeaturizerTests(ITestOutputHelper helper) : base(helper)
        {
        }

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
            var pipe = new DnnImageFeaturizerEstimator(Env, DnnImageFeaturizerEstimator.DnnImageModel.ResNet18, "data_0", "output_1");

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

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // x86 fails with "An attempt was made to load a program with an incorrect format."
        public void OnnxStatic()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;

            using (var env = new ConsoleEnvironment(null, false, 0, 1, null, null))
            {
                var imageHeight = 224;
                var imageWidth = 224;
                var dataFile = GetDataPath("images/images.tsv");
                var imageFolder = Path.GetDirectoryName(dataFile);

                var data = TextLoader.CreateReader(env, ctx => (
                    imagePath: ctx.LoadText(0),
                    name: ctx.LoadText(1)))
                    .Read(dataFile);

                // Note that CamelCase column names are there to match the TF graph node names.
                var pipe = data.MakeNewEstimator()
                    .Append(row => (
                        row.name,
                        data_0: row.imagePath.LoadAsImage(imageFolder).Resize(imageHeight, imageWidth).ExtractPixels(interleaveArgb: true)))
                    .Append(row => (row.name, output_1: row.data_0.DnnImageFeaturizer(DnnImageFeaturizerEstimator.DnnImageModel.ResNet18)));

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
        }
    }
}
