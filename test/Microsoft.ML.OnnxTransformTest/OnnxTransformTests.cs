// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class OnnxTransformTests : TestDataPipeBase
    {

        private const int inputSize = 150528;

        private class TestData
        {
            [VectorType(inputSize)]
            public float[] data_0;
        }

        private class TestDataMulti
        {
            [VectorType(5)]
            public float[] ina;

            [VectorType(5)]
            public float[] inb;
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

        public OnnxTransformTests(ITestOutputHelper output) : base(output)
        {
        }
        
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // x86 fails with "An attempt was made to load a program with an incorrect format."
        void TestSimpleCase()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;

            var modelFile = "squeezenet/00000001/model.onnx";

            var samplevector = getSampleArrayData();

            var dataView = ComponentCreation.CreateDataView(Env,
                new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    },
                     new TestData()
                     {
                        data_0 = samplevector
                     }
                });

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[inputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[inputSize] } };
            var sizeData = new List<TestDataSize> { new TestDataSize() { data_0 = new float[2] } };
            var pipe = new OnnxScoringEstimator(Env, modelFile, new[] { "data_0" }, new[] { "softmaxout_1" });

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
        void TestOldSavingAndLoading()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;

            var modelFile = "squeezenet/00000001/model.onnx";

            var samplevector = getSampleArrayData();

            var dataView = ComponentCreation.CreateDataView(Env,
                new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    }
                });

            var inputNames = new[] { "data_0" };
            var outputNames = new[] { "softmaxout_1" };
            var est = new OnnxScoringEstimator(Env, modelFile, inputNames, outputNames);
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);

                loadedView.Schema.TryGetColumnIndex(outputNames[0], out int softMaxOut1);
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
                                Assert.InRange(val, 0.00004, 0.00005);
                            if (i == 1)
                                Assert.InRange(val, 0.003844, 0.003845);
                            if (i == 999)
                                Assert.InRange(val, 0.0029566, 0.0029567);
                            i++;
                        }
                    }
                    Assert.InRange(sum, 1.0, 1.00001);
                }
            }
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // x86 fails with "An attempt was made to load a program with an incorrect format."
        public void OnnxStatic()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;

            var modelFile = "squeezenet/00000001/model.onnx";

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
                    .Append(row => (row.name, softmaxout_1: row.data_0.ApplyOnnxModel(modelFile)));

                TestEstimatorCore(pipe.AsDynamic, data.AsDynamic);

                var result = pipe.Fit(data).Transform(data).AsDynamic;
                result.Schema.TryGetColumnIndex("softmaxout_1", out int output);
                using (var cursor = result.GetRowCursor(col => col == output))
                {
                    var buffer = default(VBuffer<float>);
                    var getter = cursor.GetGetter<VBuffer<float>>(output);
                    var numRows = 0;
                    while (cursor.MoveNext())
                    {
                        getter(ref buffer);
                        Assert.Equal(1000, buffer.Length);
                        numRows += 1;
                    }
                    Assert.Equal(3, numRows);
                }
            }
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // x86 output differs from Baseline
        void TestCommandLine()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;

            using (var env = new ConsoleEnvironment())
            {
                var x = Maml.Main(new[] { @"showschema loader=Text{col=data_0:R4:0-150527} xf=Onnx{InputColumns={data_0} OutputColumns={softmaxout_1} model={squeezenet/00000001/model.onnx}}" });
                Assert.Equal(0, x);
            }
        }
        
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // x86 output differs from Baseline
        public void OnnxModelScenario()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;

            var modelFile = "squeezenet/00000001/model.onnx";
            using (var env = new ConsoleEnvironment(seed: 1, conc: 1))
            {
                var samplevector = getSampleArrayData();

                var dataView = ComponentCreation.CreateDataView(Env,
                    new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    }
                    });

                var onnx = OnnxTransform.Create(env, dataView, modelFile,
                    new[] { "data_0" },
                    new[] { "softmaxout_1" });

                onnx.Schema.TryGetColumnIndex("softmaxout_1", out int scores);
                using (var curs = onnx.GetRowCursor(col => col == scores))
                {
                    var getScores = curs.GetGetter<VBuffer<float>>(scores);
                    var buffer = default(VBuffer<float>);
                    while (curs.MoveNext())
                    {
                        getScores(ref buffer);
                        Assert.Equal(1000, buffer.Length);
                    }
                }
            }
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // x86 output differs from Baseline
        public void OnnxModelMultiInput()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;

            var modelFile = @"D:\models\pythoncreatemodel\twoinput\twoinput.onnx";
            using (var env = new ConsoleEnvironment(seed: 1, conc: 1))
            {
                var samplevector = getSampleArrayData();

                var dataView = ComponentCreation.CreateDataView(Env,
                    new TestDataMulti[] {
                    new TestDataMulti()
                    {
                        ina = new float[] {1,2,3,4,5},
                        inb = new float[] {1,2,3,4,5}
                    }
                    });

                var onnx = OnnxTransform.Create(env, dataView, modelFile,
                    new[] { "ina", "inb" },
                    new[] { "outa", "outb" });

                onnx.Schema.TryGetColumnIndex("outa", out int scoresa);
                onnx.Schema.TryGetColumnIndex("outb", out int scoresb);
                using (var curs = onnx.GetRowCursor(col => col == scoresa || col == scoresb))
                {
                    var getScoresa = curs.GetGetter<VBuffer<float>>(scoresa);
                    var getScoresb = curs.GetGetter<VBuffer<float>>(scoresb);
                    var buffera = default(VBuffer<float>);
                    var bufferb = default(VBuffer<float>);

                    while (curs.MoveNext())
                    {
                        getScoresa(ref buffera);
                        getScoresb(ref bufferb);
                        Console.WriteLine(buffera.Values);
                        Assert.Equal(5, buffera.Length);
                    }
                }
            }
        }
    }
}

