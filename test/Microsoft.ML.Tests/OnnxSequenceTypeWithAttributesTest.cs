// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class OnnxSequenceTypeWithAttributesTest : BaseTestBaseline
    {
        private const bool _fallbackToCpu = true;
        private static int? _gpuDeviceId = null;

        public class OutputObj
        {
            [ColumnName("output")]
            [OnnxSequenceType(typeof(IDictionary<string, float>))]
            public IEnumerable<IDictionary<string, float>> Output;
        }
        public class FloatInput
        {
            [ColumnName("input")]
            [VectorType(3)]
            public float[] Input { get; set; }
        }

        public OnnxSequenceTypeWithAttributesTest(ITestOutputHelper output) : base(output)
        {
        }
        public static PredictionEngine<FloatInput, OutputObj> LoadModel(string onnxModelFilePath)
        {
            var ctx = new MLContext(1);
            var dataView = ctx.Data.LoadFromEnumerable(new List<FloatInput>());

            var pipeline = ctx.Transforms.ApplyOnnxModel(
                                modelFile: onnxModelFilePath,
                                outputColumnNames: new[] { "output" }, inputColumnNames: new[] { "input" },
                                gpuDeviceId: _gpuDeviceId,
                                fallbackToCpu: _fallbackToCpu);

            var model = pipeline.Fit(dataView);
            return ctx.Model.CreatePredictionEngine<FloatInput, OutputObj>(model);
        }

        [OnnxFact]
        public void OnnxSequenceTypeWithColumnNameAttributeTest()
        {
            var modelFile = Path.Combine(Directory.GetCurrentDirectory(), "zipmap", "TestZipMapString.onnx");
            var predictor = LoadModel(modelFile);

            FloatInput input = new FloatInput() { Input = new float[] { 1.0f, 2.0f, 3.0f } };
            var output = predictor.Predict(input);
            var onnxOut = output.Output.FirstOrDefault();
            Assert.True(onnxOut.Count == 3, "Output missing data.");
            var keys = new List<string>(onnxOut.Keys);
            for (var i = 0; i < onnxOut.Count; ++i)
            {
                Assert.Equal(onnxOut[keys[i]], input.Input[i]);
            }
        }

        public class WrongOutputObj
        {
            [ColumnName("output")]
            [OnnxSequenceType(typeof(IEnumerable<float>))]
            public IEnumerable<float> Output;
        }

        public static PredictionEngine<FloatInput, WrongOutputObj> LoadModelWithWrongCustomType(string onnxModelFilePath)
        {
            var ctx = new MLContext(1);
            var dataView = ctx.Data.LoadFromEnumerable(new List<FloatInput>());

            var pipeline = ctx.Transforms.ApplyOnnxModel(
                                modelFile: onnxModelFilePath,
                                outputColumnNames: new[] { "output" }, inputColumnNames: new[] { "input" },
                                gpuDeviceId: _gpuDeviceId,
                                fallbackToCpu: _fallbackToCpu);

            var model = pipeline.Fit(dataView);
            return ctx.Model.CreatePredictionEngine<FloatInput, WrongOutputObj>(model);
        }

        [OnnxFact]
        public void OnnxSequenceTypeWithColumnNameAttributeTestWithWrongCustomType()
        {
            var modelFile = Path.Combine(Directory.GetCurrentDirectory(), "zipmap", "TestZipMapString.onnx");
            var expectedExceptionMessage = "The expected type 'System.Collections.Generic.IEnumerable`1[System.Collections.Generic.IDictionary`2[System.String,System.Single]]'" +
                " does not match the type of the 'output' member: 'System.Collections.Generic.IEnumerable`1[System.Single]'." +
                " Please change the output member to 'System.Collections.Generic.IEnumerable`1[System.Collections.Generic.IDictionary`2[System.String,System.Single]]'";
            try
            {
                var predictor = LoadModelWithWrongCustomType(modelFile);
                Assert.True(false);
            }
            catch (System.Exception ex)
            {
                //truncate the string to only necessary information as Linux and Windows have different way of encoding the string
                Assert.Equal(expectedExceptionMessage, ex.Message.Substring(0, 387));
                return;
            }
        }
    }
}
