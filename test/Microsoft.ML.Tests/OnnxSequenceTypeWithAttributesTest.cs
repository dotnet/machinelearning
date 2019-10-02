// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Drawing;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;
using Xunit;
using Xunit.Abstractions;
using System.Linq;
using System.IO;
using Microsoft.ML.TestFramework.Attributes;

namespace Microsoft.ML.Tests
{
    public class OnnxSequenceTypeWithAttributesTest : BaseTestBaseline
    {
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
            var ctx = new MLContext();
            var dataView = ctx.Data.LoadFromEnumerable(new List<FloatInput>());

            var pipeline = ctx.Transforms.ApplyOnnxModel(
                                modelFile: onnxModelFilePath,
                                outputColumnNames: new[] { "output" }, inputColumnNames: new[] { "input" });

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
            var onnx_out = output.Output.FirstOrDefault();
            Assert.True(onnx_out.Count == 3, "Output missing data.");
            var keys = new List<string>(onnx_out.Keys);
            for(var i =0; i < onnx_out.Count; ++i)
            {
                Assert.Equal(onnx_out[keys[i]], input.Input[i]);
            }

        }
    }
}