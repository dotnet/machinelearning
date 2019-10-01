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
using System;

namespace Microsoft.ML.Tests
{
    public class OnnxSequenceTypeTest : BaseTestBaseline
    {
        public class OutputObj
        {
            [ColumnName("output")]
            [OnnxSequenceType(typeof(IDictionary<string, float>))]
            public IEnumerable<IDictionary<string, float>> Output;
        }

        public class ProblematicOutputObj
        {

            [ColumnName("output")]
            // incorrect usage, should always specify sequence type when using OnnxSequenceType attribute
            [OnnxSequenceType]
            public IEnumerable<IDictionary<string, float>> Output;
        }

        public class FloatInput
        {
            [ColumnName("input")]
            [VectorType(3)]
            public float[] Input { get; set; }
        }

        public OnnxSequenceTypeTest(ITestOutputHelper output) : base(output)
        {
        }

        private static OnnxTransformer PrepareModel(string onnxModelFilePath, MLContext ctx)
        {
            var dataView = ctx.Data.LoadFromEnumerable(new List<FloatInput>());

            var pipeline = ctx.Transforms.ApplyOnnxModel(
                                modelFile: onnxModelFilePath,
                                outputColumnNames: new[] { "output" }, inputColumnNames: new[] { "input" });

            var model = pipeline.Fit(dataView);
            return model;
        }

        public static PredictionEngine<FloatInput, OutputObj> LoadModel(string onnxModelFilePath)
        {
            var ctx = new MLContext();
            var model = PrepareModel(onnxModelFilePath, ctx);
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

        private static PredictionEngine<FloatInput, ProblematicOutputObj> CreatePredictorWithPronlematicOutputObj()
        {
            var onnxModelFilePath = Path.Combine(Directory.GetCurrentDirectory(), "zipmap", "TestZipMapString.onnx");

            var ctx = new MLContext();
            var model = PrepareModel(onnxModelFilePath, ctx);
            return ctx.Model.CreatePredictionEngine<FloatInput, ProblematicOutputObj>(model);
        }

        [OnnxFact]
        public void OnnxSequenceTypeWithouSpecifySequenceTypeTest()
        {
            Exception ex = Assert.Throws<Exception>(() => CreatePredictorWithPronlematicOutputObj());
            Assert.Equal("Please specify sequence type when use OnnxSequenceType Attribute.", ex.Message);
        }
    }
}
