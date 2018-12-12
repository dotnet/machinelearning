// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TensorFlow;
using System;
using System.Collections.Generic;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class TensorFlowEstimatorTests : TestDataPipeBase
    {
        private class TestData
        {
            [VectorType(4)]
            public float[] a;
            [VectorType(4)]
            public float[] b;
        }
        private class TestDataSize
        {
            [VectorType(2)]
            public float[] a;
            [VectorType(2)]
            public float[] b;
        }
        private class TestDataXY
        {
            [VectorType(4)]
            public float[] A;
            [VectorType(4)]
            public float[] B;
        }
        private class TestDataDifferntType
        {
            [VectorType(4)]
            public string[] a;
            [VectorType(4)]
            public string[] b;
        }

        public TensorFlowEstimatorTests(ITestOutputHelper output) : base(output)
        {
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        void TestSimpleCase()
        {
            var modelFile = "model_matmul/frozen_saved_model.pb";

            var dataView = ComponentCreation.CreateDataView(Env,
                new List<TestData>(new TestData[] {
                    new TestData()
                    {
                        a = new[] { 1.0f, 2.0f,3.0f, 4.0f },
                        b = new[] { 1.0f, 2.0f,3.0f, 4.0f }
                    },
                     new TestData()
                     {
                         a = new[] { 2.0f, 2.0f,2.0f, 2.0f },
                         b = new[] { 3.0f, 3.0f,3.0f, 3.0f }
                     }
                }));

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[4], B = new float[4] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { a = new string[4], b = new string[4] } };
            var sizeData = new List<TestDataSize> { new TestDataSize() { a = new float[2], b = new float[2] } };
            var pipe = new TensorFlowEstimator(Env, modelFile, new[] { "a", "b" }, new[] { "c" });

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

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        void TestOldSavingAndLoading()
        {
            var modelFile = "model_matmul/frozen_saved_model.pb";

            var dataView = ComponentCreation.CreateDataView(Env,
                new List<TestData>(new TestData[] {
                    new TestData()
                    {
                        a = new[] { 1.0f, 2.0f, 3.0f, 4.0f },
                        b = new[] { 1.0f, 2.0f, 3.0f, 4.0f }
                    },
                    new TestData()
                    {
                        a = new[] { 2.0f, 2.0f, 2.0f, 2.0f },
                        b = new[] { 3.0f, 3.0f, 3.0f, 3.0f }
                    },
                    new TestData()
                    {
                        a = new[] { 5.0f, 6.0f, 10.0f, 12.0f },
                        b = new[] { 10.0f, 8.0f, 6.0f, 6.0f }
                    }
                }));
            var est = new TensorFlowEstimator(Env, modelFile, new[] { "a", "b" }, new[] { "c" });
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
                ValidateTensorFlowTransformer(loadedView);
            }
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // x86 output differs from Baseline
        void TestCommandLine()
        {
            var env = new MLContext();
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=a:R4:0-3 col=b:R4:0-3} xf=TFTransform{inputs=a inputs=b outputs=c modellocation={model_matmul/frozen_saved_model.pb}}" }), (int)0);
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TestTensorFlowStatic()
        {
            var modelLocation = "cifar_model/frozen_model.pb";

            var mlContext = new MLContext(seed: 1, conc: 1);
            var imageHeight = 32;
            var imageWidth = 32;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var data = TextLoader.CreateReader(mlContext, ctx => (
                imagePath: ctx.LoadText(0),
                name: ctx.LoadText(1)))
                .Read(dataFile);

            // Note that CamelCase column names are there to match the TF graph node names.
            var pipe = data.MakeNewEstimator()
                .Append(row => (
                    row.name,
                    Input: row.imagePath.LoadAsImage(imageFolder).Resize(imageHeight, imageWidth).ExtractPixels(interleaveArgb: true)))
                .Append(row => (row.name, Output: row.Input.ApplyTensorFlowGraph(modelLocation)));

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
                Assert.Equal(4, numRows);
            }
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))]
        public void TestTensorFlowStaticWithSchema()
        {
            const string modelLocation = "cifar_model/frozen_model.pb";

            var mlContext = new MLContext(seed: 1, conc: 1);
            var tensorFlowModel = TensorFlowUtils.LoadTensorFlowModel(mlContext, modelLocation);
            var schema = tensorFlowModel.GetInputSchema();
            Assert.True(schema.TryGetColumnIndex("Input", out int column));
            var type = (VectorType)schema.GetColumnType(column);
            var imageHeight = type.Dimensions[0];
            var imageWidth = type.Dimensions[1];

            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var data = TextLoader.CreateReader(mlContext, ctx => (
                imagePath: ctx.LoadText(0),
                name: ctx.LoadText(1)))
                .Read(dataFile);

            // Note that CamelCase column names are there to match the TF graph node names.
            var pipe = data.MakeNewEstimator()
                .Append(row => (
                    row.name,
                    Input: row.imagePath.LoadAsImage(imageFolder).Resize(imageHeight, imageWidth).ExtractPixels(interleaveArgb: true)))
                .Append(row => (row.name, Output: row.Input.ApplyTensorFlowGraph(tensorFlowModel)));

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
                Assert.Equal(4, numRows);
            }
        }

        private void ValidateTensorFlowTransformer(IDataView result)
        {
            result.Schema.TryGetColumnIndex("a", out int ColA);
            result.Schema.TryGetColumnIndex("b", out int ColB);
            result.Schema.TryGetColumnIndex("c", out int ColC);
            using (var cursor = result.GetRowCursor(x => true))
            {
                VBuffer<float> avalue = default;
                VBuffer<float> bvalue = default;
                VBuffer<float> cvalue = default;

                var aGetter = cursor.GetGetter<VBuffer<float>>(ColA);
                var bGetter = cursor.GetGetter<VBuffer<float>>(ColB);
                var cGetter = cursor.GetGetter<VBuffer<float>>(ColC);
                while (cursor.MoveNext())
                {
                    aGetter(ref avalue);
                    bGetter(ref bvalue);
                    cGetter(ref cvalue);
                    var aValues = avalue.GetValues();
                    var bValues = bvalue.GetValues();
                    var cValues = cvalue.GetValues();
                    Assert.Equal(aValues[0] * bValues[0] + aValues[1] * bValues[2], cValues[0]);
                    Assert.Equal(aValues[0] * bValues[1] + aValues[1] * bValues[3], cValues[1]);
                    Assert.Equal(aValues[2] * bValues[0] + aValues[3] * bValues[2], cValues[2]);
                    Assert.Equal(aValues[2] * bValues[1] + aValues[3] * bValues[3], cValues[3]);
                }
            }
        }
    }
}
