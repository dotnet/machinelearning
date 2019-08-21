// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;
using Microsoft.ML.RunTests;
using System;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;
using System.Linq;

namespace Microsoft.ML.Tests.Transformers
{
    public class RobustScalerTests : TestDataPipeBase
    {
        public RobustScalerTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestInvalidType()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = "Invalid Type" }};
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA");
            
            Assert.Throws<InvalidOperationException>(() => pipeline.Fit(data));
            Assert.Throws<InvalidOperationException>(() => pipeline.GetOutputSchema(SchemaShape.Create(data.Schema)));

            Done();
        }
        
        [Fact]
        public void TestNoScale()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = 1f }, new { ColA = 3f }, new { ColA = 5f }, new { ColA = 7f }, new { ColA = 9f } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA", scale: false);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(float), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { -4f, -2f, 0f, 2f, 4f };
            var index = 0;
            var getter = cursor.GetGetter<float>(schema["ColA"]);
            float value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
        
        [Fact]
        public void TestNoScaleNoCenter()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = 1f }, new { ColA = 3f }, new { ColA = 5f }, new { ColA = 7f }, new { ColA = 9f } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA", scale: false, center: false);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(float), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { 1f, 3f, 5f, 7f, 9f };
            var index = 0;
            var getter = cursor.GetGetter<float>(schema["ColA"]);
            float value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
        
        [Fact]
        public void TestFloat()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = 1f }, new { ColA = 3f }, new { ColA = 5f }, new { ColA = 7f }, new { ColA = 9f } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(float), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { -1f, -0.5f, 0f, .5f, 1f };
            var index = 0;
            var getter = cursor.GetGetter<float>(schema["ColA"]);
            float value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
        
        [Fact]
        public void TestInt64()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = 1L }, new { ColA = 3L }, new { ColA = 5L }, new { ColA = 7L }, new { ColA = 9L } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;
            var prev = output.Preview();

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(double), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { -1d, -0.5d, 0d, .5d, 1d };
            var index = 0;
            var getter = cursor.GetGetter<double>(schema["ColA"]);
            double value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
        
        [Fact]
        public void TestInt32()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = 1 }, new { ColA = 3 }, new { ColA = 5 }, new { ColA = 7 }, new { ColA = 9 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;
            var prev = output.Preview();

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(double), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { -1d, -0.5d, 0d, .5d, 1d };
            var index = 0;
            var getter = cursor.GetGetter<double>(schema["ColA"]);
            double value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
        
        [Fact]
        public void TestInt16()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = (short)1 }, new { ColA = (short)3 }, new { ColA = (short)5 }, new { ColA = (short)7 }, new { ColA = (short)9 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;
            var prev = output.Preview();

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(float), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { -1f, -0.5f, 0f, .5f, 1f };
            var index = 0;
            var getter = cursor.GetGetter<float>(schema["ColA"]);
            float value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
        
        [Fact]
        public void TestInt8()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = (sbyte)1 }, new { ColA = (sbyte)3 }, new { ColA = (sbyte)5 }, new { ColA = (sbyte)7 }, new { ColA = (sbyte)9 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;
            var prev = output.Preview();

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(float), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { -1f, -0.5f, 0f, .5f, 1f };
            var index = 0;
            var getter = cursor.GetGetter<float>(schema["ColA"]);
            float value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [Fact]
        public void TestDouble()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = 1d }, new { ColA = 3d }, new { ColA = 5d }, new { ColA = 7d }, new { ColA = 9d } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(double), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { -1d, -0.5d, 0d, .5d, 1d };
            var index = 0;
            var getter = cursor.GetGetter<double>(schema["ColA"]);
            double value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
        
        [Fact]
        public void TestUInt64()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = (ulong)1 }, new { ColA = (ulong)3 }, new { ColA = (ulong)5 }, new { ColA = (ulong)7 }, new { ColA = (ulong)9 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;
            var prev = output.Preview();

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(double), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { -1d, -0.5d, 0d, .5d, 1d };
            var index = 0;
            var getter = cursor.GetGetter<double>(schema["ColA"]);
            double value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
        
        [Fact]
        public void TestUInt32()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = (uint)1 }, new { ColA = (uint)3 }, new { ColA = (uint)5 }, new { ColA = (uint)7 }, new { ColA = (uint)9 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;
            var prev = output.Preview();

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(double), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { -1d, -0.5d, 0d, .5d, 1d };
            var index = 0;
            var getter = cursor.GetGetter<double>(schema["ColA"]);
            double value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
        
        [Fact]
        public void TestUInt16()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = (ushort)1 }, new { ColA = (ushort)3 }, new { ColA = (ushort)5 }, new { ColA = (ushort)7 }, new { ColA = (ushort)9 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;
            var prev = output.Preview();

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(float), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { -1f, -0.5f, 0f, .5f, 1f };
            var index = 0;
            var getter = cursor.GetGetter<float>(schema["ColA"]);
            float value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
        
        [Fact]
        public void TestUInt8()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new [] { new { ColA = (byte)1 }, new { ColA = (byte)3 }, new { ColA = (byte)5 }, new { ColA = (byte)7 }, new { ColA = (byte)9 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.RobustScalerTransformer("ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;
            var prev = output.Preview();

            Assert.Single(schema.Where(x => x.IsHidden == false));
            Assert.Equal(typeof(float), schema["ColA"].Type.RawType);

            var cursor = output.GetRowCursor(schema["ColA"]);
            var expectedOutput = new[] { -1f, -0.5f, 0f, .5f, 1f };
            var index = 0;
            var getter = cursor.GetGetter<float>(schema["ColA"]);
            float value = default;

            while (cursor.MoveNext())
            {
                getter(ref value);
                Assert.Equal(expectedOutput[index++], value);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
    }
}