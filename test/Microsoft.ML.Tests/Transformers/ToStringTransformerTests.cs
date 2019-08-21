// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Featurizers;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class ToStringTransformerTests : TestDataPipeBase
    {
        private class BoolInput
        {
            public bool data;
        }

        private class ByteInput
        {
            public byte data;
        }

        private class SByteInput
        {
            public sbyte data;
        }

        private class ShortInput
        {
            public short data;
        }

        private class UShortInput
        {
            public ushort data;
        }

        private class IntInput
        {
            public int data;
        }

        private class UIntInput
        {
            public uint data;
        }

        private class LongInput
        {
            public long data;
        }

        private class ULongInput
        {
            public ulong data;
        }

        private class FloatInput
        {
            public float data;
        }

        private class DoubleInput
        {
            public double data;
        }
        private class StringInput
        {
            public string data;
        }

        public ToStringTransformerTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestBool()
        {
            MLContext mlContext = new MLContext(1);            
            var data = new[] { new BoolInput() { data = true }, new BoolInput() { data = false } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("output", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            Assert.Equal("True", rows[0].Values[1].Value.ToString()); 
            Assert.Equal("False", rows[1].Values[1].Value.ToString()); 
            Done();
        }

        
        [Fact]
        public void TestByte()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new ByteInput() { data = byte.MinValue }, new ByteInput() { data = byte.MaxValue } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("output", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            Assert.Equal("0", rows[0].Values[1].Value.ToString()); 
            Assert.Equal("255", rows[1].Values[1].Value.ToString()); 
            Done();
        }

        [Fact]
        public void TestSByte()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new SByteInput() { data = sbyte.MinValue }, new SByteInput() { data = sbyte.MaxValue } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("output", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            Assert.Equal("-128", rows[0].Values[1].Value.ToString()); 
            Assert.Equal("127", rows[1].Values[1].Value.ToString()); 
            Done();
        }
        
        [Fact]
        public void TestUShort()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new UShortInput() { data = ushort.MinValue }, new UShortInput() { data = ushort.MaxValue } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("output", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            Assert.Equal("0", rows[0].Values[1].Value.ToString()); 
            Assert.Equal("65535", rows[1].Values[1].Value.ToString()); 
            Done();
        }
        
        [Fact]
        public void TestInt()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new IntInput() { data = int.MinValue }, new IntInput() { data = int.MaxValue } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("output", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            Assert.Equal("-2147483648", rows[0].Values[1].Value.ToString()); 
            Assert.Equal("2147483647", rows[1].Values[1].Value.ToString()); 
            Done();
        }
        
        [Fact]
        public void TestUInt()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new UIntInput() { data = uint.MinValue }, new UIntInput() { data = uint.MaxValue } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("output", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            Assert.Equal("0", rows[0].Values[1].Value.ToString()); 
            Assert.Equal("4294967295", rows[1].Values[1].Value.ToString()); 
            Done();
        }        
        
        [Fact]
        public void TestLong()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new LongInput() { data = long.MinValue }, new LongInput() { data = long.MaxValue } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("output", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            Assert.Equal("-9223372036854775808", rows[0].Values[1].Value.ToString()); 
            Assert.Equal("9223372036854775807", rows[1].Values[1].Value.ToString()); 
            Done();
        }

        [Fact]
        public void TestULong()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new ULongInput() { data = ulong.MinValue }, new ULongInput() { data = ulong.MaxValue } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("output", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            Assert.Equal("0", rows[0].Values[1].Value.ToString());
            Assert.Equal("18446744073709551615", rows[1].Values[1].Value.ToString());
            Done();
        }

        [Fact]
        public void TestFloat()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new FloatInput() { data = float.MinValue}, new FloatInput() { data = float.MaxValue }, new FloatInput() { data = float.NaN } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("output", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            Assert.Equal("-340282346638528859811704183484516925440.000000", rows[0].Values[1].Value.ToString());
            Assert.Equal("340282346638528859811704183484516925440.000000", rows[1].Values[1].Value.ToString());
            Done();
        }
        
        [Fact]
        public void TestShort()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new ShortInput() { data = short.MinValue }, new ShortInput() { data = short.MaxValue } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("output", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            Assert.Equal("-32768", rows[0].Values[1].Value.ToString()); 
            Assert.Equal("32767", rows[1].Values[1].Value.ToString()); 
            Done();
        }
        
        [Fact]
        public void TestDouble()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new DoubleInput() { data = double.MinValue}, new DoubleInput() { data = double.MaxValue }, new DoubleInput() { data = double.NaN } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("data.out", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            // Since we can't set the precision yet on the Native side and it returns the whole string value, only checking the first 10 places.
            Assert.Equal(double.MinValue.ToString("F10").Substring(0,10), rows[0].Values[1].Value.ToString().Substring(0, 10));
            Assert.Equal(double.MaxValue.ToString("F10").Substring(0, 10), rows[1].Values[1].Value.ToString().Substring(0, 10));

            Done();
        }

        [Fact]
        public void TestString()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new StringInput() { data = ""}, new StringInput() { data = "Long Dummy String Value" } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = mlContext.Transforms.ToStringTransformer("output", "data");

            TestEstimatorCore(pipeline, input);

            var model = pipeline.Fit(input);
            var output = model.Transform(input);
            var rows = output.Preview().RowView;

            Assert.Equal("", rows[0].Values[1].Value.ToString());
            Assert.Equal("Long Dummy String Value", rows[1].Values[1].Value.ToString());
            Done();



        }
        
        [Fact]
        public void TestEntryPoint()
        {
            MLContext mlContext = new MLContext(1);

            var data = new[] { new StringInput() { data = ""}, new StringInput() { data = "Long Dummy String Value" } };
            IDataView input = mlContext.Data.LoadFromEnumerable(data);

            var options = new ToStringTransformerEstimator.Options()
            {
                Columns = new ToStringTransformerEstimator.Column[1]
                {
                    new ToStringTransformerEstimator.Column()
                    {
                        Name = "data"
                    }
                },
                Data = input
            };

            var output = ToStringTransformerEntrypoint.ToString(mlContext.Transforms.GetEnvironment(), options);

            var rows = output.OutputData.Preview().RowView;

            Assert.Equal("", rows[0].Values[1].Value.ToString());
            Assert.Equal("Long Dummy String Value", rows[1].Values[1].Value.ToString());
            Done();
        }
    }
}
