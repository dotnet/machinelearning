// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class RangeFilterTests : TestDataPipeBase
    {
        public RangeFilterTests(ITestOutputHelper helper) : base(helper)
        {
        }

        [Fact]
        public void RangeFilterTest()
        {
            var builder = new ArrayDataViewBuilder(ML);
            builder.AddColumn("Strings", new[] { "foo", "bar", "baz" });
            builder.AddColumn("Floats", NumberDataViewType.Single, new float[] { 1, 2, 3 });
            var data = builder.GetDataView();

            var data1 = ML.Data.FilterRowsByColumn(data, "Floats", upperBound: 2.8);
            var cnt = data1.GetColumn<float>(data1.Schema["Floats"]).Count();
            Assert.Equal(2L, cnt);

            data = ML.Transforms.Conversion.Hash("Key", "Strings", numberOfBits: 20).Fit(data).Transform(data);
            var data2 = ML.Data.FilterRowsByKeyColumnFraction(data, "Key", upperBound: 0.15);
            cnt = data2.GetColumn<float>(data.Schema["Floats"]).Count();
            Assert.Equal(1L, cnt);
        }

        [Fact]
        public void RangeFilterDoubleColumn()
        {
            var builder = new ArrayDataViewBuilder(ML);
            builder.AddColumn("Doubles", NumberDataViewType.Double, new double[] { 1, 2, 3 });
            var data = builder.GetDataView();

            var filtered = ML.Data.FilterRowsByColumn(data, "Doubles", lowerBound: 1.5);
            var values = filtered.GetColumn<double>(filtered.Schema["Doubles"]).ToArray();
            Assert.Equal(new double[] { 2, 3 }, values);
        }

        [Fact]
        public void RangeFilterRemovesNaN()
        {
            var builder = new ArrayDataViewBuilder(ML);
            builder.AddColumn("Floats", NumberDataViewType.Single, new float[] { 1, float.NaN, 3 });
            var data = builder.GetDataView();

            var filtered = ML.Data.FilterRowsByColumn(data, "Floats");
            var values = filtered.GetColumn<float>(filtered.Schema["Floats"]).ToArray();
            Assert.Equal(new float[] { 1, 3 }, values);
        }

        [Fact]
        public void RangeFilterRejectsIntegerColumn()
        {
            var builder = new ArrayDataViewBuilder(ML);
            builder.AddColumn("Ints", NumberDataViewType.Int32, new int[] { 1, 2, 3 });
            var data = builder.GetDataView();

            Assert.Throws<ArgumentOutOfRangeException>(() => ML.Data.FilterRowsByColumn(data, "Ints"));
        }
    }
}
