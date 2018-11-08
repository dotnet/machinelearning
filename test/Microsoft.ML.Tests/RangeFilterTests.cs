// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using System.Linq;
using System.Threading;
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
            builder.AddColumn("Floats", NumberType.R4, new float[] { 1, 2, 3 });
            var data = builder.GetDataView();

            var data1 = ML.Data.FilterByColumn(data, "Floats", upperBound: 2.8);
            var cnt = data1.GetColumn<float>(ML, "Floats").Count();
            Assert.Equal(2L, cnt);

            data = ML.Transforms.Conversion.Hash("Strings", "Key", hashBits: 20).Fit(data).Transform(data);
            var data2 = ML.Data.FilterByKeyColumnFraction(data, "Key", upperBound: 0.5);
            cnt = data2.GetColumn<float>(ML, "Floats").Count();
            Assert.Equal(1L, cnt);
        }
    }
}
