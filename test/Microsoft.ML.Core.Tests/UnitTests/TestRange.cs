// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Xunit;
namespace Microsoft.ML.RunTests
{
    public sealed class RangeTests
    {
        [Fact]
        public void TestRangeParseRange()
        {
            var range = Range.Parse("1-3");
            Assert.Equal(1, range.Min);
            Assert.Equal(3, range.Max); 
        }

        [Fact]
        public void TestRangeParseSingle()
        {
            var range = Range.Parse("1");
            Assert.Equal(1, range.Min);
            Assert.Equal(1, range.Max); 
        }

        [Fact]
        public void TestRangeParseInvalid()
        {
            var range = Range.Parse("foo");
            Assert.Null(range);
        }

        [Fact]
        public void TestRangeParseMaxGreaterThanMin()
        {
            var range = Range.Parse("3-1");
            Assert.Null(range);
        }

        [Fact]
        public void TestRangeParseMaxStar()
        {
            var range = Range.Parse("0-*");
            Assert.Equal(0,range.Min);
            Assert.Null(range.Max);
        }
    }
}
