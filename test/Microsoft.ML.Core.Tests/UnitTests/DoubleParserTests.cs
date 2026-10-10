// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public class DoubleParserTests : BaseTestClass
    {
        public DoubleParserTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void Parse_WithLeadingAndTrailingWhitespace_ReturnsGood()
        {
            var result = DoubleParser.Parse(" 1.234 ".AsSpan(), out double value);

            Assert.Equal(DoubleParser.Result.Good, result);
            Assert.Equal(1.234, value);
        }

        [Fact]
        public void Parse_WithoutWhitespace_ReturnsGood()
        {
            var result = DoubleParser.Parse("1.234".AsSpan(), out double value);

            Assert.Equal(DoubleParser.Result.Good, result);
            Assert.Equal(1.234, value);
        }

        [Fact]
        public void Parse_WithTrailingGarbageCharacter_ReturnsExtra()
        {
            var result = DoubleParser.Parse("1.234x".AsSpan(), out double value);

            Assert.Equal(DoubleParser.Result.Extra, result);
        }

        [Fact]
        public void Parse_Single_WithLeadingAndTrailingWhitespace_ReturnsGood()
        {
            var result = DoubleParser.Parse(" 1.234 ".AsSpan(), out float value);

            Assert.Equal(DoubleParser.Result.Good, result);
            Assert.Equal(1.234f, value);
        }
    }
}
