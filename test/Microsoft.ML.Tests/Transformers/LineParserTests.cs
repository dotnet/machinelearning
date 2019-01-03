// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Internal.Utilities;
using Xunit;

namespace Microsoft.ML.Tests.Transformers
{
    public class LineParserTests
    {
        public static IEnumerable<object[]> ValidInputs()
        {
            yield return new object[] { "key 0.1 0.2 0.3", "key", new float[] { 0.1f, 0.2f, 0.3f } };
            yield return new object[] { "key 0.1 0.2 0.3 ", "key", new float[] { 0.1f, 0.2f, 0.3f } };
            yield return new object[] { "key\t0.1\t0.2\t0.3", "key", new float[] { 0.1f, 0.2f, 0.3f } }; // tab can also be a separator
            yield return new object[] { "key\t0.1\t0.2\t0.3\t", "key", new float[] { 0.1f, 0.2f, 0.3f } };
        }

        [Theory]
        [MemberData(nameof(ValidInputs))]
        public void WhenProvidedAValidInputParserParsesKeyAndValues(string input, string expectedKey, float[] expectedValues)
        {
            var result = LineParser.ParseKeyThenNumbers(input);

            Assert.True(result.isSuccess);
            Assert.Equal(expectedKey, result.key);
            Assert.Equal(expectedValues, result.values);
        }

        [Theory]
        [InlineData("")]
        [InlineData("key 0.1 NOT_A_NUMBER")] // invalid number
        public void WhenProvidedAnInvalidInputParserReturnsFailure(string input)
        {
            Assert.False(LineParser.ParseKeyThenNumbers(input).isSuccess);
        }
    }
}
