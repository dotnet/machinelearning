using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tests.Transformers
{
    public class LineParserTests
    {
        public static IEnumerable<object[]> ValidInputs()
        {
            foreach (var line in new string[]
            {
                "key 0.1 0.2 0.3", "key 0.1 0.2 0.3 ",
                "key\t0.1\t0.2\t0.3", "key\t0.1\t0.2\t0.3\t" // tab can also be a separator
            })
            {
                yield return new object[] { line, "key", new float[] { 0.1f, 0.2f, 0.3f } };
            }
        }

        [Theory]
        [MemberData(nameof(ValidInputs))]
        public void WhenProvidedAValidInputParserParsesKeyAndValues(string input, string expectedKey, float[] expectedValues)
        {
            var result = Transforms.Text.LineParser.ParseKeyThenNumbers(input);

            Assert.True(result.isSuccess);
            Assert.Equal(expectedKey, result.key);
            Assert.Equal(expectedValues, result.values);
        }

        [Theory]
        [InlineData("")]
        [InlineData("key 0.1 NOT_A_NUMBER")] // invalid number
        public void WhenProvidedAnInvalidInputParserReturnsFailure(string input)
        {
            Assert.False(Transforms.Text.LineParser.ParseKeyThenNumbers(input).isSuccess);
        }
    }
}
