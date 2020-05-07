// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class LineParserTests : BaseTestClass
    {
        public LineParserTests(ITestOutputHelper output) : base(output)
        {
        }

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
            var result = LineParser.ParseKeyThenNumbers(input, true);

            Assert.True(result.isSuccess);
            Assert.Equal(expectedKey, result.key);
            Assert.Equal(expectedValues, result.values);
        }


        [Theory]
        [InlineData("")]
        [InlineData("key 0.1 NOT_A_NUMBER")] // invalid number
        public void WhenProvidedAnInvalidInputParserReturnsFailure(string input)
        {
            Assert.False(LineParser.ParseKeyThenNumbers(input, true).isSuccess);
        }

        [Theory]
        [IterationData(iterations: 20)]
        [Trait("Category", "RunSpecificTest")]
        public void CompleteLineParserAndCulture(int iterations)
        {
            Output.WriteLine($"{iterations} - th");

            int timeout = 20 * 60 * 1000;

            var runTask = Task.Run(LineParserAndCulture);
            var timeoutTask = Task.Delay(timeout + iterations);
            var finishedTask = Task.WhenAny(timeoutTask, runTask).Result;
            if (finishedTask == timeoutTask)
            {
                Console.WriteLine("LineParserAndCulture test Hanging: fail to complete in 20 minutes");
                Environment.FailFast("Fail here to take memory dump");
            }
        }

        [Fact]
        public void LineParserAndCulture()
        {
            var currentCulture = Thread.CurrentThread.CurrentCulture;
            Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("de-DE");
            Random rand = new Random();
            var arraySize = 100;
            var data = new float[arraySize];
            var separator = new string[2] { " ", "\t" };

            for (int sep = 0; sep < 4; sep++)
            {
                for (int i = 0; i < arraySize; i++)
                    data[i] = rand.NextSingle() * 50 - 25;
                var result = LineParser.ParseKeyThenNumbers("word" + separator[sep % 2] + string.Join(separator[sep / 2], data.Select(x=>x.ToString("G9"))), false);
                Assert.True(result.isSuccess);
                Assert.Equal("word", result.key);
                for (int i = 0; i < arraySize; i++)
                    Assert.Equal(data[i], result.values[i]);
            }


            Thread.CurrentThread.CurrentCulture = currentCulture;
        }
    }
}
