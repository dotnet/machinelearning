// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data.Conversion;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    
    public class ConversionTests
    {
        private readonly ITestOutputHelper output;

        public ConversionTests(ITestOutputHelper output)
        {
            this.output = output;
        }

        [Fact]
        public void ConvertFloatMissingValues()
        {
            var missingValues = new string[]
            {
                //"",
                "?", " ",
                "na", "n/a", "nan",
                "NA", "N/A", "NaN", "NAN"
            };

            foreach(var missingValue in missingValues)
            {
                float value;
                var success = Conversions.Instance.TryParse(missingValue.AsMemory(), out value);
                output.WriteLine($"{missingValue} parsed as {value}");
                Assert.True(success);
                //Assert.Equal(float.NaN, value);
            }
        }

        [Fact]
        public void ConvertFloatParseFailure()
        {
            var values = new string[]
            {
                "a", "aa", "nb", "aaa", "naa", "nba", "n/b" 
            };

            foreach (var value in values)
            {
                var success = Conversions.Instance.TryParse(value.AsMemory(), out float _);
                Assert.False(success);
            }
        }

        [Fact]
        public void ConvertBoolMissingValues()
        {
            var missingValues = new string[]
            {
                "",
                "no", "NO", "+1", "-1",
                "yes", "YES",
                "true", "TRUE",
                "false", "FALSE"
            };

            foreach (var missingValue in missingValues)
            {
                var success = Conversions.Instance.TryParse(missingValue.AsMemory(), out bool _);
                Assert.True(success);
            }
        }

        [Fact]
        public void ConvertBoolParseFailure()
        {
            var values = new string[]
            {
                "aa", "na", "+a", "-a",
                "aaa", "yaa", "yea",
                "aaaa", "taaa", "traa", "trua",
                "aaaaa", "fbbbb", "faaaa", "falaa", "falsa"
            };

            foreach (var value in values)
            {
                var success = Conversions.Instance.TryParse(value.AsMemory(), out bool _);
                Assert.False(success);
            }
        }
    }
}
