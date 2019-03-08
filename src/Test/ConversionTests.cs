// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class ConversionTests
    {
        [TestMethod]
        public void ConvertFloatMissingValues()
        {
            var missingValues = new string[]
            {
                "",
                "?", " ",
                "na", "n/a", "nan",
                "NA", "N/A", "NaN", "NAN"
            };

            foreach(var missingValue in missingValues)
            {
                float value;
                var success = Conversions.TryParse(missingValue.AsMemory(), out value);
                Assert.IsTrue(success);
                Assert.AreEqual(value, float.NaN);
            }
        }

        [TestMethod]
        public void ConvertFloatParseFailure()
        {
            var values = new string[]
            {
                "a", "aa", "nb", "aaa", "naa", "nba", "n/b" 
            };

            foreach (var value in values)
            {
                var success = Conversions.TryParse(value.AsMemory(), out float _);
                Assert.IsFalse(success);
            }
        }

        [TestMethod]
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
                var success = Conversions.TryParse(missingValue.AsMemory(), out bool _);
                Assert.IsTrue(success);
            }
        }

        [TestMethod]
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
                var success = Conversions.TryParse(value.AsMemory(), out bool _);
                Assert.IsFalse(success);
            }
        }
    }
}
