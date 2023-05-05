// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public class TestUtilities : BaseTestBaseline
    {
        public TestUtilities(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsMonotonicallyIncreasingInt()
        {
            // A sorted (increasing) array
            int[] x = Enumerable.Range(0, 10).ToArray();
            Assert.True(Utils.IsMonotonicallyIncreasing(x));

            // A monotonically increasing array
            var x1Temp = x[1];
            var x7Temp = x[7];
            x[1] = x[0];
            x[7] = x[6];
            Assert.True(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;
            x[7] = x7Temp;

            // Not sorted
            x[1] = x[6];
            Assert.False(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;

            // Null lists are considered to be sorted
            int[] nullX = null;
            Assert.True(Utils.IsMonotonicallyIncreasing(nullX));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsMonotonicallyIncreasingFloat()
        {
            // A sorted (increasing) array
            List<float> x = Enumerable.Range(0, 1000000).Select(i => (float)i).ToList();
            Assert.True(Utils.IsMonotonicallyIncreasing(x));

            // A monotonically increasing array
            var x1Temp = x[1];
            var x7Temp = x[7];
            x[1] = x[0];
            x[7] = x[6];
            Assert.True(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;
            x[7] = x7Temp;

            // Not sorted
            x[1] = x[6];
            Assert.False(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;

            // NaN: `Array.Sort()` will put NaNs into the first position,
            // but we want to guarantee that NaNs aren't allowed in these arrays.
            var x0Temp = x[0];
            x[0] = float.NaN;
            Assert.False(Utils.IsMonotonicallyIncreasing(x));
            x[0] = x0Temp;

            // Null lists are considered to be sorted
            List<float> nullX = null;
            Assert.True(Utils.IsMonotonicallyIncreasing(nullX));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsMonotonicallyIncreasingDouble()
        {
            // A sorted (increasing) array
            double[] x = Enumerable.Range(0, 1000000).Select(i => (double)i).ToArray();
            Assert.True(Utils.IsMonotonicallyIncreasing(x));

            // A monotonically increasing array
            var x1Temp = x[1];
            var x7Temp = x[7];
            x[1] = x[0];
            x[7] = x[6];
            Assert.True(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;
            x[7] = x7Temp;

            // Not sorted
            x[1] = x[6];
            Assert.False(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;

            // NaN: `Array.Sort()` will put NaNs into the first position,
            // but we want to guarantee that NaNs aren't allowed in these arrays.
            var x0Temp = x[0];
            x[0] = float.NaN;
            Assert.False(Utils.IsMonotonicallyIncreasing(x));
            x[0] = x0Temp;

            // Null lists are considered to be sorted
            List<float> nullX = null;
            Assert.True(Utils.IsMonotonicallyIncreasing(nullX));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsIncreasing()
        {
            // An increasing array
            int[] x = Enumerable.Range(0, 10).ToArray();
            Assert.True(Utils.IsIncreasing(0, x, 10));
            // Check the lower bound
            Assert.False(Utils.IsIncreasing(1, x, 10));
            // The upper bound should be exclusive
            Assert.False(Utils.IsIncreasing(0, x, 9));
            // Any length shorter than the array should work
            Assert.True(Utils.IsIncreasing(0, x, 0, 10));
            Assert.True(Utils.IsIncreasing(0, x, 1, 10));
            Assert.True(Utils.IsIncreasing(0, x, 5, 10));
            Assert.True(Utils.IsIncreasing(0, x, 10, 10));
            // Lengths longer than the array shall throw
            Assert.Throws<InvalidOperationException>(() => Utils.IsIncreasing(0, x, 11, 10));

            // A monotonically increasing array should fail
            var x7Temp = x[7];
            x[7] = x[6];
            Assert.False(Utils.IsIncreasing(0, x, 10));
            // But until then, it should be fine
            Assert.True(Utils.IsIncreasing(0, x, 7, 10));
            x[7] = x7Temp;

            // Not sorted
            x[7] = x[9];
            Assert.False(Utils.IsIncreasing(0, x, 10));
            // Before the mismatched entry, it should be fine
            Assert.True(Utils.IsIncreasing(0, x, 7, 10));
            x[1] = x7Temp;

            // Null arrays return true
            int[] nullX = null;
            Assert.True(Utils.IsIncreasing(0, nullX, 10));

            // Null arrays with a length accession shall throw an exception
            Assert.Throws<InvalidOperationException>(() => Utils.IsIncreasing(0, nullX, 7, 10));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckAreEqualInt()
        {
            // A sorted (increasing) array
            int[] x = Enumerable.Range(0, 10).ToArray();
            int[] y = Enumerable.Range(0, 10).ToArray();
            Assert.True(Utils.AreEqual(x, y));

            // Not Equal
            var x1Temp = x[1];
            x[1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[1] = x1Temp;

            // Beginning is different
            var x0Temp = x[0];
            x[0] = x[x.Length - 1];
            Assert.False(Utils.AreEqual(x, y));
            x[0] = x0Temp;

            // End is different
            var xLengthTemp = x[x.Length - 1];
            x[x.Length - 1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[x.Length - 1] = xLengthTemp;

            // Different Array Lengths
            int[] xOfDifferentLength = Enumerable.Range(0, 9).ToArray();
            Assert.False(Utils.AreEqual(xOfDifferentLength, y));

            // Nulls
            Assert.False(Utils.AreEqual(null, y));
            Assert.False(Utils.AreEqual(x, null));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckAreEqualBool()
        {
            // A sorted (increasing) array
            bool[] x = new bool[] { true, true, false, false };
            bool[] y = new bool[] { true, true, false, false };
            Assert.True(Utils.AreEqual(x, y));

            // Not Equal
            var x1Temp = x[1];
            x[1] = x[2];
            Assert.False(Utils.AreEqual(x, y));
            x[1] = x1Temp;

            // Beginning is different
            var x0Temp = x[0];
            x[0] = x[x.Length - 1];
            Assert.False(Utils.AreEqual(x, y));
            x[0] = x0Temp;

            // End is different
            var xLengthTemp = x[x.Length - 1];
            x[x.Length - 1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[x.Length - 1] = xLengthTemp;

            // Different Array Lengths
            bool[] xOfDifferentLength = new bool[] { true, true, false };
            Assert.False(Utils.AreEqual(xOfDifferentLength, y));

            // Nulls
            Assert.False(Utils.AreEqual(null, y));
            Assert.False(Utils.AreEqual(x, null));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckAreEqualFloat()
        {
            // A sorted (increasing) array
            float[] x = Enumerable.Range(0, 10).Select(i => (float)i).ToArray();
            float[] y = Enumerable.Range(0, 10).Select(i => (float)i).ToArray();
            Assert.True(Utils.AreEqual(x, y));

            // Not Equal
            var x1Temp = x[1];
            x[1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[1] = x1Temp;

            // Beginning is different
            var x0Temp = x[0];
            x[0] = x[x.Length - 1];
            Assert.False(Utils.AreEqual(x, y));
            x[0] = x0Temp;

            // End is different
            var xLengthTemp = x[x.Length - 1];
            x[x.Length - 1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[x.Length - 1] = xLengthTemp;

            // Different Array Lengths
            float[] xOfDifferentLength = Enumerable.Range(0, 9).Select(i => (float)i).ToArray();
            Assert.False(Utils.AreEqual(xOfDifferentLength, y));

            // Nulls
            Assert.False(Utils.AreEqual(null, y));
            Assert.False(Utils.AreEqual(x, null));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckAreEqualDouble()
        {
            // A sorted (increasing) array
            double[] x = Enumerable.Range(0, 10).Select(i => (double)i).ToArray();
            double[] y = Enumerable.Range(0, 10).Select(i => (double)i).ToArray();
            Assert.True(Utils.AreEqual(x, y));

            // Not Equal
            var x1Temp = x[1];
            x[1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[1] = x1Temp;

            // Beginning is different
            var x0Temp = x[0];
            x[0] = x[x.Length - 1];
            Assert.False(Utils.AreEqual(x, y));
            x[0] = x0Temp;

            // End is different
            var xLengthTemp = x[x.Length - 1];
            x[x.Length - 1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[x.Length - 1] = xLengthTemp;

            // Different Array Lengths
            double[] xOfDifferentLength = Enumerable.Range(0, 9).Select(i => (double)i).ToArray();
            Assert.False(Utils.AreEqual(xOfDifferentLength, y));

            // Nulls
            Assert.False(Utils.AreEqual(null, y));
            Assert.False(Utils.AreEqual(x, null));
        }
    }
}
