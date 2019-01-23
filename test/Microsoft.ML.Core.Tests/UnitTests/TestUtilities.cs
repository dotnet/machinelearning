// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers.FastTree.Internal;
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
        public void CheckIsSortedInt()
        {
            // A sorted (increasing) array
            int[] x = Enumerable.Range(0, 10).ToArray();
            Assert.True(Utils.IsSorted(x));

            // A monotonically increasing array
            var x1Temp = x[1];
            var x7Temp = x[7];
            x[1] = x[0];
            x[7] = x[6];
            Assert.True(Utils.IsSorted(x));
            x[1] = x1Temp;
            x[7] = x7Temp;

            // Not sorted
            x[1] = x[6];
            Assert.False(Utils.IsSorted(x));
            x[1] = x1Temp;
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsSortedFloat()
        {
            // A sorted (increasing) array
            List<float> x = Enumerable.Range(0, 1000000).Select(i => (float)i).ToList();
            Assert.True(Utils.IsSorted(x));

            // A monotonically increasing array
            var x1Temp = x[1];
            var x7Temp = x[7];
            x[1] = x[0];
            x[7] = x[6];
            Assert.True(Utils.IsSorted(x));
            x[1] = x1Temp;
            x[7] = x7Temp;

            // Not sorted
            x[1] = x[6];
            Assert.False(Utils.IsSorted(x));
            x[1] = x1Temp;
            
            // NaN: `Array.Sort()` will put NaNs into the first position,
            // but we want to guarantee that NaNs aren't allowed in these arrays.
            var x0Temp = x[0];
            x[0] = float.NaN;
            Assert.False(Utils.IsSorted(x));
            x[0] = x0Temp;
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
            // Any len should work
            Assert.True(Utils.IsIncreasing(0, x, 5, 10));

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
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsAreEqualInt()
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
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsAreEqualBool()
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
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsAreEqualFloat()
        {
            // A sorted (increasing) array
            float[] x = Enumerable.Range(0, 10).Select(i => (float) i).ToArray();
            float[] y = Enumerable.Range(0, 10).Select(i => (float)i).ToArray();
            Assert.True(Utils.AreEqual(x, y));

            // Not Equal
            var x1Temp = x[1];
            x[1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[1] = x1Temp;
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsAreEqualDouble()
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
        }
    }
}
