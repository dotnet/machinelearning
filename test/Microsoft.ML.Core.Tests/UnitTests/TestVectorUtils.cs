// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Numeric;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Runtime.RunTests
{
    public class TestVectorUtils
    {
        /// <summary>
        /// Tests SparsifyNormalize works correctly.
        /// </summary>
        [Theory]
        [InlineData(1, true, new[] { 0.8f, 0.9f, 1f }, new[] { 7, 8, 9 })]
        [InlineData(1, false, new[] { 8f, 9f, 10f }, new[] { 7, 8, 9 })]
        [InlineData(-4, true, new[] { -0.8f, -0.6f, -0.4f, 0.6f, 0.8f, 1f }, new[] { 0, 1, 2, 7, 8, 9 })]
        [InlineData(-4, false, new[] { -4f, -3f, -2f, 3f, 4f, 5f }, new[] { 0, 1, 2, 7, 8, 9 })]
        [InlineData(-10, true, new[] { -1f, -0.9f, -0.8f }, new[] { 0, 1, 2 })]
        [InlineData(-10, false, new[] { -10f, -9f, -8f }, new[] { 0, 1, 2 })]
        public void TestSparsifyNormalize(int startRange, bool normalize, float[] expectedValues, int[] expectedIndices)
        {
            float[] values = Enumerable.Range(startRange, 10).Select(i => (float)i).ToArray();
            var a = new VBuffer<float>(10, values);

            VectorUtils.SparsifyNormalize(ref a, 3, 3, normalize);

            Assert.False(a.IsDense);
            Assert.Equal(10, a.Length);
            Assert.Equal(expectedIndices, a.GetIndices().ToArray());

            var actualValues = a.GetValues().ToArray();
            Assert.Equal(expectedValues.Length, actualValues.Length);
            for (int i = 0; i < expectedValues.Length; i++)
                Assert.Equal(expectedValues[i], actualValues[i], precision: 6);
        }

        /// <summary>
        /// Tests SparsifyNormalize works when asked for all values.
        /// </summary>
        [Theory]
        [InlineData(10, 0)]
        [InlineData(10, 10)]
        [InlineData(20, 20)]
        public void TestSparsifyNormalizeReturnsDense(int top, int bottom)
        {
            float[] values = Enumerable.Range(1, 10).Select(i => (float)i).ToArray();
            var a = new VBuffer<float>(10, values);

            VectorUtils.SparsifyNormalize(ref a, top, bottom, false);

            Assert.True(a.IsDense);
            Assert.Equal(10, a.Length);
            Assert.True(a.GetIndices().IsEmpty);

            Assert.Equal(values, a.GetValues().ToArray());
        }
    }
}
