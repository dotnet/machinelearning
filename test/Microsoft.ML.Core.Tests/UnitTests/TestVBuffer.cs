// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Numeric;
using Xunit;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public sealed class TestVBuffer
    {
        // How many random trials to perform per test.
        private const int _trials = 1000;
        // Controls the maximum length of the vectors to generate. Should be
        // small enough so that among _trials trials, there will be a few
        // likely to have zero counts.
        private const int _maxLen = 20;

        [Fact]
        public void TestApplyAt()
        {
            var buffer = new VBuffer<float>(10, 3, new[] { 0.5f, 1.2f, -3.8f }, new[] { 1, 5, 8 });
            VBufferUtils.ApplyAt(ref buffer, 6, (int slot, ref float value) => { value = value + 1; });
            Assert.Equal(4, buffer.GetValues().Length);
            Assert.Equal(1, buffer.GetValues()[2]);
        }

        [Fact]
        public void VBufferOpScaleBy()
        {
            var rgen = RandomUtils.Create(9);
            VBuffer<float> a = default;
            VBuffer<float> actualDst = default;
            VBuffer<float> dst = default;

            for (int trial = 0; trial < _trials; ++trial)
            {
                float c = ScaleFactor(trial, rgen);
                int len = rgen.Next(_maxLen) + 1;
                GenerateSingle(rgen, len, out a);
                FullCopy(ref a, ref dst);
                FullCopy(ref a, ref actualDst);
                VectorUtils.ScaleBy(ref actualDst, c);
                VBufferUtils.Apply(ref dst, (int i, ref float v) => v *= c);
                TestSame(ref dst, ref actualDst, 1e-5);
            }
        }

        private static float ScaleFactor(int trial, Random rgen)
        {
            switch (trial % 4)
            {
            case 0:
                return 0;
            case 1:
                return 1;
            case 2:
                return -1;
            default:
                return rgen.NextDouble().ToFloat() * 10 - 5;
            }
        }

        private static void GenerateSingle(Random rgen, int len, out VBuffer<float> a)
        {
            int count = rgen.Next(2) == 0 ? len : rgen.Next(len);
            GenerateVBuffer(rgen, len, count, out a);
        }

        private static void GenerateVBuffer(Random rgen, int len, int count, out VBuffer<float> dst)
        {
            const int excess = 4;
            Contracts.Assert(count <= len);
            int[] indices = null;
            if (count != len)
            {
                indices = new int[count + rgen.Next(excess)];
                FillRandomIndices(rgen, indices, len, count);
                for (int i = count; i < indices.Length; ++i)
                    indices[i] = rgen.Next(200) - 100;
                if (indices.Length == 0 && rgen.Next(2) == 0)
                    indices = null;
            }
            float[] values = new float[count + rgen.Next(excess)];
            for (int i = 0; i < values.Length; ++i)
                values[i] = rgen.NextDouble().ToFloat() * 10 - 5;
            if (values.Length == 0 && rgen.Next(2) == 0)
                values = null;
            dst = new VBuffer<float>(len, count, values, indices);
        }

        private static void FillRandomIndices(Random rgen, int[] indices, int len, int count)
        {
            Contracts.Assert(Utils.Size(indices) >= count);
            int max = len - count + 1;
            for (int i = 0; i < count; ++i)
                indices[i] = rgen.Next(max);
            Array.Sort(indices, 0, count);
            for (int i = 0; i < count; ++i)
                indices[i] += i;
            Contracts.Assert(Utils.IsIncreasing(0, indices, count, len));
        }

        /// <summary>
        /// Copy everything about the <paramref name="src"/>, to <paramref name="dst"/>,
        /// not just something logically equivalent but possibly with different internal
        /// structure. This will produce a totally inefficient copy right down to the
        /// length and contents of the excess indices/values.
        /// </summary>
        private static void FullCopy<T>(ref VBuffer<T> src, ref VBuffer<T> dst)
        {
            var editor = VBufferEditor.Create(ref dst, src.Length, src.GetValues().Length);
            var indices = editor.Indices;
            var values = editor.Values;
            src.GetIndices().CopyTo(indices);
            src.GetValues().CopyTo(values);
            dst = editor.Commit();
        }

        private static void TestSame(ref VBuffer<float> expected, ref VBuffer<float> actual, Double tol = 0)
        {
            TestSame<float>(ref expected, ref actual, FloatEquality((float)tol));
        }

        private static void TestSame<T>(ref VBuffer<T> expected, ref VBuffer<T> actual, Func<T, T, bool> equalityFunc)
        {
            Assert.Equal(expected.Length, actual.Length);
            Assert.Equal(expected.GetValues().Length, actual.GetValues().Length);
            Assert.Equal(expected.IsDense, actual.IsDense);
            if (!expected.IsDense)
            {
                for (int i = 0; i < expected.GetIndices().Length; ++i)
                    Assert.Equal(expected.GetIndices()[i], actual.GetIndices()[i]);
            }
            for (int i = 0; i < expected.GetValues().Length; ++i)
            {
                var result = equalityFunc(expected.GetValues()[i], actual.GetValues()[i]);
                if (!result)
                    Console.WriteLine("Friendly debug here");
                Assert.True(result, $"Value [{i}] mismatch on expected {expected.GetValues()[i]} vs. actual {actual.GetValues()[i]}");
            }
        }

        private static Func<float, float, bool> FloatEquality(float tol)
        {
            if (tol == 0)
                return (i, j) => FloatUtils.GetBits(i) == FloatUtils.GetBits(j);
            return (i, j) =>
            {
                if (FloatUtils.GetBits(i) == FloatUtils.GetBits(j) || Math.Abs(i - j) == 0)
                    return true;
                // Seemingly needlessly verbose here for the purpose of setting breakpoints.
                float comp = Math.Abs(i - j) / (Math.Abs(i) + Math.Abs(j));
                return comp <= tol;
            };
        }
    }
}
