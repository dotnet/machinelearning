// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Numeric;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public sealed class TestVBuffer : BaseTestBaseline
    {
        // How many random trials to perform per test.
        private const int _trials = 1000;
        // Controls the maximum length of the vectors to generate. Should be
        // small enough so that among _trials trials, there will be a few
        // likely to have zero counts.
        private const int _maxLen = 20;

        public TestVBuffer(ITestOutputHelper output)
            : base(output)
        {
        }

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

        [Fact]
        public void VBufferOpScaleByCopy()
        {
            var rgen = RandomUtils.Create(9);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> actualDst = default(VBuffer<float>);
            VBuffer<float> dst = default(VBuffer<float>);

            for (int trial = 0; trial < _trials; ++trial)
            {
                float c = ScaleFactor(trial, rgen);
                int len = rgen.Next(_maxLen) + 1;
                GenerateSingle(rgen, len, out a);
                FullCopy(ref a, ref dst);
                VectorUtils.ScaleBy(ref dst, c);
                VectorUtils.ScaleBy(a, ref actualDst, c);
                TestSame(ref dst, ref actualDst, 1e-5);
            }
        }

        [Fact]
        public void VBufferOpMath()
        {
            const int tol = 4;
            var rgen = RandomUtils.Create(42);

            VBuffer<float> a;
            VBuffer<float> aOrig = default(VBuffer<float>);
            for (int trial = 0; trial < _trials; ++trial)
            {
                GenerateSingle(rgen, rgen.Next(_maxLen) + 1, out a);
                a.CopyTo(ref aOrig);

                float sum = a.Items().Sum(iv => iv.Value);
                float l1 = a.Items().Sum(iv => Math.Abs(iv.Value));
                float l2Squared = a.Items().Sum(iv => iv.Value * iv.Value);
                float l2 = MathUtils.Sqrt(l2Squared);

                float infNorm = a.GetValues().Length == 0 ? 0 : a.Items().Max(iv => Math.Abs(iv.Value));

                Assert.True(CompareNumbersWithTolerance(sum, VectorUtils.Sum(in a), digitsOfPrecision: tol));
                Assert.True(CompareNumbersWithTolerance(l1, VectorUtils.L1Norm(in a), digitsOfPrecision: tol));
                Assert.True(CompareNumbersWithTolerance(l2Squared, VectorUtils.NormSquared(in a), digitsOfPrecision: tol));
                Assert.True(CompareNumbersWithTolerance(l2, VectorUtils.Norm(in a), digitsOfPrecision: tol));
                Assert.True(CompareNumbersWithTolerance(infNorm, VectorUtils.MaxNorm(in a), digitsOfPrecision: tol));

                float d = 0;
                switch (trial % 3)
                {
                case 0:
                    d = 0;
                    break;
                case 1:
                    d = 1;
                    break;
                case 2:
                    d = rgen.NextDouble().ToFloat();
                    break;
                }
                VectorUtils.ScaleBy(ref a, d);
                var editor = VBufferEditor.CreateFromBuffer(ref aOrig);
                for (int i = 0; i < editor.Values.Length; ++i)
                    editor.Values[i] *= d;
                aOrig = editor.Commit();
                TestSame(ref aOrig, ref a, 1e-5);

            }
        }

        [Fact]
        public void VBufferOpMathMul()
        {
            VBuffer<float> a;
            // Test element-wise multiplication.
            var rgen = RandomUtils.Create(0);
            GenerateSingle(rgen, rgen.Next(_maxLen) + 1, out a);
            var length = 15;
            var values = Enumerable.Range(0, length).Select(x => x + 0.1f).ToArray();
            var indicies = Enumerable.Range(0, 15).Where(x => x % 2 == 0).ToArray();
            a = new VBuffer<float>(length, values, indicies);
            float[] aDense = a.DenseValues().ToArray();
            float[] a2Values = aDense.Select(x => x + 1).ToArray();
            VBuffer<float> a2DenseVbuff = new VBuffer<float>(length, a2Values);
            var multResExpected = new float[length];
            for (var i = 0; i < length; i++)
                multResExpected[i] = aDense[i] * a2Values[i];

            var vbufMultExpected = new VBuffer<float>(length, multResExpected);
            VectorUtils.MulElementWise(in a, ref a2DenseVbuff);
            TestSame(ref vbufMultExpected, ref a2DenseVbuff, 1e-5);
        }

        [Fact]
        public void VBufferOpMathMul2()
        {
            VBuffer<float> a;
            // Test element-wise multiplication.
            var rgen = RandomUtils.Create(0);
            GenerateSingle(rgen, rgen.Next(_maxLen) + 1, out a);
            var length = 15;
            var values = Enumerable.Range(0, length).Select(x => x + 0.1f).ToArray();
            var indicies = Enumerable.Range(0, 15).Where(x => x % 2 == 0).ToArray();
            a = new VBuffer<float>(length, values, indicies);
            float[] aDense = a.DenseValues().ToArray();
            float[] a2Values = aDense.Select(x => x + 1).ToArray();
            VBuffer<float> a2DenseVbuff = new VBuffer<float>(length, a2Values);
            var multResExpected = new float[length];
            for (var i = 0; i < length; i++)
                multResExpected[i] = aDense[i] * a2Values[i];

            var vbufMultExpected = new VBuffer<float>(length, multResExpected);
            VectorUtils.MulElementWise(in a2DenseVbuff, ref a);
            TestSame(ref vbufMultExpected, ref a, 1e-5);
        }

        [Fact]
        public void SparsifyNormalize()
        {
            var a = new VBuffer<float>(5, new float[5] { 1, 2, 3, 4, 5 });
            var length = a.Length;
            float[] aDense = a.DenseValues().ToArray();
            float[] a2Values = aDense.Select(x => x + 1).ToArray();
            VBuffer<float> a2 = new VBuffer<float>(length, a2Values);
            var multResExpected = new float[2];
            multResExpected[0] = aDense[3] * a2Values[3];
            multResExpected[1] = aDense[4] * a2Values[4];

            var vbufMultExpected = new VBuffer<float>(5, 2, multResExpected, new int[2] { 3, 4 });
            var multResActual = new VBuffer<float>(length, new float[length]);
            VectorUtils.MulElementWise(in a, ref a2);
            VectorUtils.SparsifyNormalize(ref a2, 2, 2, normalize: false);
            TestSame(ref vbufMultExpected, ref a2, 1e-5);
        }

        [Fact]
        public void SparsifyNormalizeTop2()
        {
            var a = new VBuffer<float>(5, new float[5] { 1, 2, 3, 4, 5 });
            var length = a.Length;
            float[] aDense = a.DenseValues().ToArray();
            float[] a2Values = new float[5] { -1, -2, 3, 4, 5 };
            VBuffer<float> a2 = new VBuffer<float>(length, a2Values);
            var multResExpected = new float[4];
            multResExpected[0] = aDense[0] * a2Values[0];
            multResExpected[1] = aDense[1] * a2Values[1];
            multResExpected[2] = aDense[3] * a2Values[3];
            multResExpected[3] = aDense[4] * a2Values[4];

            var vbufMultExpected = new VBuffer<float>(5, 4, multResExpected, new int[4] { 0, 1, 3, 4 });
            VectorUtils.MulElementWise(in a, ref a2);
            VectorUtils.SparsifyNormalize(ref a2, 2, 2, normalize: false);
            TestSame(ref vbufMultExpected, ref a2, 1e-5);
        }

        [Fact]
        public void SparsifyNormalizeTopSparse()
        {
            for (var i = 0; i < 2; i++)
            {
                var norm = i != 0;
                var a = new VBuffer<float>(7, 6, new float[6] { 10, 20, 40, 50, 60, -70 },
                    new int[] { 0, 1, /*2 is missed*/3, 4, 5, 6 });
                var length = a.Length;
                float[] aDense = a.DenseValues().ToArray();
                float[] a2Values = aDense.Select(x => x).ToArray();
                a2Values[6] = -a2Values[6];
                VBuffer<float> a2 = new VBuffer<float>(length, a2Values);
                var multResExpected = new float[3];
                multResExpected[0] = 2500;
                multResExpected[1] = 3600;
                multResExpected[2] = -4900;

                if (norm)
                    for (var j = 0; j < multResExpected.Length; j++)
                        multResExpected[j] = multResExpected[j] / 4900;

                var vbufMultExpected = new VBuffer<float>(7, 3, multResExpected, new int[3] { 4, 5, 6 });
                VectorUtils.MulElementWise(in a, ref a2);
                VectorUtils.SparsifyNormalize(ref a2, 2, 3, normalize: norm);
                TestSame(ref vbufMultExpected, ref a2, 1e-5);
            }
        }

        [Fact]
        public void SparsifyNormalizeTopSparse2()
        {
            for (var i = 0; i < 2; i++)
            {
                var norm = i != 0;
                var a = new VBuffer<float>(7, 6, new float[6] { 100, 20, 40, 50, 60, 70 },
                    new int[] { 0, 1, /*2 is missed*/3, 4, 5, 6 });
                var b = new VBuffer<float>(7, 6, new float[6] { 100, 20, 30, 40, 50, 70 },
                    new int[] { 0, 1, 2, 3, 4, /*5 is  missed*/ 6 });
                var length = a.Length;
                var multResExpected = new float[2];
                multResExpected[0] = 10000;
                multResExpected[1] = 4900;

                if (norm)
                    for (var j = 0; j < multResExpected.Length; j++)
                        multResExpected[j] = multResExpected[j] / 10000;

                var vbufMultExpected = new VBuffer<float>(7, 2, multResExpected, new int[2] { 0, 6 });
                VectorUtils.MulElementWise(in a, ref b);
                VectorUtils.SparsifyNormalize(ref b, 2, 2, normalize: norm);
                TestSame(ref vbufMultExpected, ref b, 1e-5);
            }
        }

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

        /// <summary>
        /// A trivial inefficient implementation equivalent to <see cref="VBufferUtils.ApplyWith"/>.
        /// </summary>
        private static void NaiveApplyWith<T1, T2>(ref VBuffer<T1> a, ref VBuffer<T2> b, VBufferUtils.PairManipulator<T1, T2> manip)
        {
            Contracts.Assert(a.Length == b.Length);
            var aIndices = new HashSet<int>(a.Items().Select(iv => iv.Key));
            int[] indices = aIndices.Union(b.Items().Select(iv => iv.Key)).OrderBy(i => i).ToArray();
            T2[] values = new T2[indices.Length];
            T1 temp = default(T1);
            for (int ii = 0; ii < indices.Length; ++ii)
            {
                int i = indices[ii];
                b.GetItemOrDefault(i, ref values[ii]);
                if (aIndices.Contains(i))
                {
                    a.GetItemOrDefault(i, ref temp);
                    manip(i, temp, ref values[ii]);
                }
            }
            b = new VBuffer<T2>(a.Length, indices.Length, values, indices);
        }

        [Fact]
        public void VBufferOpApplyWith()
        {
            var rgen = RandomUtils.Create(1);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);
            VBuffer<float> aOrig = default(VBuffer<float>);
            VBuffer<float> bOrig = default(VBuffer<float>);

            VBufferUtils.PairManipulator<float, float> manip = (int ind, float av, ref float bv) => bv = 2 * bv + av - ind;

            for (int trial = 0; trial < _trials; ++trial)
            {
                int len = rgen.Next(_maxLen) + 1;
                GenLogic genLogic = 0;
                GeneratePair(rgen, len, out a, out b, out genLogic);
                FullCopy(ref a, ref aOrig);
                FullCopy(ref b, ref bOrig);
                VBufferUtils.ApplyWith(in a, ref b, manip);
                NaiveApplyWith(ref aOrig, ref bOrig, manip);
                TestSame(ref bOrig, ref b);
            }
        }

        /// <summary>
        /// A trivial inefficient implementation equivalent to <see cref="VBufferUtils.ApplyWithEitherDefined"/>.
        /// </summary>
        private static void NaiveApplyWithEither<T1, T2>(ref VBuffer<T1> a, ref VBuffer<T2> b, VBufferUtils.PairManipulator<T1, T2> manip)
        {
            int[] indices = a.Items().Select(iv => iv.Key).Union(b.Items().Select(iv => iv.Key)).OrderBy(i => i).ToArray();
            T2[] values = new T2[indices.Length];
            T1 temp = default(T1);
            for (int ii = 0; ii < indices.Length; ++ii)
            {
                int i = indices[ii];
                a.GetItemOrDefault(i, ref temp);
                b.GetItemOrDefault(i, ref values[ii]);
                manip(i, temp, ref values[ii]);
            }
            b = new VBuffer<T2>(a.Length, indices.Length, values, indices);
        }

        [Fact]
        public void VBufferOpApplyWithEither()
        {
            var rgen = RandomUtils.Create(2);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);
            VBuffer<float> aOrig = default(VBuffer<float>);
            VBuffer<float> bOrig = default(VBuffer<float>);
            VBuffer<float> dst = default(VBuffer<float>);

            VBufferUtils.PairManipulator<float, float> manip = (int ind, float av, ref float bv) => bv = 2 * bv + av - ind;

            for (int trial = 0; trial < _trials; ++trial)
            {
                int len = rgen.Next(_maxLen) + 1;
                GenLogic genLogic = 0;
                GeneratePair(rgen, len, out a, out b, out genLogic);
                FullCopy(ref a, ref aOrig);
                FullCopy(ref b, ref bOrig);
                FullCopy(ref b, ref dst);
                VBufferUtils.ApplyWithEitherDefined(in a, ref b, manip);
                NaiveApplyWithEither(ref aOrig, ref dst, manip);
                TestSame(ref dst, ref b);
            }
        }

        /// <summary>
        /// A trivial inefficient implementation equivalent to <see cref="VBufferUtils.ForEachEitherDefined"/>
        /// if <paramref name="union"/> is true, or if false <see cref="VBufferUtils.ForEachBothDefined"/>.
        /// </summary>
        private static void NaiveForEach<T1, T2>(ref VBuffer<T1> a, ref VBuffer<T2> b, Action<int, T1, T2> vis, bool union)
        {
            var aIndices = a.Items().Select(iv => iv.Key);
            var bIndices = b.Items().Select(iv => iv.Key);
            var indices = union ? aIndices.Union(bIndices) : aIndices.Intersect(bIndices);
            indices = indices.Distinct().OrderBy(x => x);
            T1 aValue = default(T1);
            T2 bValue = default(T2);
            foreach (var index in indices)
            {
                a.GetItemOrDefault(index, ref aValue);
                b.GetItemOrDefault(index, ref bValue);
                vis(index, aValue, bValue);
            }
        }

        private void VBufferOpForEachHelper(bool union)
        {
            var rgen = RandomUtils.Create(union ? 3 : 4);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);

            float accum = 0;
            Action<int, float, float> vis = (int ind, float av, float bv) => accum += 2 * bv + av - ind;

            for (int trial = 0; trial < _trials; ++trial)
            {
                int len = rgen.Next(_maxLen) + 1;
                GenLogic genLogic = 0;
                GeneratePair(rgen, len, out a, out b, out genLogic);
                if (union)
                    VBufferUtils.ForEachEitherDefined(in a, in b, vis);
                else
                    VBufferUtils.ForEachBothDefined(in a, in b, vis);
                var actualAccum = accum;
                accum = 0;
                NaiveForEach(ref a, ref b, vis, union);
                Assert.Equal(accum, actualAccum);
                accum = 0;
            }
        }

        [Fact]
        public void VBufferOpForEachEither()
        {
            VBufferOpForEachHelper(union: true);
        }

        [Fact]
        public void VBufferOpForEachBoth()
        {
            VBufferOpForEachHelper(union: false);
        }

        private static void NaiveApplyInto<T, TDst>(ref VBuffer<T> a, ref VBuffer<TDst> dst, Func<int, T, TDst> func)
        {
            List<int> indices = new List<int>(a.GetIndices().Length);
            TDst[] values = new TDst[a.GetValues().Length];
            foreach (var iv in a.Items())
            {
                values[indices.Count] = func(iv.Key, iv.Value);
                indices.Add(iv.Key);
            }
            dst = new VBuffer<TDst>(a.Length, indices.Count, values, indices.ToArray());
        }

        [Fact]
        public void VBufferOpApplyIntoSingle()
        {
            var rgen = RandomUtils.Create(5);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> actualDst = default(VBuffer<float>);
            VBuffer<float> dst = default(VBuffer<float>);

            Func<int, float, float> func = (int ind, float av) => av - ind;

            for (int trial = 0; trial < _trials; ++trial)
            {
                int len = rgen.Next(_maxLen) + 1;
                GenerateSingle(rgen, len, out a);
                VBufferUtils.ApplyIntoEitherDefined(in a, ref actualDst, func);
                NaiveApplyInto(ref a, ref dst, func);
                TestSame(ref dst, ref actualDst);
            }
        }

        private static void NaiveApplyInto<T1, T2, TDst>(ref VBuffer<T1> a, ref VBuffer<T2> b, ref VBuffer<TDst> dst, Func<int, T1, T2, TDst> func)
        {
            int[] indices = a.Items().Select(iv => iv.Key)
                .Union(b.Items().Select(iv => iv.Key)).Distinct().OrderBy(x => x).ToArray();
            TDst[] values = new TDst[indices.Length];
            T1 aValue = default(T1);
            T2 bValue = default(T2);
            for (int i = 0; i < indices.Length; ++i)
            {
                a.GetItemOrDefault(indices[i], ref aValue);
                b.GetItemOrDefault(indices[i], ref bValue);
                values[i] = func(indices[i], aValue, bValue);
            }
            dst = new VBuffer<TDst>(a.Length, indices.Length, values, indices);
        }

        [Fact]
        public void VBufferOpApplyIntoPair()
        {
            var rgen = RandomUtils.Create(6);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);
            VBuffer<float> actualDst = default(VBuffer<float>);
            VBuffer<float> dst = default(VBuffer<float>);

            Func<int, float, float, float> func = (int ind, float av, float bv) => 2 * bv + av - ind;

            for (int trial = 0; trial < _trials; ++trial)
            {
                int len = rgen.Next(_maxLen) + 1;
                GenLogic genLogic = 0;
                GeneratePair(rgen, len, out a, out b, out genLogic);
                VBufferUtils.ApplyInto(in a, in b, ref actualDst, func);
                NaiveApplyInto(ref a, ref b, ref dst, func);
                TestSame(ref dst, ref actualDst);
            }
        }

        /// <summary>
        /// Naive version of <see cref="VectorUtils.AddMultWithOffset(in VBuffer{float}, float, ref VBuffer{float}, int)"/>,
        /// which can be used to generalize all the other add-mult functions.
        /// </summary>
        private static void NaiveAddMult(ref VBuffer<float> a, float c, ref VBuffer<float> b, int offset)
        {
            Contracts.Assert(0 <= offset && a.Length <= b.Length - offset);
            if (a.GetValues().Length == 0 || c == 0)
                return;
            VBuffer<float> aa = default(VBuffer<float>);
            if (offset == 0 && a.Length == b.Length)
                aa = a;
            else
            {
                a.CopyTo(ref aa);
                var editor = VBufferEditor.Create(ref aa, b.Length, aa.GetValues().Length, requireIndicesOnDense: true);
                var indices = editor.Indices;
                if (aa.IsDense)
                {
                    for (int i = 0; i < aa.Length; i++)
                        indices[i] = i;
                }
                for (int i = 0; i < editor.Indices.Length; ++i)
                    indices[i] += offset;
                aa = editor.Commit();
            }
            VBufferUtils.PairManipulator<float, float> manip =
                (int ind, float av, ref float bv) => bv = bv + c * av;
            NaiveApplyWith(ref aa, ref b, manip);
        }

        [Fact]
        public void VBufferOpAddMult()
        {
            var rgen = RandomUtils.Create(7);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);
            VBuffer<float> actualDst = default(VBuffer<float>);
            VBuffer<float> dst = default(VBuffer<float>);

            for (int trial = 0; trial < _trials; ++trial)
            {
                float c = ScaleFactor(trial, rgen);
                int len = rgen.Next(_maxLen) + 1;
                GenLogic genLogic;
                GeneratePair(rgen, len, out a, out b, out genLogic);
                // Keep b around as an original for debugging purposes.
                FullCopy(ref b, ref dst);
                FullCopy(ref b, ref actualDst);
                NaiveAddMult(ref a, c, ref dst, 0);
                VectorUtils.AddMult(in a, c, ref actualDst);
                TestSame(ref dst, ref actualDst, 1e-4);
                if (c == 1)
                {
                    // While we're at it, test Add.
                    FullCopy(ref b, ref actualDst);
                    VectorUtils.Add(in a, ref actualDst);
                    TestSame(ref dst, ref actualDst);
                }
            }
        }

        [Fact]
        public void VBufferOpAddMultCopy()
        {
            var rgen = RandomUtils.Create(7);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);
            VBuffer<float> actualDst = default(VBuffer<float>);
            VBuffer<float> dst = default(VBuffer<float>);

            for (int trial = 0; trial < _trials; ++trial)
            {
                float c = ScaleFactor(trial, rgen);
                int len = rgen.Next(_maxLen) + 1;
                GenLogic genLogic;
                GeneratePair(rgen, len, out a, out b, out genLogic);
                FullCopy(ref b, ref dst);
                NaiveAddMult(ref a, c, ref dst, 0);
                VectorUtils.AddMult(in a, c, ref b, ref actualDst);
                TestEquivalent(ref dst, ref actualDst, 1e-4);
            }
        }

        [Fact]
        public void VBufferOpAddMultWithOffset()
        {
            var rgen = RandomUtils.Create(8);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);
            VBuffer<float> actualDst = default(VBuffer<float>);
            VBuffer<float> dst = default(VBuffer<float>);

            for (int trial = 0; trial < _trials; ++trial)
            {
                float c = ScaleFactor(trial, rgen);
                int len = rgen.Next(_maxLen) + 1;
                GenerateSingle(rgen, len, out b);
                GenerateSingle(rgen, rgen.Next(len) + 1, out a);
                int offset = rgen.Next(b.Length - a.Length + 1);
                // Keep b around as an original for debugging purposes.
                FullCopy(ref b, ref dst);
                FullCopy(ref b, ref actualDst);
                NaiveAddMult(ref a, c, ref dst, offset);
                VectorUtils.AddMultWithOffset(in a, c, ref actualDst, offset);
                TestSame(ref dst, ref actualDst, 1e-4);
            }
        }

        [Fact]
        public void VBufferOpScaleInto()
        {
            var rgen = RandomUtils.Create(10);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);
            VBuffer<float> actualDst = default(VBuffer<float>);
            VBuffer<float> dst = default(VBuffer<float>);

            for (int trial = 0; trial < _trials; ++trial)
            {
                float c = ScaleFactor(trial, rgen);
                int len = rgen.Next(_maxLen) + 1;
                GenerateSingle(rgen, len, out a);
                GenerateSingle(rgen, rgen.Next(_maxLen) + 1, out b);
                FullCopy(ref b, ref dst);
                FullCopy(ref b, ref actualDst);
                // Inefficient ScaleInto, including the c==0 deviation from ApplyInto.
                if (c == 0 && !a.IsDense)
                    dst = VBufferEditor.Create(ref dst, a.Length, 0).Commit();
                else
                    NaiveApplyInto(ref a, ref dst, (i, av) => c * av);
                VectorUtils.ScaleInto(in a, c, ref actualDst);
                TestSame(ref dst, ref actualDst, 1e-5);
            }
        }

        [Fact]
        public void VBufferOpAddMultInto()
        {
            var rgen = RandomUtils.Create(11);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);
            VBuffer<float> actualDst = default(VBuffer<float>);
            VBuffer<float> dst = default(VBuffer<float>);

            for (int trial = 0; trial < _trials; ++trial)
            {
                float c = ScaleFactor(trial, rgen);
                int len = rgen.Next(_maxLen) + 1;
                GenLogic genLogic;
                GeneratePair(rgen, len, out a, out b, out genLogic);
                FullCopy(ref a, ref dst);
                NaiveAddMult(ref b, c, ref dst, 0);
                VectorUtils.AddMultInto(in a, c, in b, ref actualDst);
                TestSame(ref dst, ref actualDst);
            }
        }

        [Fact]
        public void VBufferOpApplySlot()
        {
            var rgen = RandomUtils.Create(12);
            VBuffer<float> a = default(VBuffer<float>);
            float[] expected = new float[_maxLen];
            float[] actual = new float[_maxLen];

            VBufferUtils.SlotValueManipulator<float> manip = (int i, ref float value) => value += i;

            for (int trial = 0; trial < _trials; ++trial)
            {
                int len = rgen.Next(_maxLen) + 1;
                GenerateSingle(rgen, len, out a);
                a.CopyTo(expected);
                int slot = rgen.Next(len);
                manip(slot, ref expected[slot]);
                VBufferUtils.ApplyAt(ref a, slot, manip);
                Assert.Equal(len, a.Length);
                a.CopyTo(actual);
                for (int i = 0; i < len; ++i)
                    Assert.Equal(expected[i], actual[i]);
            }
        }

        [Fact]
        public void VBufferOpCopyRange()
        {
            var rgen = RandomUtils.Create(13);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> dst = default(VBuffer<float>);

            VBufferUtils.SlotValueManipulator<float> manip = (int i, ref float value) => value += i;

            for (int trial = 0; trial < _trials; ++trial)
            {
                int len = rgen.Next(_maxLen) + 1;
                GenerateSingle(rgen, len, out a);
                int copyMin = rgen.Next(len + 1);
                int copyLen = rgen.Next(len - copyMin + 1);
                a.CopyTo(ref dst, copyMin, copyLen);
                Assert.Equal(copyLen, dst.Length);
                float value = 0;
                foreach (var iv in dst.Items(all: true))
                {
                    a.GetItemOrDefault(iv.Key + copyMin, ref value);
                    Assert.Equal(value, iv.Value);
                }
            }
        }

        [Fact]
        public void VBufferOpDensify()
        {
            var rgen = RandomUtils.Create(14);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);

            VBufferUtils.SlotValueManipulator<float> manip = (int i, ref float value) => value += i;

            for (int trial = 0; trial < _trials; ++trial)
            {
                int len = rgen.Next(_maxLen) + 1;
                GenerateSingle(rgen, len, out a);
                FullCopy(ref a, ref b);
                VBufferUtils.Densify(ref b);

                Assert.Equal(len, b.Length);
                Assert.True(b.IsDense, "Result was not dense, as expected");
                int count = 0;
                float value = 0;
                foreach (var iv in b.Items(all: false))
                {
                    a.GetItemOrDefault(iv.Key, ref value);
                    Assert.Equal(value, iv.Value);
                    count++;
                }
                Assert.Equal(len, count);
            }
        }

        [Fact]
        public void VBufferOpDensifyFirst()
        {
            var rgen = RandomUtils.Create(15);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);

            VBufferUtils.SlotValueManipulator<float> manip = (int i, ref float value) => value += i;

            for (int trial = 0; trial < _trials; ++trial)
            {
                int len = rgen.Next(_maxLen) + 1;
                GenerateSingle(rgen, len, out a);
                FullCopy(ref a, ref b);
                int dense = rgen.Next(len + 1);
                VBufferUtils.DensifyFirst(ref b, dense);

                Assert.Equal(len, b.Length);
                Assert.True(b.IsDense || !a.IsDense, "Density of a did not imply density of b");
                float value = 0;

                HashSet<int> aIndices = new HashSet<int>(a.Items().Select(iv => iv.Key));
                HashSet<int> bIndices = new HashSet<int>(b.Items().Select(iv => iv.Key));

                foreach (var iv in b.Items(all: false))
                {
                    a.GetItemOrDefault(iv.Key, ref value);
                    Assert.Equal(value, iv.Value);
                }
                for (int i = 0; i < dense; ++i)
                {
                    Assert.True(bIndices.Remove(i), $"Slot {i} not explicitly represented");
                    aIndices.Remove(i);
                }
                // Now we consider the set of indices beyond those we explicitly densified.
                Assert.True(aIndices.SetEquals(bIndices), "Indices disagreed on explicit representation");
            }
        }

        [Fact]
        public void VBufferOpPairwiseMath()
        {
            var rgen = RandomUtils.Create(16);
            VBuffer<float> a = default(VBuffer<float>);
            VBuffer<float> b = default(VBuffer<float>);

            for (int trial = 0; trial < _trials; ++trial)
            {
                GenLogic genLogic;
                int len = rgen.Next(_maxLen) + 1;
                GeneratePair(rgen, len, out a, out b, out genLogic);

                var l1Dist = a.Items(all: true).Zip(b.Items(all: true), (av, bv) => Math.Abs(av.Value - bv.Value)).Sum();
                var l2Dist2 = a.Items(all: true).Zip(b.Items(all: true), (av, bv) => MathUtils.Pow(av.Value - bv.Value, 2)).Sum();
                var l2Dist = MathUtils.Sqrt(l2Dist2);
                var dot = a.Items(all: true).Zip(b.Items(all: true), (av, bv) => av.Value * bv.Value).Sum();

                const int tol = 4;
                Assert.True(CompareNumbersWithTolerance(l1Dist, VectorUtils.L1Distance(in a, in b), digitsOfPrecision: tol));
                Assert.True(CompareNumbersWithTolerance(l2Dist2, VectorUtils.L2DistSquared(in a, in b), digitsOfPrecision: tol));
                Assert.True(CompareNumbersWithTolerance(l2Dist, VectorUtils.Distance(in a, in b), digitsOfPrecision: tol));
                Assert.True(CompareNumbersWithTolerance(dot, VectorUtils.DotProduct(in a, in b), digitsOfPrecision: tol));
            }
        }

        [Fact]
        public void VBufferDropSlots()
        {
            var rgen = RandomUtils.Create(16);
            var a = default(VBuffer<float>);
            int dropSlotSparseOpCount = 0;
            int dropSlotDenseOpCount = 0;
            var minSlots = new List<int>();
            var maxSlots = new List<int>();
            bool dropAll = true;
            while (dropSlotSparseOpCount < _trials || dropSlotDenseOpCount < _trials)
            {
                int len = rgen.Next(_maxLen + 300);
                GenerateSingle(rgen, len, out a);
                minSlots.Clear();
                maxSlots.Clear();
                int min;
                int max = -2;
                while (max + 2 < len)
                {
                    if (dropAll)
                    {
                        min = 0;
                        max = len;
                    }
                    else
                    {
                        min = rgen.Next(Math.Min(max + 10, len + 100) - max - 2) + max + 2;
                        max = rgen.Next(Math.Min(min + 10, len + 100) - min) + min;
                    }

                    minSlots.Add(min);
                    maxSlots.Add(max);

                    if (maxSlots.Count > 20 || len < 200 && rgen.Next(20) < 2 || dropAll)
                    {
                        dropAll = false;
                        var slotDropper = new SlotDropper(len, minSlots.ToArray(), maxSlots.ToArray());
                        int slotsDropped = 0;
                        int nonExistentSlotsDropped = 0;

                        for (int i = 0; i < minSlots.Count; i++)
                        {
                            if (maxSlots[i] < len)
                                slotsDropped += maxSlots[i] - minSlots[i] + 1;
                            else if (minSlots[i] >= len)
                                nonExistentSlotsDropped += maxSlots[i] - minSlots[i] + 1;
                            else
                            {
                                slotsDropped += len - minSlots[i];
                                nonExistentSlotsDropped += maxSlots[i] - len + 1;
                            }
                        }

                        Assert.True(slotsDropped + nonExistentSlotsDropped > 0);

                        int expectedLength = Math.Max(1, a.Length - slotsDropped);

                        Assert.True(expectedLength >= 1);

                        var expectedIndices = new List<int>();
                        var expectedValues = new List<float>();
                        int index = 0;
                        int dropSlotMinIndex = 0;
                        int slotsDroppedSoFar = 0;
                        while (index < a.GetValues().Length)
                        {
                            int logicalIndex = a.IsDense ? index : a.GetIndices()[index];
                            if (dropSlotMinIndex >= minSlots.Count || logicalIndex < minSlots[dropSlotMinIndex])
                            {
                                Assert.True(logicalIndex - slotsDroppedSoFar >= 0);
                                expectedIndices.Add(logicalIndex - slotsDroppedSoFar);
                                expectedValues.Add(a.GetValues()[index]);
                                index++;
                            }
                            else if (logicalIndex <= maxSlots[dropSlotMinIndex])
                                index++;
                            else
                            {
                                slotsDroppedSoFar += maxSlots[dropSlotMinIndex] - minSlots[dropSlotMinIndex] + 1;
                                dropSlotMinIndex++;
                            }
                        }

                        Assert.Equal(expectedIndices.Count, expectedValues.Count);
                        Assert.True(expectedIndices.Count <= a.GetValues().Length);

                        if (a.IsDense)
                            dropSlotDenseOpCount++;
                        else
                            dropSlotSparseOpCount++;

                        var expectedVector = new VBuffer<float>(Math.Max(1, expectedLength),
                            expectedIndices.Count, expectedValues.ToArray(), a.IsDense ? null : expectedIndices.ToArray());

                        var dst = rgen.Next(2) == 0 ? a : default(VBuffer<float>);
                        slotDropper.DropSlots(ref a, ref dst);
                        TestSame(ref expectedVector, ref dst);
                        minSlots.Clear();
                        maxSlots.Clear();
                        max = -1;
                        len = dst.Length;
                        Utils.Swap(ref a, ref dst);
                    }
                }
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
            Assert.True(Utils.IsIncreasing(0, indices, count, len));
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

        /// <summary>
        /// Returned out of <see cref="GeneratePair"/> so that debugging runs can see what
        /// specific subcase of pair relationship was being explored.
        /// </summary>
        private enum GenLogic : byte
        {
            BothDense,
            ASparseBDense,
            ADenseBSparse,
            BothSparseASameB,
            BothSparseASubsetB,
            BothSparseBSubsetA,
            BothSparseAUnrelatedB,
            BothSparseADisjointB,
        }

        /// <summary>
        /// Generates a pair of vectors, where the pairs are generated in such a way that we
        /// have a good chance (across many trials) to exercise all of the special casing logic
        /// we see in <see cref="VectorUtils"/> and <see cref="VBufferUtils"/>, e.g., various
        /// density/sparsity settings, different ways of the indices overlapping each other, etc.
        /// </summary>
        /// <param name="rgen">The random number generator</param>
        /// <param name="len">The length of the vectors to generate</param>
        /// <param name="a">The first of the pair</param>
        /// <param name="b">The second of the pair</param>
        /// <param name="subcase">An enum describing the specific case of logic that generated
        /// the two generated vectors, which goes some way to describing the relationship between
        /// the two</param>
        private static void GeneratePair(Random rgen, int len, out VBuffer<float> a, out VBuffer<float> b, out GenLogic subcase)
        {
            // 0. Both dense.
            // 1. a sparse, b dense.
            // 2. a dense, b sparse.
            // Both sparse:
            // 3. a indices the same as b's.
            // 4. a indices a subset of b's.
            // 5. b indices a subset of a's.
            // 6. a and b may overlap. (But I don't prevent distinct.)
            // 7. a and b totally disjoint.
            const int cases = 8;
            Contracts.Assert(cases == Enum.GetValues(typeof(GenLogic)).Length);
            subcase = (GenLogic)rgen.Next(cases);
            VBufferEditor<float> bEditor;
            switch (subcase)
            {
            case GenLogic.BothDense:
                // Both dense.
                GenerateVBuffer(rgen, len, len, out a);
                GenerateVBuffer(rgen, len, len, out b);
                break;
            case GenLogic.ASparseBDense:
            case GenLogic.ADenseBSparse:
                GenerateVBuffer(rgen, len, len, out a);
                GenerateVBuffer(rgen, len, rgen.Next(len), out b);
                if (subcase == GenLogic.ASparseBDense)
                    Utils.Swap(ref a, ref b);
                break;
            case GenLogic.BothSparseASameB:
                GenerateVBuffer(rgen, len, rgen.Next(len), out a);
                GenerateVBuffer(rgen, len, a.GetValues().Length, out b);
                bEditor = VBufferEditor.CreateFromBuffer(ref b);
                for (int i = 0; i < a.GetIndices().Length; ++i)
                    bEditor.Indices[i] = a.GetIndices()[i];
                b = bEditor.Commit();
                break;
            case GenLogic.BothSparseASubsetB:
            case GenLogic.BothSparseBSubsetA:
                GenerateVBuffer(rgen, len, rgen.Next(len), out a);
                GenerateVBuffer(rgen, a.GetValues().Length, rgen.Next(a.GetValues().Length), out b);
                bEditor = VBufferEditor.Create(ref b, len, b.GetValues().Length);
                for (int i = 0; i < bEditor.Values.Length; ++i)
                    bEditor.Indices[i] = a.GetIndices()[bEditor.Indices[i]];
                b = bEditor.Commit();
                if (subcase == GenLogic.BothSparseASubsetB)
                    Utils.Swap(ref a, ref b);
                break;
            case GenLogic.BothSparseAUnrelatedB:
                GenerateVBuffer(rgen, len, rgen.Next(len), out a);
                GenerateVBuffer(rgen, len, rgen.Next(len), out b);
                break;
            case GenLogic.BothSparseADisjointB:
                GenerateVBuffer(rgen, len, rgen.Next(len), out a);
                int boundary = rgen.Next(a.GetValues().Length + 1);
                GenerateVBuffer(rgen, len, a.GetValues().Length - boundary, out b);
                if (a.GetValues().Length != 0 && b.GetValues().Length != 0 && a.GetValues().Length != b.GetValues().Length)
                {
                    var aEditor = VBufferEditor.CreateFromBuffer(ref a);
                    bEditor = VBufferEditor.CreateFromBuffer(ref b);
                    Utils.Shuffle(rgen, aEditor.Indices);
                    aEditor.Indices.Slice(boundary).CopyTo(bEditor.Indices);

                    GenericSpanSortHelper<int>.Sort(aEditor.Indices, 0, boundary);
                    GenericSpanSortHelper<int>.Sort(bEditor.Indices, 0, bEditor.Indices.Length);
                    a = aEditor.CommitTruncated(boundary);
                    b = bEditor.Commit();
                }
                if (rgen.Next(2) == 0)
                    Utils.Swap(ref a, ref b);
                break;
            default:
                throw Contracts.Except("Whoops, did you miss a case?");
            }
            Contracts.Assert(a.Length == len);
            Contracts.Assert(b.Length == len);
        }

        private static void TestEquivalent(ref VBuffer<float> expected, ref VBuffer<float> actual, Double tol = 0)
        {
            TestEquivalent<float>(ref expected, ref actual, FloatEquality((float)tol));
        }

        private static void TestEquivalent<T>(ref VBuffer<T> expected, ref VBuffer<T> actual, Func<T, T, bool> equalityFunc)
        {
            Contracts.AssertValue(equalityFunc);
            Assert.Equal(expected.Length, actual.Length);

            int length = expected.Length;
            if (length == 0)
                return;

            Contracts.Assert(length > 0);
            if (expected.IsDense && actual.IsDense)
            {
                for (int i = 0; i < length; ++i)
                    Assert.True(equalityFunc(expected.GetValues()[i], actual.GetValues()[i]));
            }
            else if (expected.IsDense)
            {
                // expected is dense, actual is sparse.
                int jj = 0;
                int j = actual.GetValues().Length == 0 ? length : actual.GetIndices()[jj];
                for (int i = 0; i < length; ++i)
                {
                    if (i == j)
                    {
                        Assert.True(equalityFunc(expected.GetValues()[i], actual.GetValues()[jj]));
                        j = ++jj == actual.GetValues().Length ? length : actual.GetIndices()[jj];
                    }
                    else
                        Assert.True(equalityFunc(expected.GetValues()[i], default(T)));
                }
            }
            else if (actual.IsDense)
            {
                // expected is sparse, actual is dense.
                int ii = 0;
                int i = expected.GetValues().Length == 0 ? length : expected.GetIndices()[ii];
                for (int j = 0; j < length; ++j)
                {
                    if (j == i)
                    {
                        Assert.True(equalityFunc(expected.GetValues()[ii], actual.GetValues()[j]));
                        i = ++ii == expected.GetValues().Length ? length : expected.GetIndices()[ii];
                    }
                    else
                        Assert.True(equalityFunc(actual.GetValues()[j], default(T)));
                }
            }
            else
            {
                // Both expected and actual are sparse.
                int ii = 0;
                int jj = 0;
                int i = expected.GetValues().Length == 0 ? length : expected.GetIndices()[ii];
                int j = actual.GetValues().Length == 0 ? length : actual.GetIndices()[jj];

                while (i < length || j < length)
                {
                    if (i == j)
                    {
                        // Common slot for expected and actual.
                        Assert.True(equalityFunc(expected.GetValues()[ii], actual.GetValues()[jj]));
                        i = ++ii == expected.GetValues().Length ? length : expected.GetIndices()[ii];
                        j = ++jj == actual.GetValues().Length ? length : actual.GetIndices()[jj];
                    }
                    else if (i < j)
                    {
                        Assert.True(equalityFunc(expected.GetValues()[ii], default(T)));
                        i = ++ii == expected.GetValues().Length ? length : expected.GetIndices()[ii];
                    }
                    else
                    {
                        // i > j
                        Assert.True(equalityFunc(actual.GetValues()[ii], default(T)));
                        j = ++jj == actual.GetValues().Length ? length : actual.GetIndices()[jj];
                    }
                }
            }
        }
    }
}