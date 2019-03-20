// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    public sealed class TestSparseDataView : TestDataViewBase
    {
        private const string Cat = "DataView";

        public TestSparseDataView(ITestOutputHelper obj) : base(obj)
        {
        }

        private class DenseExample<T>
        {
            [VectorType(2)]
            public T[] X;
        }

        private class SparseExample<T>
        {
            [VectorType(5)]
            public VBuffer<T> X;
        }

        [Fact]
        [TestCategory(Cat)]
        public void SparseDataView()
        {
            GenericSparseDataView(new[] { 1f, 2f, 3f }, new[] { 1f, 10f, 100f });
            GenericSparseDataView(new int[] { 1, 2, 3 }, new int[] { 1, 10, 100 });
            GenericSparseDataView(new bool[] { true, true, true }, new bool[] { false, false, false });
            GenericSparseDataView(new double[] { 1, 2, 3 }, new double[] { 1, 10, 100 });
            GenericSparseDataView(new ReadOnlyMemory<char>[] { "a".AsMemory(), "b".AsMemory(), "c".AsMemory() },
                                  new ReadOnlyMemory<char>[] { "aa".AsMemory(), "bb".AsMemory(), "cc".AsMemory() });
        }

        private void GenericSparseDataView<T>(T[] v1, T[] v2)
        {
            var inputs = new[] {
                new SparseExample<T>() { X = new VBuffer<T> (5, 3, v1, new int[] { 0, 2, 4 }) },
                new SparseExample<T>() { X = new VBuffer<T> (5, 3, v2, new int[] { 0, 1, 3 }) }
            };
            var env = new MLContext();
            var data = env.Data.LoadFromEnumerable(inputs);
            var value = new VBuffer<T>();
            int n = 0;
            using (var cur = data.GetRowCursorForAllColumns())
            {
                var getter = cur.GetGetter<VBuffer<T>>(cur.Schema[0]);
                while (cur.MoveNext())
                {
                    getter(ref value);
                    Assert.True(value.GetValues().Length == 3);
                    ++n;
                }
            }
            Assert.True(n == 2);
            var iter = env.Data.CreateEnumerable<SparseExample<T>>(data, false).GetEnumerator();
            n = 0;
            while (iter.MoveNext())
                ++n;
            Assert.True(n == 2);
        }

        [Fact]
        [TestCategory(Cat)]
        public void DenseDataView()
        {
            GenericDenseDataView(new[] { 1f, 2f, 3f }, new[] { 1f, 10f, 100f });
            GenericDenseDataView(new int[] { 1, 2, 3 }, new int[] { 1, 10, 100 });
            GenericDenseDataView(new bool[] { true, true, true }, new bool[] { false, false, false });
            GenericDenseDataView(new double[] { 1, 2, 3 }, new double[] { 1, 10, 100 });
            GenericDenseDataView(new ReadOnlyMemory<char>[] { "a".AsMemory(), "b".AsMemory(), "c".AsMemory() },
                                 new ReadOnlyMemory<char>[] { "aa".AsMemory(), "bb".AsMemory(), "cc".AsMemory() });
        }

        private void GenericDenseDataView<T>(T[] v1, T[] v2)
        {
            var inputs = new[] {
                new DenseExample<T>() { X = v1 },
                new DenseExample<T>() { X = v2 }
            };
            var env = new MLContext();
            var data = env.Data.LoadFromEnumerable(inputs);
            var value = new VBuffer<T>();
            int n = 0;
            using (var cur = data.GetRowCursorForAllColumns())
            {
                var getter = cur.GetGetter<VBuffer<T>>(cur.Schema[0]);
                while (cur.MoveNext())
                {
                    getter(ref value);
                    Assert.True(value.GetValues().Length == 3);
                    ++n;
                }
            }
            Assert.True(n == 2);
            var iter = env.Data.CreateEnumerable<DenseExample<T>>(data, false).GetEnumerator();
            n = 0;
            while (iter.MoveNext())
                ++n;
            Assert.True(n == 2);
        }
    }
}
