// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
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
            GenericSparseDataView(new DvBool[] { true, true, true }, new DvBool[] { false, false, false });
            GenericSparseDataView(new double[] { 1, 2, 3 }, new double[] { 1, 10, 100 });
            GenericSparseDataView(new DvText[] { new DvText("a"), new DvText("b"), new DvText("c") },
                                  new DvText[] { new DvText("aa"), new DvText("bb"), new DvText("cc") });
        }

        private void GenericSparseDataView<T>(T[] v1, T[] v2)
        {
            var inputs = new[] {
                new SparseExample<T>() { X = new VBuffer<T> (5, 3, v1, new int[] { 0, 2, 4 }) },
                new SparseExample<T>() { X = new VBuffer<T> (5, 3, v2, new int[] { 0, 1, 3 }) }
            };
            using (var host = new TlcEnvironment())
            {
                var data = host.CreateStreamingDataView(inputs);
                var value = new VBuffer<T>();
                int n = 0;
                using (var cur = data.GetRowCursor(i => true))
                {
                    var getter = cur.GetGetter<VBuffer<T>>(0);
                    while (cur.MoveNext())
                    {
                        getter(ref value);
                        Assert.True(value.Count == 3);
                        ++n;
                    }
                }
                Assert.True(n == 2);
                var iter = data.AsEnumerable<SparseExample<T>>(host, false).GetEnumerator();
                n = 0;
                while (iter.MoveNext())
                    ++n;
                Assert.True(n == 2);
            }
        }

        [Fact]
        [TestCategory(Cat)]
        public void DenseDataView()
        {
            GenericDenseDataView(new[] { 1f, 2f, 3f }, new[] { 1f, 10f, 100f });
            GenericDenseDataView(new int[] { 1, 2, 3 }, new int[] { 1, 10, 100 });
            GenericDenseDataView(new DvBool[] { true, true, true }, new DvBool[] { false, false, false });
            GenericDenseDataView(new double[] { 1, 2, 3 }, new double[] { 1, 10, 100 });
            GenericDenseDataView(new DvText[] { new DvText("a"), new DvText("b"), new DvText("c") },
                                 new DvText[] { new DvText("aa"), new DvText("bb"), new DvText("cc") });
        }

        private void GenericDenseDataView<T>(T[] v1, T[] v2)
        {
            var inputs = new[] {
                new DenseExample<T>() { X = v1 },
                new DenseExample<T>() { X = v2 }
            };
            using (var host = new TlcEnvironment())
            {
                var data = host.CreateStreamingDataView(inputs);
                var value = new VBuffer<T>();
                int n = 0;
                using (var cur = data.GetRowCursor(i => true))
                {
                    var getter = cur.GetGetter<VBuffer<T>>(0);
                    while (cur.MoveNext())
                    {
                        getter(ref value);
                        Assert.True(value.Count == 3);
                        ++n;
                    }
                }
                Assert.True(n == 2);
                var iter = data.AsEnumerable<DenseExample<T>>(host, false).GetEnumerator();
                n = 0;
                while (iter.MoveNext())
                    ++n;
                Assert.True(n == 2);
            }
        }
    }
}
