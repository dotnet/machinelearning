// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Tools;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed partial class TestSparseDataView : TestDataViewBase
    {
        const string Cat = "DataView";

        public TestSparseDataView(ITestOutputHelper obj): base(obj)
        {
        }

        class ExampleA
        {
            [VectorType(2)]
            public float[] X;
        }

        class ExampleASparse
        {
            [VectorType(5)]
            public VBuffer<float> X;
        }

        [Fact]
        [TestCategory(Cat)]
        public void SparseDataView()
        {
            var inputs = new[] {
                new ExampleASparse() { X = new VBuffer<float> (5, 3, new float[] { 1, 10, 100 }, new int[] { 0, 2, 4 }) },
                new ExampleASparse() { X = new VBuffer<float> (5, 3, new float[] { 2, 3, 5 }, new int[] { 0, 1, 3 }) }
            };
            var host = new TlcEnvironment();
            var data = host.CreateStreamingDataView(inputs);
            VBuffer<float> value = new VBuffer<float>();
            int n = 0;
            using (var cur = data.GetRowCursor(i => true))
            {
                var getter = cur.GetGetter<VBuffer<float>>(0);
                while (cur.MoveNext())
                {
                    getter(ref value);
                    Assert.True(value.Count == 3);
                    ++n;
                }
            }
            Assert.True(n == 2);
            Done();
        }

        [Fact]
        [TestCategory(Cat)]
        public void DenseDataView()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };
            var host = new TlcEnvironment();
            var data = host.CreateStreamingDataView(inputs);
            VBuffer<float> value = new VBuffer<float>();
            int n = 0;
            using (var cur = data.GetRowCursor(i => true))
            {
                var getter = cur.GetGetter<VBuffer<float>>(0);
                while (cur.MoveNext())
                {
                    getter(ref value);
                    Assert.True(value.Count == 3);
                    ++n;
                }
            }
            Assert.True(n == 2);
            Done();
        }
    }
}
