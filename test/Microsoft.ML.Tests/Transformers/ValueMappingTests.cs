//
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class ValueMappingTests : TestDataPipeBase
    {
        public ValueMappingTests(ITestOutputHelper output) : base(output)
        {
        }

        class TestClass
        {
            public string A;
            public string B;
            public string C;
        }

        [Fact]
        public void ValueMapOneValueTest()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory()};
            IEnumerable<int> values = new List<int>() { 1, 2, 3, 4 };

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, int>(Env, keys, values, new [] { ("A", "D"), ("B", "E"), ("C", "F") });
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursor((col) => true);
            var getterD = cursor.GetGetter<int>(3);
            var getterE = cursor.GetGetter<int>(4);
            var getterF = cursor.GetGetter<int>(5);
            cursor.MoveNext();

            int dValue = 0;
            getterD(ref dValue);
            Assert.Equal(2, dValue);
            int eValue = 0;
            getterE(ref eValue);
            Assert.Equal(3, eValue);
            int fValue = 0;
            getterF(ref fValue);
            Assert.Equal(1, fValue);
        }

        [Fact]
        public void ValueMapVectorValueTest()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory()};
            List<int[]> values = new List<int[]>() { 
                new int[] {2, 3, 4 },
                new int[] {100, 200 },
                new int[] {400, 500, 600, 700 }};

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, int>(Env, keys, values, new [] { ("A", "D"), ("B", "E"),("C", "F") });
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursor((col) => true);
            var getterD = cursor.GetGetter<VBuffer<int>>(3);
            var getterE = cursor.GetGetter<VBuffer<int>>(4);
            var getterF = cursor.GetGetter<VBuffer<int>>(5);
            cursor.MoveNext();

            var valuesArray = values.ToArray();
            VBuffer<int> dValue = default;
            getterD(ref dValue);
            Assert.Equal(values[1].Length, dValue.Length);
            VBuffer<int> eValue = default;
            getterE(ref eValue);
            Assert.Equal(values[2].Length, eValue.Length);
            VBuffer<int> fValue = default;
            getterF(ref fValue);
            Assert.Equal(values[0].Length, fValue.Length);
        }

        [Fact]
        public void ValueMappingWorkout()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory()};
            IEnumerable<int> values = new List<int>() { 1, 2, 3, 4 };

            // Workout on value mapping
            //var est = ML.Transforms.ValueMap<ReadOnlyMemory<char>, int>(keys, values, new [] { ("A", "D"), ("B", "E"),("C", "F") });
            //TestEstimatorCore(est, validFitInput: dataView);
        }
    }
}
