//
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
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

        class TestWrong
        {
            public string A;
            public float B;
        }

        [Fact]
        public void ValueMapOneValueTest()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };
            IEnumerable<int> values = new List<int>() { 1, 2, 3, 4 };

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, int>(Env, keys, values, new[] { ("A", "D"), ("B", "E"), ("C", "F") });
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

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory() };
            List<int[]> values = new List<int[]>() {
                new int[] {2, 3, 4 },
                new int[] {100, 200 },
                new int[] {400, 500, 600, 700 }};

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, int>(Env, keys, values, new[] { ("A", "D"), ("B", "E"), ("C", "F") });
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
        public void ValueMappingMissingKey()
        {
            var data = new[] { new TestClass() { A = "barTest", B = "test", C = "foo" } };
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };
            IEnumerable<int> values = new List<int>() { 1, 2, 3, 4 };

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, int>(Env, keys, values, new[] { ("A", "D"), ("B", "E"), ("C", "F") });
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursor((col) => true);
            var getterD = cursor.GetGetter<int>(3);
            var getterE = cursor.GetGetter<int>(4);
            var getterF = cursor.GetGetter<int>(5);
            cursor.MoveNext();

            int dValue = 1;
            getterD(ref dValue);
            Assert.Equal(0, dValue);
            int eValue = 0;
            getterE(ref eValue);
            Assert.Equal(3, eValue);
            int fValue = 0;
            getterF(ref fValue);
            Assert.Equal(1, fValue);
        }

        [Fact]
        public void ValueMappingOutputSchema()
        {
            var data = new[] { new TestClass() { A = "barTest", B = "test", C = "foo" } };
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };
            IEnumerable<int> values = new List<int>() { 1, 2, 3, 4 };

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, int>(Env, keys, values, new[] { ("A", "D"), ("B", "E"), ("C", "F") });
            var outputSchema  = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));
            Assert.Equal(6, outputSchema.Columns.Length);
            Assert.True(outputSchema.TryFindColumn("D", out SchemaShape.Column dColumn));
            Assert.True(outputSchema.TryFindColumn("E", out SchemaShape.Column eColumn));
            Assert.True(outputSchema.TryFindColumn("F", out SchemaShape.Column fColumn));

            Assert.Equal(typeof(int), dColumn.ItemType.RawType);
            Assert.False(dColumn.IsKey);

            Assert.Equal(typeof(int), eColumn.ItemType.RawType);
            Assert.False(eColumn.IsKey);
            
            Assert.Equal(typeof(int), fColumn.ItemType.RawType);
            Assert.False(fColumn.IsKey);
        }
/*
        [Fact]
        public void ValueMappingWithValuesAsKeyTypesOutputSchema()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };
            IEnumerable<ReadOnlyMemory<char>> values = new List<ReadOnlyMemory<char>>() { "t".AsMemory(), "s".AsMemory(), "u".AsMemory(), "v".AsMemory() };

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, ReadOnlyMemory<char>>(Env, keys, values, true, new[] { ("A", "D"), ("B", "E"), ("C", "F") });
            var outputSchema  = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));
            Assert.Equal(6, outputSchema.Columns.Length);
            Assert.True(outputSchema.TryFindColumn("D", out SchemaShape.Column dColumn));
            Assert.True(outputSchema.TryFindColumn("E", out SchemaShape.Column eColumn));
            Assert.True(outputSchema.TryFindColumn("F", out SchemaShape.Column fColumn));

            Assert.Equal(typeof(int), dColumn.ItemType.RawType);
            Assert.True(dColumn.IsKey);

            Assert.Equal(typeof(int), eColumn.ItemType.RawType);
            Assert.True(eColumn.IsKey);
            
            Assert.Equal(typeof(int), fColumn.ItemType.RawType);
            Assert.True(fColumn.IsKey);

            var t = estimator.Fit(dataView);
        }
        */

        [Fact]
        public void ValueMappingValuesAsKeyTypes()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };
            IEnumerable<uint> values = new List<uint>() { 51, 25, 42, 61 };

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, uint>(Env, keys, values, true, new[] { ("A", "D"), ("B", "E"), ("C", "F") });

            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursor((col) => true);
            var getterD = cursor.GetGetter<uint>(3);
            var getterE = cursor.GetGetter<uint>(4);
            var getterF = cursor.GetGetter<uint>(5);
            cursor.MoveNext();

            uint dValue = 1;
            getterD(ref dValue);
            Assert.Equal<uint>(1, dValue);
            uint eValue = 0;
            getterE(ref eValue);
            Assert.Equal<uint>(2, eValue);
            uint fValue = 0;
            getterF(ref fValue);
            Assert.Equal<uint>(0, fValue);
        }


        [Fact]
        public void ValueMappingWorkout()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var badData = new[] { new TestWrong() { A = "bar", B = 1.2f } };
            var badDataView = ComponentCreation.CreateDataView(Env, badData);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };
            IEnumerable<int> values = new List<int>() { 1, 2, 3, 4 };

            // Workout on value mapping
            var est = ML.Transforms.ValueMap(keys, values, new[] { ("A", "D"), ("B", "E"), ("C", "F") });
            TestEstimatorCore(est, validFitInput: dataView, invalidInput: badDataView);
        }

        [Fact]
        void TestCommandLine()
        {
            var dataFile = GetDataPath("QuotingData.csv");
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=valuemap{key=ID value=Text data="
                                    + dataFile
                                    + @" col=A:B loader=Text{col=ID:U8:0 col=Text:TX:1 sep=, header=+} } in=f:\1.txt" }), (int)0);
        }
    }
}
