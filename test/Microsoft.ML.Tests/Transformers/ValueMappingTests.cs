// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
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

        public class TestTermLookup
        {
            public string Label;
            public int GroupId;

            [VectorType(2107)]
            public float[] Features;
        };


        [Fact]
        public void ValueMapOneValueTest()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var keys = new List<string>() { "foo", "bar", "test", "wahoo" };
            var values = new List<int>() { 1, 2, 3, 4 };

            var estimator = new ValueMappingEstimator<string, int>(Env, keys, values, new[] { ("D", "A"), ("E", "B"), ("F", "C") });
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursorForAllColumns();
            var getterD = cursor.GetGetter<int>(result.Schema["D"]);
            var getterE = cursor.GetGetter<int>(result.Schema["E"]);
            var getterF = cursor.GetGetter<int>(result.Schema["F"]);
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
        public void ValueMapInputIsVectorTest()
        {
            var data = new[] { new TestClass() { A = "bar test foo", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };
            var values = new List<int>() { 1, 2, 3, 4 };

            var estimator = new WordTokenizingEstimator(Env, new[]{
                    new WordTokenizingEstimator.ColumnOptions("TokenizeA", "A")
                }).Append(new ValueMappingEstimator<ReadOnlyMemory<char>, int>(Env, keys, values, new[] { ("VecD", "TokenizeA"), ("E", "B"), ("F", "C") }));
            var schema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));
            Assert.True(schema.TryFindColumn("VecD", out var originalColumn));
            Assert.Equal(SchemaShape.Column.VectorKind.VariableVector, originalColumn.Kind);
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursorForAllColumns();
            var getterVecD = cursor.GetGetter<VBuffer<int>>(result.Schema["VecD"]);
            var getterE = cursor.GetGetter<int>(result.Schema["E"]);
            var getterF = cursor.GetGetter<int>(result.Schema["F"]);
            cursor.MoveNext();

            VBuffer<int> dValue = default;
            getterVecD(ref dValue);
            Assert.True(dValue.GetValues().SequenceEqual(new int[] { 2, 3, 1 }));

            int eValue = 0;
            getterE(ref eValue);
            Assert.Equal(3, eValue);
            int fValue = 0;
            getterF(ref fValue);
            Assert.Equal(1, fValue);
        }

        [Fact]
        public void ValueMapInputIsVectorAndValueAsStringKeyTypeTest()
        {
            var data = new[] { new TestClass() { A = "bar test foo", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };
            var values = new List<ReadOnlyMemory<char>>() { "a".AsMemory(), "b".AsMemory(), "c".AsMemory(), "d".AsMemory() };

            var estimator = new WordTokenizingEstimator(Env, new[]{
                    new WordTokenizingEstimator.ColumnOptions("TokenizeA", "A")
                }).Append(new ValueMappingEstimator<ReadOnlyMemory<char>, ReadOnlyMemory<char>>(Env, keys, values, true, new[] { ("VecD", "TokenizeA"), ("E", "B"), ("F", "C") }));
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursorForAllColumns();
            var getterVecD = cursor.GetGetter<VBuffer<uint>>(result.Schema["VecD"]);
            var getterE = cursor.GetGetter<uint>(result.Schema["E"]);
            var getterF = cursor.GetGetter<uint>(result.Schema["F"]);
            cursor.MoveNext();

            VBuffer<uint> dValue = default;
            getterVecD(ref dValue);
            Assert.True(dValue.GetValues().SequenceEqual(new uint[] { 2, 3, 1 }));

            uint eValue = 0;
            getterE(ref eValue);
            Assert.Equal(3u, eValue);
            uint fValue = 0;
            getterF(ref fValue);
            Assert.Equal(1u, fValue);
        }

        [Fact]
        public void ValueMapVectorValueTest()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            IEnumerable<string> keys = new List<string>() { "foo", "bar", "test" };
            List<int[]> values = new List<int[]>() {
                new int[] {2, 3, 4 },
                new int[] {100, 200 },
                new int[] {400, 500, 600, 700 }};

            var estimator = new ValueMappingEstimator<string, int>(Env, keys, values, new[] { ("D", "A"), ("E", "B"), ("F", "C") });
            var schema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));
            foreach (var name in new[] { "D", "E", "F" })
            {
                Assert.True(schema.TryFindColumn(name, out var originalColumn));
                Assert.Equal(SchemaShape.Column.VectorKind.VariableVector, originalColumn.Kind);
            }

            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursorForAllColumns();
            var getterD = cursor.GetGetter<VBuffer<int>>(result.Schema["D"]);
            var getterE = cursor.GetGetter<VBuffer<int>>(result.Schema["E"]);
            var getterF = cursor.GetGetter<VBuffer<int>>(result.Schema["F"]);
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

        class Map
        {
            public string Key;
            public int Value;
        }

        [Fact]
        public void ValueMapDataViewAsMapTest()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var map = new[] { new Map() { Key = "foo", Value = 1 },
                              new Map() { Key = "bar", Value = 2 },
                              new Map() { Key = "test", Value = 3 },
                              new Map() { Key = "wahoo", Value = 4 }
                            };
            var mapView = ML.Data.LoadFromEnumerable(map);

            var estimator = new ValueMappingEstimator(Env, mapView, "Key", "Value", new[] { ("D", "A"), ("E", "B"), ("F", "C") });
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursorForAllColumns();
            var getterD = cursor.GetGetter<int>(result.Schema["D"]);
            var getterE = cursor.GetGetter<int>(result.Schema["E"]);
            var getterF = cursor.GetGetter<int>(result.Schema["F"]);
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
        public void ValueMapVectorStringValueTest()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            IEnumerable<string> keys = new List<string>() { "foo", "bar", "test" };
            List<string[]> values = new List<string[]>() {
                new string[] {"foo", "bar" },
                new string[] {"forest", "city", "town" },
                new string[] {"winter", "summer", "autumn", "spring" }};

            var estimator = new ValueMappingEstimator<string, string>(Env, keys, values, new[] { ("D", "A"), ("E", "B"), ("F", "C") });
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);

            var cursor = result.GetRowCursorForAllColumns();
            var getterD = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(result.Schema[3]);
            var getterE = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(result.Schema[4]);
            var getterF = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(result.Schema[5]);
            cursor.MoveNext();

            VBuffer<ReadOnlyMemory<char>> dValue = default;
            getterD(ref dValue);
            Assert.Equal(3, dValue.Length);

            VBuffer<ReadOnlyMemory<char>> eValue = default;
            getterE(ref eValue);
            Assert.Equal(4, eValue.Length);

            VBuffer<ReadOnlyMemory<char>> fValue = default;
            getterF(ref fValue);
            Assert.Equal(2, fValue.Length);
        }

        [Fact]
        public void ValueMappingMissingKey()
        {
            var data = new[] { new TestClass() { A = "barTest", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var keys = new List<string>() { "foo", "bar", "test", "wahoo" };
            var values = new List<int>() { 1, 2, 3, 4 };

            var estimator = new ValueMappingEstimator<string, int>(Env, keys, values, new[] { ("D", "A"), ("E", "B"), ("F", "C") });
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursorForAllColumns();
            var getterD = cursor.GetGetter<int>(result.Schema["D"]);
            var getterE = cursor.GetGetter<int>(result.Schema["E"]);
            var getterF = cursor.GetGetter<int>(result.Schema["F"]);
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
        void TestDuplicateKeys()
        {
            var data = new[] { new TestClass() { A = "barTest", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var keys = new List<string>() { "foo", "foo" };
            var values = new List<int>() { 1, 2 };

            Assert.Throws<InvalidOperationException>(() => new ValueMappingEstimator<string, int>(Env, keys, values, new[] { ("D", "A"), ("E", "B"), ("F", "C") }));
        }

        [Fact]
        public void ValueMappingOutputSchema()
        {
            var data = new[] { new TestClass() { A = "barTest", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var keys = new List<string>() { "foo", "bar", "test", "wahoo" };
            var values = new List<int>() { 1, 2, 3, 4 };

            var estimator = new ValueMappingEstimator<string, int>(Env, keys, values, new[] { ("D", "A"), ("E", "B"), ("F", "C") });
            var outputSchema  = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));

            Assert.Equal(6, outputSchema.Count());
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

        [Fact]
        public void ValueMappingWithValuesAsKeyTypesOutputSchema()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var keys = new List<string>() { "foo", "bar", "test", "wahoo" };
            var values = new List<string>() { "t", "s", "u", "v" };

            var estimator = new ValueMappingEstimator<string, string>(Env, keys, values, true, new[] { ("D", "A"), ("E", "B"), ("F", "C") });
            var outputSchema  = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));
            Assert.Equal(6, outputSchema.Count());
            Assert.True(outputSchema.TryFindColumn("D", out SchemaShape.Column dColumn));
            Assert.True(outputSchema.TryFindColumn("E", out SchemaShape.Column eColumn));
            Assert.True(outputSchema.TryFindColumn("F", out SchemaShape.Column fColumn));

            Assert.Equal(typeof(uint), dColumn.ItemType.RawType);
            Assert.True(dColumn.IsKey);

            Assert.Equal(typeof(uint), eColumn.ItemType.RawType);
            Assert.True(eColumn.IsKey);

            Assert.Equal(typeof(uint), fColumn.ItemType.RawType);
            Assert.True(fColumn.IsKey);

            var t = estimator.Fit(dataView);
        }

        [Fact]
        public void ValueMappingValuesAsUintKeyTypes()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test2", C = "wahoo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var keys = new List<string>() { "foo", "bar", "test", "wahoo" };

            // These are the expected key type values
            var values = new List<uint>() { 51, 25, 42, 61 };

            var estimator = new ValueMappingEstimator<string, uint>(Env, keys, values, true, new[] { ("D", "A"), ("E", "B"), ("F", "C") });

            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursorForAllColumns();
            var getterD = cursor.GetGetter<uint>(result.Schema["D"]);
            var getterE = cursor.GetGetter<uint>(result.Schema["E"]);
            var getterF = cursor.GetGetter<uint>(result.Schema["F"]);
            cursor.MoveNext();

            // The expected values will contain the actual uints and are not generated.
            uint dValue = 1;
            getterD(ref dValue);
            Assert.Equal<uint>(25, dValue);

            // Should be 0 as test2 is a missing key
            uint eValue = 0;
            getterE(ref eValue);
            Assert.Equal<uint>(0, eValue);

            // Testing the last key
            uint fValue = 0;
            getterF(ref fValue);
            Assert.Equal<uint>(61, fValue);
        }


        [Fact]
        public void ValueMappingValuesAsUlongKeyTypes()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test2", C = "wahoo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var keys = new List<string>() { "foo", "bar", "test", "wahoo" };

            // These are the expected key type values
            var values = new List<ulong>() { 51, Int32.MaxValue, 42, 61 };

            var estimator = new ValueMappingEstimator<string, ulong>(Env, keys, values, true, new[] { ("D", "A"), ("E", "B"), ("F", "C") });

            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursorForAllColumns();
            var getterD = cursor.GetGetter<ulong>(result.Schema["D"]);
            var getterE = cursor.GetGetter<ulong>(result.Schema["E"]);
            var getterF = cursor.GetGetter<ulong>(result.Schema["F"]);
            cursor.MoveNext();

            // The expected values will contain the actual uints and are not generated.
            ulong dValue = 1;
            getterD(ref dValue);
            Assert.Equal<ulong>(Int32.MaxValue, dValue);

            // Should be 0 as test2 is a missing key
            ulong eValue = 0;
            getterE(ref eValue);
            Assert.Equal<ulong>(0, eValue);

            // Testing the last key
            ulong fValue = 0;
            getterF(ref fValue);
            Assert.Equal<ulong>(61, fValue);
        }

        [Fact]
        public void ValueMappingValuesAsStringKeyTypes()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "notfound" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var keys = new List<string>() { "foo", "bar", "test", "wahoo" };

            // Generating the list of strings for the key type values, note that foo1 is duplicated as intended to test that the same index value is returned
            var values = new List<string>() { "foo1", "foo2", "foo1", "foo3" };

            var estimator = new ValueMappingEstimator<string, string>(Env, keys, values, true, new[] { ("D", "A"), ("E", "B"), ("F", "C") });
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursorForAllColumns();
            var getterD = cursor.GetGetter<uint>(result.Schema["D"]);
            var getterE = cursor.GetGetter<uint>(result.Schema["E"]);
            var getterF = cursor.GetGetter<uint>(result.Schema["F"]);
            cursor.MoveNext();

            // The expected values will contain the generated key type values starting from 1.
            uint dValue = 1;
            getterD(ref dValue);
            Assert.Equal<uint>(2, dValue);

            // eValue will equal 1 since foo1 occurs first.
            uint eValue = 0;
            getterE(ref eValue);
            Assert.Equal<uint>(1, eValue);

            // fValue will be 0 since its missing
            uint fValue = 0;
            getterF(ref fValue);
            Assert.Equal<uint>(0, fValue);
        }

        [Fact]
        public void ValueMappingValuesAsKeyTypesReverseLookup()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "notfound" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };

            // Generating the list of strings for the key type values, note that foo1 is duplicated as intended to test that the same index value is returned
            var values = new List<ReadOnlyMemory<char>>() { "foo1".AsMemory(), "foo2".AsMemory(), "foo1".AsMemory(), "foo3".AsMemory() };

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, ReadOnlyMemory<char>>(Env, keys, values, true, new[] { ("D", "A") })
                            .Append(new KeyToValueMappingEstimator(Env, ("DOutput","D")));
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursorForAllColumns();
            var getterD = cursor.GetGetter<ReadOnlyMemory<char>>(result.Schema["DOutput"]);
            cursor.MoveNext();

            // The expected values will contain the generated key type values starting from 1.
            ReadOnlyMemory<char> dValue = default;
            getterD(ref dValue);
            Assert.Equal("foo2".AsMemory(), dValue);
        }

        [Fact]
        public void ValueMappingWorkout()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var badData = new[] { new TestWrong() { A = "bar", B = 1.2f } };
            var badDataView = ML.Data.LoadFromEnumerable(badData);

            var keyValuePairs = new List<KeyValuePair<string,int>>() {
                new KeyValuePair<string,int>("foo", 1),
                new KeyValuePair<string,int>("bar", 2),
                new KeyValuePair<string,int>("test", 3),
                new KeyValuePair<string,int>("wahoo", 4)
                };

            // Workout on value mapping
            var est = ML.Transforms.Conversion.MapValue(keyValuePairs, new ColumnOptions[] { ("D", "A"), ("E", "B"), ("F", "C") });
            TestEstimatorCore(est, validFitInput: dataView, invalidInput: badDataView);
        }

        [Fact]
        public void ValueMappingValueTypeIsVectorWorkout()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var badData = new[] { new TestWrong() { A = "bar", B = 1.2f } };
            var badDataView = ML.Data.LoadFromEnumerable(badData);

            var keyValuePairs = new List<KeyValuePair<string,int[]>>() {
                new KeyValuePair<string,int[]>("foo", new int[] {2, 3, 4 }),
                new KeyValuePair<string,int[]>("bar", new int[] {100, 200 }),
                new KeyValuePair<string,int[]>("test", new int[] {400, 500, 600, 700 }),
                };

            // Workout on value mapping
            var est = ML.Transforms.Conversion.MapValue(keyValuePairs, new ColumnOptions[] { ("D", "A"), ("E", "B"), ("F", "C") });
            TestEstimatorCore(est, validFitInput: dataView, invalidInput: badDataView);
        }

        [Fact]
        public void ValueMappingInputIsVectorWorkout()
        {
            var data = new[] { new TestClass() { B = "bar test foo" } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var badData = new[] { new TestWrong() { B = 1.2f } };
            var badDataView = ML.Data.LoadFromEnumerable(badData);

            var keyValuePairs = new List<KeyValuePair<ReadOnlyMemory<char>,int>>() {
                new KeyValuePair<ReadOnlyMemory<char>,int>("foo".AsMemory(), 1),
                new KeyValuePair<ReadOnlyMemory<char>,int>("bar".AsMemory(), 2),
                new KeyValuePair<ReadOnlyMemory<char>,int>("test".AsMemory(), 3),
                new KeyValuePair<ReadOnlyMemory<char>,int>("wahoo".AsMemory(), 4) 
                };

            var est = ML.Transforms.Text.TokenizeIntoWords("TokenizeB", "B")
                .Append(ML.Transforms.Conversion.MapValue(keyValuePairs, new ColumnOptions[] { ("VecB", "TokenizeB") }));
            TestEstimatorCore(est, validFitInput: dataView, invalidInput: badDataView);
        }

        [Fact]
        void TestCommandLine()
        {
            var dataFile = GetDataPath("QuotingData.csv");
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=valuemap{keyCol=ID valueCol=Text data="
                                    + dataFile
                                    + @" col=A:B loader=Text{col=ID:U8:0 col=Text:TX:1 sep=, header=+} } in=f:\1.txt" }), (int)0);
        }

        [Fact]
        void TestCommandLineNoLoader()
        {
            var dataFile = GetDataPath("lm.labels.txt");
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=valuemap{data="
                                    + dataFile
                                    + @" col=A:B } in=f:\1.txt" }), (int)0);
        }

        [Fact]
        void TestCommandLineNoLoaderWithColumnNames()
        {
            var dataFile = GetDataPath("lm.labels.txt");
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=valuemap{data="
                                    + dataFile
                                    + @" col=A:B keyCol=foo valueCol=bar} in=f:\1.txt" }), (int)0);
        }

        [Fact]
        void TestCommandLineNoLoaderWithoutTreatValuesAsKeys()
        {
            var dataFile = GetDataPath("lm.labels.txt");
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=valuemap{data="
                                    + dataFile
                                    + @" col=A:B valuesAsKeyType=-} in=f:\1.txt" }), (int)0);
        }

        [Fact]
        void TestSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = "bar", B = "foo", C = "test", } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var est = new ValueMappingEstimator<string, int>(Env, 
                                                new List<string>() { "foo", "bar", "test" }, 
                                                new List<int>() { 2, 43, 56 }, 
                                                new [] { ("D", "A"), ("E", "B") });

            var transformer = est.Fit(dataView);
            using (var ms = new MemoryStream())
            {
                ML.Model.Save(transformer, null, ms);
                ms.Position = 0;
                var loadedTransformer = ML.Model.Load(ms, out var schema);
                var result = loadedTransformer.Transform(dataView);
                Assert.Equal(5, result.Schema.Count);
                Assert.True(result.Schema.TryGetColumnIndex("D", out int col));
                Assert.True(result.Schema.TryGetColumnIndex("E", out col));
            }
        }


        [Fact]
        void TestValueMapBackCompatTermLookup()
        {
            // Model generated with: xf=drop{col=A} 
            // Expected output: Features Label B C
            var data = new[] { new TestTermLookup() { Label = "good", GroupId = 1 } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            string termLookupModelPath = GetDataPath("backcompat/termlookup.zip");
            using (FileStream fs = File.OpenRead(termLookupModelPath))
            {
                var result = ModelFileUtils.LoadTransforms(Env, dataView, fs);
                Assert.True(result.Schema.TryGetColumnIndex("Features", out int featureIdx));
                Assert.True(result.Schema.TryGetColumnIndex("Label", out int labelIdx));
                Assert.True(result.Schema.TryGetColumnIndex("GroupId", out int groupIdx));
            }
        }

        [Fact]
        void TestValueMapBackCompatTermLookupKeyTypeValue()
        {
            // Model generated with: xf=drop{col=A} 
            // Expected output: Features Label B C
            var data = new[] { new TestTermLookup() { Label = "Good", GroupId = 1 } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            string termLookupModelPath = GetDataPath("backcompat/termlookup_with_key.zip");
            using (FileStream fs = File.OpenRead(termLookupModelPath))
            {
                var result = ModelFileUtils.LoadTransforms(Env, dataView, fs);
                Assert.True(result.Schema.TryGetColumnIndex("Features", out int featureIdx));
                Assert.True(result.Schema.TryGetColumnIndex("Label", out int labelIdx));
                Assert.True(result.Schema.TryGetColumnIndex("GroupId", out int groupIdx));

                Assert.True(result.Schema[labelIdx].Type is KeyType);
                Assert.Equal((ulong)5, result.Schema[labelIdx].Type.GetItemType().GetKeyCount());

                var t = result.GetColumn<uint>(result.Schema["Label"]);
                uint s = t.First();
                Assert.Equal((uint)3, s);
            }
        }
    }
}
