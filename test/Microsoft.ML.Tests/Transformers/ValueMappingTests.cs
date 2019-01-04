// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms.Conversions;
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
        void TestDuplicateKeys()
        {
            var data = new[] { new TestClass() { A = "barTest", B = "test", C = "foo" } };
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "foo".AsMemory() };
            IEnumerable<int> values = new List<int>() { 1, 2 };

            Assert.Throws<InvalidOperationException>(() => new ValueMappingEstimator<ReadOnlyMemory<char>, int>(Env, keys, values, new[] { ("A", "D"), ("B", "E"), ("C", "F") }));
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
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };
            IEnumerable<ReadOnlyMemory<char>> values = new List<ReadOnlyMemory<char>>() { "t".AsMemory(), "s".AsMemory(), "u".AsMemory(), "v".AsMemory() };

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, ReadOnlyMemory<char>>(Env, keys, values, true, new[] { ("A", "D"), ("B", "E"), ("C", "F") });
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
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };

            // These are the expected key type values
            IEnumerable<uint> values = new List<uint>() { 51, 25, 42, 61 };

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, uint>(Env, keys, values, true, new[] { ("A", "D"), ("B", "E"), ("C", "F") });

            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursor((col) => true);
            var getterD = cursor.GetGetter<uint>(3);
            var getterE = cursor.GetGetter<uint>(4);
            var getterF = cursor.GetGetter<uint>(5);
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
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };

            // These are the expected key type values
            IEnumerable<ulong> values = new List<ulong>() { 51, Int32.MaxValue, 42, 61 };

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, ulong>(Env, keys, values, true, new[] { ("A", "D"), ("B", "E"), ("C", "F") });

            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursor((col) => true);
            var getterD = cursor.GetGetter<ulong>(3);
            var getterE = cursor.GetGetter<ulong>(4);
            var getterF = cursor.GetGetter<ulong>(5);
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
            var dataView = ComponentCreation.CreateDataView(Env, data);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };

            // Generating the list of strings for the key type values, note that foo1 is duplicated as intended to test that the same index value is returned
            IEnumerable<ReadOnlyMemory<char>> values = new List<ReadOnlyMemory<char>>() { "foo1".AsMemory(), "foo2".AsMemory(), "foo1".AsMemory(), "foo3".AsMemory() };

            var estimator = new ValueMappingEstimator<ReadOnlyMemory<char>, ReadOnlyMemory<char>>(Env, keys, values, true, new[] { ("A", "D"), ("B", "E"), ("C", "F") });
            var t = estimator.Fit(dataView);

            var result = t.Transform(dataView);
            var cursor = result.GetRowCursor((col) => true);
            var getterD = cursor.GetGetter<uint>(3);
            var getterE = cursor.GetGetter<uint>(4);
            var getterF = cursor.GetGetter<uint>(5);
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
        public void ValueMappingWorkout()
        {
            var data = new[] { new TestClass() { A = "bar", B = "test", C = "foo" } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var badData = new[] { new TestWrong() { A = "bar", B = 1.2f } };
            var badDataView = ComponentCreation.CreateDataView(Env, badData);

            IEnumerable<ReadOnlyMemory<char>> keys = new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory(), "wahoo".AsMemory() };
            IEnumerable<int> values = new List<int>() { 1, 2, 3, 4 };

            // Workout on value mapping
            var est = ML.Transforms.Conversion.ValueMap(keys, values, new[] { ("A", "D"), ("B", "E"), ("C", "F") });
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
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = new ValueMappingEstimator<ReadOnlyMemory<char>, int>(Env, 
                                                new List<ReadOnlyMemory<char>>() { "foo".AsMemory(), "bar".AsMemory(), "test".AsMemory() }, 
                                                new List<int>() { 2, 43, 56 }, 
                                                new [] {("A","D"), ("B", "E")});
            var transformer = est.Fit(dataView);
            using (var ms = new MemoryStream())
            {
                transformer.SaveTo(Env, ms);
                ms.Position = 0;
                var loadedTransformer = TransformerChain.LoadFrom(Env, ms);
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
            var data = new[] { new TestTermLookup() { Label = "good", GroupId=1 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
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
            var data = new[] { new TestTermLookup() { Label = "Good", GroupId=1 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            string termLookupModelPath = GetDataPath("backcompat/termlookup_with_key.zip");
            using (FileStream fs = File.OpenRead(termLookupModelPath))
            {
                var result = ModelFileUtils.LoadTransforms(Env, dataView, fs);
                Assert.True(result.Schema.TryGetColumnIndex("Features", out int featureIdx));
                Assert.True(result.Schema.TryGetColumnIndex("Label", out int labelIdx));
                Assert.True(result.Schema.TryGetColumnIndex("GroupId", out int groupIdx));
                
                Assert.True(result.Schema[labelIdx].Type.IsKey);
                Assert.Equal(5, result.Schema[labelIdx].Type.ItemType.KeyCount);

                var t = result.GetColumn<uint>(Env, "Label");
                uint s = t.First();
                Assert.Equal((uint)3, s);
            }
        }
    }
}
