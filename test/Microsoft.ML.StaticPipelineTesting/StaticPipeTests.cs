// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data.StaticPipe;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.TestFramework;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.StaticPipelineTesting
{
    public abstract class BaseTestClassWithConsole : BaseTestClass, IDisposable
    {
        private readonly TextWriter _originalOut;
        private readonly TextWriter _textWriter;

        public BaseTestClassWithConsole(ITestOutputHelper output)
            : base(output)
        {
            _originalOut = Console.Out;
            _textWriter = new StringWriter();
            Console.SetOut(_textWriter);
        }

        public void Dispose()
        {
            Output.WriteLine(_textWriter.ToString());
            Console.SetOut(_originalOut);
        }
    }

    public sealed class StaticPipeTests : BaseTestClassWithConsole
    {
        public StaticPipeTests(ITestOutputHelper output)
            : base(output)
        {
        }

        private void CheckSchemaHasColumn(ISchema schema, string name, out int idx)
            => Assert.True(schema.TryGetColumnIndex(name, out idx), "Could not find column '" + name + "'");

        [Fact]
        public void SimpleTextLoaderCopyColumnsTest()
        {
            var env = new ConsoleEnvironment(0, verbose: true);

            const string data = "0 hello 3.14159 -0 2\n"
                + "1 1 2 4 15";
            var dataSource = new BytesStreamSource(data);

            var text = TextLoader.CreateReader(env, ctx => (
                label: ctx.LoadBool(0),
                text: ctx.LoadText(1),
                numericFeatures: ctx.LoadFloat(2, null)), // If fit correctly, this ought to be equivalent to max of 4, that is, length of 3.
                dataSource, separator: ' ');

            // While we have a type-safe wrapper for `IDataView` it is utterly useless except as an input to the `Fit` functions
            // of the other statically typed wrappers. We perhaps ought to make it useful in its own right, but perhaps not now.
            // For now, just operate over the actual `IDataView`.
            var textData = text.Read(dataSource).AsDynamic;

            var schema = textData.Schema;
            // First verify that the columns are there. There ought to be at least one column corresponding to the identifiers in the tuple.
            CheckSchemaHasColumn(schema, "label", out int labelIdx);
            CheckSchemaHasColumn(schema, "text", out int textIdx);
            CheckSchemaHasColumn(schema, "numericFeatures", out int numericFeaturesIdx);
            // Next verify they have the expected types.
            Assert.Equal(BoolType.Instance, schema.GetColumnType(labelIdx));
            Assert.Equal(TextType.Instance, schema.GetColumnType(textIdx));
            Assert.Equal(new VectorType(NumberType.R4, 3), schema.GetColumnType(numericFeaturesIdx));
            // Next actually inspect the data.
            using (var cursor = textData.GetRowCursor(c => true))
            {
                var textGetter = cursor.GetGetter<ReadOnlyMemory<char>>(textIdx);
                var numericFeaturesGetter = cursor.GetGetter<VBuffer<float>>(numericFeaturesIdx);
                ReadOnlyMemory<char> textVal = default;
                var labelGetter = cursor.GetGetter<bool>(labelIdx);
                bool labelVal = default;
                VBuffer<float> numVal = default;

                void CheckValuesSame(bool bl, string tx, float v0, float v1, float v2)
                {
                    labelGetter(ref labelVal);
                    textGetter(ref textVal);
                    numericFeaturesGetter(ref numVal);
                    Assert.True(tx.AsSpan().SequenceEqual(textVal.Span));
                    Assert.Equal((bool)bl, labelVal);
                    Assert.Equal(3, numVal.Length);
                    Assert.Equal(v0, numVal.GetItemOrDefault(0));
                    Assert.Equal(v1, numVal.GetItemOrDefault(1));
                    Assert.Equal(v2, numVal.GetItemOrDefault(2));
                }

                Assert.True(cursor.MoveNext(), "Could not move even to first row");
                CheckValuesSame(false, "hello", 3.14159f, -0f, 2f);
                Assert.True(cursor.MoveNext(), "Could not move to second row");
                CheckValuesSame(true, "1", 2f, 4f, 15f);
                Assert.False(cursor.MoveNext(), "Moved to third row, but there should have been only two");
            }

            // The next step where we shuffle the names around a little bit is one where we are
            // testing out the implicit usage of copy columns.

            var est = text.MakeNewEstimator().Append(r => (text: r.label, label: r.numericFeatures));
            var newText = text.Append(est);
            var newTextData = newText.Fit(dataSource).Read(dataSource);

            schema = newTextData.AsDynamic.Schema;
            // First verify that the columns are there. There ought to be at least one column corresponding to the identifiers in the tuple.
            CheckSchemaHasColumn(schema, "label", out labelIdx);
            CheckSchemaHasColumn(schema, "text", out textIdx);
            // Next verify they have the expected types.
            Assert.Equal(BoolType.Instance, schema.GetColumnType(textIdx));
            Assert.Equal(new VectorType(NumberType.R4, 3), schema.GetColumnType(labelIdx));
        }

        private static KeyValuePair<string, ColumnType> P(string name, ColumnType type)
            => new KeyValuePair<string, ColumnType>(name, type);

        [Fact]
        public void AssertStaticSimple()
        {
            var env = new ConsoleEnvironment(0, verbose: true);
            var schema = new SimpleSchema(env,
                P("hello", TextType.Instance),
                P("my", new VectorType(NumberType.I8, 5)),
                P("friend", new KeyType(DataKind.U4, 0, 3)));
            var view = new EmptyDataView(env, schema);

            view.AssertStatic(env, c => (
                my: c.I8.Vector,
                friend: c.KeyU4.NoValue.Scalar,
                hello: c.Text.Scalar
            ));
        }

        private sealed class MetaCounted : ICounted
        {
            public long Position => 0;
            public long Batch => 0;
            public ValueGetter<UInt128> GetIdGetter() => (ref UInt128 v) => v = default;
        }

        [Fact]
        public void AssertStaticKeys()
        {
            var env = new ConsoleEnvironment(0, verbose: true);
            var counted = new MetaCounted();

            // We'll test a few things here. First, the case where the key-value metadata is text.
            var metaValues1 = new VBuffer<ReadOnlyMemory<char>>(3, new[] { "a".AsMemory(), "b".AsMemory(), "c".AsMemory() });
            var meta1 = RowColumnUtils.GetColumn(MetadataUtils.Kinds.KeyValues, new VectorType(TextType.Instance, 3), ref metaValues1);
            uint value1 = 2;
            var col1 = RowColumnUtils.GetColumn("stay", new KeyType(DataKind.U4, 0, 3), ref value1, RowColumnUtils.GetRow(counted, meta1));

            // Next the case where those values are ints.
            var metaValues2 = new VBuffer<int>(3, new int[] { 1, 2, 3, 4 });
            var meta2 = RowColumnUtils.GetColumn(MetadataUtils.Kinds.KeyValues, new VectorType(NumberType.I4, 4), ref metaValues2);
            var value2 = new VBuffer<byte>(2, 0, null, null);
            var col2 = RowColumnUtils.GetColumn("awhile", new VectorType(new KeyType(DataKind.U1, 2, 4), 2), ref value2, RowColumnUtils.GetRow(counted, meta2));

            // Then the case where a value of that kind exists, but is of not of the right kind, in which case it should not be identified as containing that metadata.
            var metaValues3 = (float)2;
            var meta3 = RowColumnUtils.GetColumn(MetadataUtils.Kinds.KeyValues, NumberType.R4, ref metaValues3);
            var value3 = (ushort)1;
            var col3 = RowColumnUtils.GetColumn("and", new KeyType(DataKind.U2, 0, 2), ref value3, RowColumnUtils.GetRow(counted, meta3));

            // Then a final case where metadata of that kind is actaully simply altogether absent.
            var value4 = new VBuffer<uint>(5, 0, null, null);
            var col4 = RowColumnUtils.GetColumn("listen", new VectorType(new KeyType(DataKind.U4, 0, 2)), ref value4);

            // Finally compose a trivial data view out of all this.
            var row = RowColumnUtils.GetRow(counted, col1, col2, col3, col4);
            var view = RowCursorUtils.RowAsDataView(env, row);

            // Whew! I'm glad that's over with. Let us start running the test in ernest.
            // First let's do a direct match of the types to ensure that works.
            view.AssertStatic(env, c => (
               stay: c.KeyU4.TextValues.Scalar,
               awhile: c.KeyU1.I4Values.Vector,
               and: c.KeyU2.NoValue.Scalar,
               listen: c.KeyU4.NoValue.VarVector));

            // Next let's match against the superclasses (where no value types are
            // asserted), to ensure that the less specific case still passes.
            view.AssertStatic(env, c => (
               stay: c.KeyU4.NoValue.Scalar,
               awhile: c.KeyU1.NoValue.Vector,
               and: c.KeyU2.NoValue.Scalar,
               listen: c.KeyU4.NoValue.VarVector));

            // Here we assert a subset.
            view.AssertStatic(env, c => (
               stay: c.KeyU4.TextValues.Scalar,
               awhile: c.KeyU1.I4Values.Vector));

            // OK. Now we've confirmed the basic stuff works, let's check other scenarios.
            // Due to the fact that we cannot yet assert only a *single* column, these always appear
            // in at least pairs.

            // First try to get the right type of exception to test against.
            Type e = null;
            try
            {
                view.AssertStatic(env, c => (
                   stay: c.KeyU4.TextValues.Scalar,
                   awhile: c.KeyU2.I4Values.Vector));
            }
            catch (Exception eCaught)
            {
                e = eCaught.GetType();
            }
            Assert.NotNull(e);

            // What if the key representation type is wrong?
            Assert.Throws(e, () =>
                view.AssertStatic(env, c => (
                   stay: c.KeyU4.TextValues.Scalar,
                   awhile: c.KeyU2.I4Values.Vector)));

            // What if the key value type is wrong?
            Assert.Throws(e, () =>
                view.AssertStatic(env, c => (
                   stay: c.KeyU4.TextValues.Scalar,
                   awhile: c.KeyU1.I2Values.Vector)));

            // Same two tests, but for scalar?
            Assert.Throws(e, () =>
                view.AssertStatic(env, c => (
                   stay: c.KeyU2.TextValues.Scalar,
                   awhile: c.KeyU1.I2Values.Vector)));

            Assert.Throws(e, () =>
                view.AssertStatic(env, c => (
                   stay: c.KeyU4.BoolValues.Scalar,
                   awhile: c.KeyU1.I2Values.Vector)));

            // How about if we misidentify the vectorness?
            Assert.Throws(e, () =>
                view.AssertStatic(env, c => (
                   stay: c.KeyU4.TextValues.Vector,
                   awhile: c.KeyU1.I2Values.Vector)));

            // How about the names?
            Assert.Throws(e, () =>
                view.AssertStatic(env, c => (
                   stay: c.KeyU4.TextValues.Scalar,
                   alot: c.KeyU1.I4Values.Vector)));
        }

        [Fact]
        public void Normalizer()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("generated_regression_dataset.csv");
            var dataSource = new MultiFileSource(dataPath);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);
            var data = reader.Read(dataSource);

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, r.features, bin: r.features.NormalizeByBinning(), mm: r.features.Normalize()));
            var tdata = est.Fit(data).Transform(data);

            var schema = tdata.AsDynamic.Schema;
            Assert.True(schema.TryGetColumnIndex("features", out int featCol));
            Assert.True(schema.TryGetColumnIndex("bin", out int binCol));
            Assert.True(schema.TryGetColumnIndex("mm", out int mmCol));
            Assert.False(schema.IsNormalized(featCol));
            Assert.True(schema.IsNormalized(binCol));
            Assert.True(schema.IsNormalized(mmCol));
        }

        [Fact]
        public void NormalizerWithOnFit()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("generated_regression_dataset.csv");
            var dataSource = new MultiFileSource(dataPath);

            var reader = TextLoader.CreateReader(env,
                c => c.LoadFloat(0, 2),
                separator: ';', hasHeader: true);
            var data = reader.Read(dataSource);

            // These will be populated once we call fit.
            ImmutableArray<float> mm;
            ImmutableArray<float> ss;
            ImmutableArray<ImmutableArray<float>> bb;

            var est = reader.MakeNewEstimator()
                .Append(r => (r, 
                    ncdf: r.NormalizeByCumulativeDistribution(onFit: (m, s) => mm = m),
                    n: r.NormalizeByMeanVar(onFit: (s, o) => { ss = s; Assert.Empty(o); }),
                    b: r.NormalizeByBinning(onFit: b => bb = b)));
            var tdata = est.Fit(data).Transform(data);

            Assert.Equal(3, mm.Length);
            Assert.Equal(3, ss.Length);
            Assert.Equal(3, bb.Length);

            // Just for fun, let's also write out some of the lines of the data to the console.
            using (var stream = new MemoryStream())
            {
                IDataView v = new ChooseColumnsTransform(env, tdata.AsDynamic, "r", "ncdf", "n", "b");
                v = TakeFilter.Create(env, v, 10);
                var saver = new TextSaver(env, new TextSaver.Arguments()
                {
                    Dense = true,
                    Separator = ",",
                    OutputHeader = false
                });
                saver.SaveData(stream, v, Utils.GetIdentityPermutation(v.Schema.ColumnCount));
                Console.WriteLine(Encoding.UTF8.GetString(stream.ToArray()));
            }
        }

        [Fact]
        public void ToKey()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("iris.data");
            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadText(4), values: c.LoadFloat(0, 3)),
                separator: ',');
            var dataSource = new MultiFileSource(dataPath);
            var data = reader.Read(dataSource);

            var est = data.MakeNewEstimator()
                .Append(r => (labelKey: r.label.ToKey(), valuesKey: r.values.ToKey(onFit: m => { })))
                .Append(r => (r.labelKey, r.valuesKey, valuesKeyKey: r.valuesKey.ToKey()));

            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;
            Assert.True(schema.TryGetColumnIndex("labelKey", out int labelCol));
            Assert.True(schema.TryGetColumnIndex("valuesKey", out int valuesCol));
            Assert.True(schema.TryGetColumnIndex("valuesKeyKey", out int valuesKeyCol));

            Assert.Equal(3, schema.GetColumnType(labelCol).KeyCount);
            Assert.True(schema.GetColumnType(valuesCol).ItemType.IsKey);
            Assert.True(schema.GetColumnType(valuesKeyCol).ItemType.IsKey);

            var labelKeyType = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, labelCol);
            var valuesKeyType = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, valuesCol);
            var valuesKeyKeyType = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, valuesKeyCol);
            Assert.NotNull(labelKeyType);
            Assert.NotNull(valuesKeyType);
            Assert.NotNull(valuesKeyKeyType);
            Assert.True(labelKeyType.IsVector && labelKeyType.ItemType == TextType.Instance);
            Assert.True(valuesKeyType.IsVector && valuesKeyType.ItemType == NumberType.Float);
            Assert.True(valuesKeyKeyType.IsVector && valuesKeyKeyType.ItemType == NumberType.Float);
            // Because they're over exactly the same data, they ought to have the same cardinality and everything.
            Assert.True(valuesKeyKeyType.Equals(valuesKeyType));
        }

        [Fact]
        public void ConcatWith()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("iris.data");
            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadText(4), values: c.LoadFloat(0, 3), value: c.LoadFloat(2)),
                separator: ',');
            var dataSource = new MultiFileSource(dataPath);
            var data = reader.Read(dataSource);

            var est = data.MakeNewEstimator()
                .Append(r => (
                    r.label, r.values, r.value,
                    c0: r.label.AsVector(), c1: r.label.ConcatWith(r.label),
                    c2: r.value.ConcatWith(r.values), c3: r.values.ConcatWith(r.value, r.values)));

            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;

            int[] idx = new int[4];
            for (int i = 0; i < idx.Length; ++i)
                Assert.True(schema.TryGetColumnIndex("c" + i, out idx[i]), $"Could not find col c{i}");
            var types = new VectorType[idx.Length];
            int[] expectedLen = new int[] { 1, 2, 5, 9 };
            for (int i = 0; i < idx.Length; ++i)
            {
                var type = schema.GetColumnType(idx[i]);
                Assert.True(type.VectorSize > 0, $"Col c{i} had unexpected type {type}");
                types[i] = type.AsVector;
                Assert.Equal(expectedLen[i], type.VectorSize);
            }
            Assert.Equal(TextType.Instance, types[0].ItemType);
            Assert.Equal(TextType.Instance, types[1].ItemType);
            Assert.Equal(NumberType.Float, types[2].ItemType);
            Assert.Equal(NumberType.Float, types[3].ItemType);
        }
    }
}
