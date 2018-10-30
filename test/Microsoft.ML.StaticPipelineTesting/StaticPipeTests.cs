// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Recommender;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.PCA;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
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

        private sealed class Obnoxious1
        {
            public Scalar<string> Foo { get; }
            public Vector<float> Bar { get; }

            public Obnoxious1(Scalar<string> f1, Vector<float> f2)
            {
                Foo = f1;
                Bar = f2;
            }
        }

        private sealed class Obnoxious2
        {
            public Scalar<string> Biz { get; set; }
            public Vector<double> Blam { get; set; }
        }

        private sealed class Obnoxious3<T>
        {
            public (Scalar<bool> hi, Obnoxious1 my, T friend) Donut { get; set; }
        }

        private static Obnoxious3<T> MakeObnoxious3<T>(Scalar<bool> hi, Obnoxious1 my, T friend)
            => new Obnoxious3<T>() { Donut = (hi, my, friend) };

        [Fact]
        public void SimpleTextLoaderObnoxiousTypeTest()
        {
            var env = new ConsoleEnvironment(0, verbose: true);

            const string data = "0 hello 3.14159 -0 2\n"
                + "1 1 2 4 15";
            var dataSource = new BytesStreamSource(data);

            // Ahhh. No one would ever, ever do this, of course, but just having fun with it.

            void Helper(ISchema thisSchema, string name, ColumnType expected)
            {
                Assert.True(thisSchema.TryGetColumnIndex(name, out int thisCol), $"Could not find column '{name}'");
                Assert.Equal(expected, thisSchema.GetColumnType(thisCol));
            }

            var text = TextLoader.CreateReader(env, ctx => (
                yo: new Obnoxious1(ctx.LoadText(0), ctx.LoadFloat(1, 5)),
                dawg: new Obnoxious2() { Biz = ctx.LoadText(2), Blam = ctx.LoadDouble(1, 2) },
                how: MakeObnoxious3(ctx.LoadBool(2), new Obnoxious1(ctx.LoadText(0), ctx.LoadFloat(1, 4)),
                    new Obnoxious2() { Biz = ctx.LoadText(5), Blam = ctx.LoadDouble(1, 10) })));

            var schema = text.AsDynamic.GetOutputSchema();
            Helper(schema, "yo.Foo", TextType.Instance);
            Helper(schema, "yo.Bar", new VectorType(NumberType.Float, 5));
            Helper(schema, "dawg.Biz", TextType.Instance);
            Helper(schema, "dawg.Blam", new VectorType(NumberType.R8, 2));

            Helper(schema, "how.Donut.hi", BoolType.Instance);
            Helper(schema, "how.Donut.my.Foo", TextType.Instance);
            Helper(schema, "how.Donut.my.Bar", new VectorType(NumberType.Float, 4));
            Helper(schema, "how.Donut.friend.Biz", TextType.Instance);
            Helper(schema, "how.Donut.friend.Blam", new VectorType(NumberType.R8, 10));

            var textData = text.Read(null);

            var est = text.MakeNewEstimator().Append(r => r.how.Donut.friend.Blam.ConcatWith(r.dawg.Blam));
            var outData = est.Fit(textData).Transform(textData);

            var xfSchema = outData.AsDynamic.Schema;
            Helper(xfSchema, "Data", new VectorType(NumberType.R8, 12));
        }

        private static KeyValuePair<string, ColumnType> P(string name, ColumnType type)
            => new KeyValuePair<string, ColumnType>(name, type);

        [Fact]
        public void AssertStaticSimple()
        {
            var env = new ConsoleEnvironment(0, verbose: true);
            var schema = SimpleSchemaUtils.Create(env,
                P("hello", TextType.Instance),
                P("my", new VectorType(NumberType.I8, 5)),
                P("friend", new KeyType(DataKind.U4, 0, 3)));
            var view = new EmptyDataView(env, schema);

            view.AssertStatic(env, c => new
            {
                my = c.I8.Vector,
                friend = c.KeyU4.NoValue.Scalar,
                hello = c.Text.Scalar
            });

            view.AssertStatic(env, c => (
                my: c.I8.Vector,
                friend: c.KeyU4.NoValue.Scalar,
                hello: c.Text.Scalar
            ));
        }

        [Fact]
        public void AssertStaticSimpleFailure()
        {
            var env = new ConsoleEnvironment(0, verbose: true);
            var schema = SimpleSchemaUtils.Create(env,
                P("hello", TextType.Instance),
                P("my", new VectorType(NumberType.I8, 5)),
                P("friend", new KeyType(DataKind.U4, 0, 3)));
            var view = new EmptyDataView(env, schema);

            Assert.ThrowsAny<Exception>(() =>
                view.AssertStatic(env, c => new
                {
                    my = c.I8.Scalar, // Shouldn't work, the type is wrong.
                    friend = c.KeyU4.NoValue.Scalar,
                    hello = c.Text.Scalar
                }));

            Assert.ThrowsAny<Exception>(() =>
                view.AssertStatic(env, c => (
                    mie: c.I8.Vector, // Shouldn't work, the name is wrong.
                    friend: c.KeyU4.NoValue.Scalar,
                    hello: c.Text.Scalar)));
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

        [Fact]
        public void Tokenize()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var reader = TextLoader.CreateReader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);
            var dataSource = new MultiFileSource(dataPath);
            var data = reader.Read(dataSource);

            var est = data.MakeNewEstimator()
                .Append(r => (
                    r.label,
                    tokens: r.text.TokenizeText(),
                    chars: r.text.TokenizeIntoCharacters()));

            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;

            Assert.True(schema.TryGetColumnIndex("tokens", out int tokensCol));
            var type = schema.GetColumnType(tokensCol);
            Assert.True(type.IsVector && !type.IsKnownSizeVector && type.ItemType.IsText);

            Assert.True(schema.TryGetColumnIndex("chars", out int charsCol));
            type = schema.GetColumnType(charsCol);
            Assert.True(type.IsVector && !type.IsKnownSizeVector && type.ItemType.IsKey);
            Assert.True(type.ItemType.AsKey.RawKind == DataKind.U2);
        }

        [Fact]
        public void NormalizeTextAndRemoveStopWords()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var reader = TextLoader.CreateReader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);
            var dataSource = new MultiFileSource(dataPath);
            var data = reader.Read(dataSource);

            var est = data.MakeNewEstimator()
                .Append(r => (
                    r.label,
                    normalized_text: r.text.NormalizeText(),
                    words_without_stopwords: r.text.TokenizeText().RemoveStopwords()));

            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;

            Assert.True(schema.TryGetColumnIndex("words_without_stopwords", out int stopwordsCol));
            var type = schema.GetColumnType(stopwordsCol);
            Assert.True(type.IsVector && !type.IsKnownSizeVector && type.ItemType.IsText);

            Assert.True(schema.TryGetColumnIndex("normalized_text", out int normTextCol));
            type = schema.GetColumnType(normTextCol);
            Assert.True(type.IsText && !type.IsVector);
        }

        [Fact]
        public void ConvertToWordBag()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var reader = TextLoader.CreateReader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);
            var dataSource = new MultiFileSource(dataPath);
            var data = reader.Read(dataSource);

            var est = data.MakeNewEstimator()
                .Append(r => (
                    r.label,
                    bagofword: r.text.ToBagofWords(),
                    bagofhashedword: r.text.ToBagofHashedWords()));

            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;

            Assert.True(schema.TryGetColumnIndex("bagofword", out int bagofwordCol));
            var type = schema.GetColumnType(bagofwordCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);

            Assert.True(schema.TryGetColumnIndex("bagofhashedword", out int bagofhashedwordCol));
            type = schema.GetColumnType(bagofhashedwordCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);
        }

        [Fact]
        public void Ngrams()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var reader = TextLoader.CreateReader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);
            var dataSource = new MultiFileSource(dataPath);
            var data = reader.Read(dataSource);

            var est = data.MakeNewEstimator()
                .Append(r => (
                    r.label,
                    ngrams: r.text.TokenizeText().ToKey().ToNgrams(),
                    ngramshash: r.text.TokenizeText().ToKey().ToNgramsHash()));

            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;

            Assert.True(schema.TryGetColumnIndex("ngrams", out int ngramsCol));
            var type = schema.GetColumnType(ngramsCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);

            Assert.True(schema.TryGetColumnIndex("ngramshash", out int ngramshashCol));
            type = schema.GetColumnType(ngramshashCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);
        }


        [Fact]
        public void LpGcNormAndWhitening()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("generated_regression_dataset.csv");
            var dataSource = new MultiFileSource(dataPath);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);
            var data = reader.Read(dataSource);

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label,
                              lpnorm: r.features.LpNormalize(),
                              gcnorm: r.features.GlobalContrastNormalize(),
                              zcawhitened: r.features.ZcaWhitening(),
                              pcswhitened: r.features.PcaWhitening()));
            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;

            Assert.True(schema.TryGetColumnIndex("lpnorm", out int lpnormCol));
            var type = schema.GetColumnType(lpnormCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);

            Assert.True(schema.TryGetColumnIndex("gcnorm", out int gcnormCol));
            type = schema.GetColumnType(gcnormCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);

            Assert.True(schema.TryGetColumnIndex("zcawhitened", out int zcawhitenedCol));
            type = schema.GetColumnType(zcawhitenedCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);

            Assert.True(schema.TryGetColumnIndex("pcswhitened", out int pcswhitenedCol));
            type = schema.GetColumnType(pcswhitenedCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);
        }

        [Fact]
        public void LdaTopicModel()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var reader = TextLoader.CreateReader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);
            var dataSource = new MultiFileSource(dataPath);
            var data = reader.Read(dataSource);

            var est = data.MakeNewEstimator()
                .Append(r => (
                    r.label,
                    topics: r.text.ToBagofWords().ToLdaTopicVector(numTopic: 10, advancedSettings: s =>
                    {
                        s.AlphaSum = 10;
                    })));

            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;

            Assert.True(schema.TryGetColumnIndex("topics", out int topicsCol));
            var type = schema.GetColumnType(topicsCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);
        }

        [Fact]
        public void FeatureSelection()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var reader = TextLoader.CreateReader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);
            var dataSource = new MultiFileSource(dataPath);
            var data = reader.Read(dataSource);

            var est = data.MakeNewEstimator()
                .Append(r => (
                    r.label,
                    bag_of_words_count: r.text.ToBagofWords().SelectFeaturesBasedOnCount(10),
                    bag_of_words_mi: r.text.ToBagofWords().SelectFeaturesBasedOnMutualInformation(r.label)));

            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;

            Assert.True(schema.TryGetColumnIndex("bag_of_words_count", out int bagofwordCountCol));
            var type = schema.GetColumnType(bagofwordCountCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);

            Assert.True(schema.TryGetColumnIndex("bag_of_words_mi", out int bagofwordMiCol));
            type = schema.GetColumnType(bagofwordMiCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);
        }

        [Fact]
        public void TrainTestSplit()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var ctx = new BinaryClassificationContext(env);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(0), features: c.LoadFloat(1, 4)));
            var data = reader.Read(dataSource);

            var (train, test) = ctx.TrainTestSplit(data, 0.5);

            // Just make sure that the train is about the same size as the test set.
            var trainCount = train.GetColumn(r => r.label).Count();
            var testCount = test.GetColumn(r => r.label).Count();

            Assert.InRange(trainCount * 1.0 / testCount, 0.8, 1.2);

            // Now stratify by label. Silly thing to do.
            (train, test) = ctx.TrainTestSplit(data, 0.5, stratificationColumn: r => r.label);
            var trainLabels = train.GetColumn(r => r.label).Distinct();
            var testLabels = test.GetColumn(r => r.label).Distinct();
            Assert.True(trainLabels.Count() > 0);
            Assert.True(testLabels.Count() > 0);
            Assert.False(trainLabels.Intersect(testLabels).Any());
        }

        [Fact]
        public void PrincipalComponentAnalysis()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("generated_regression_dataset.csv");
            var dataSource = new MultiFileSource(dataPath);

            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);
            var data = reader.Read(dataSource);

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label,
                              pca: r.features.ToPrincipalComponents(rank: 5)));
            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;

            Assert.True(schema.TryGetColumnIndex("pca", out int pcaCol));
            var type = schema.GetColumnType(pcaCol);
            Assert.True(type.IsVector && type.IsKnownSizeVector && type.ItemType.IsNumber);
        }

        [Fact]
        public void NAIndicatorStatic()
        {
            var Env = new ConsoleEnvironment(seed: 0);

            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoader.CreateReader(Env, ctx => (
                ScalarFloat: ctx.LoadFloat(1),
                ScalarDouble: ctx.LoadDouble(1),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDoulbe: ctx.LoadDouble(1, 4)
            ));

            var data = reader.Read(new MultiFileSource(dataPath));

            var est = data.MakeNewEstimator().
                   Append(row => (
                   A: row.ScalarFloat.IsMissingValue(),
                   B: row.ScalarDouble.IsMissingValue(),
                   C: row.VectorFloat.IsMissingValue(),
                   D: row.VectorDoulbe.IsMissingValue()
                   ));

            IDataView newData = TakeFilter.Create(Env, est.Fit(data).Transform(data).AsDynamic, 4);
            Assert.NotNull(newData);
            bool[] ScalarFloat = newData.GetColumn<bool>(Env, "A").ToArray();
            bool[] ScalarDouble = newData.GetColumn<bool>(Env, "B").ToArray();
            bool[][] VectorFloat = newData.GetColumn<bool[]>(Env, "C").ToArray();
            bool[][] VectorDoulbe = newData.GetColumn<bool[]>(Env, "D").ToArray();

            Assert.NotNull(ScalarFloat);
            Assert.NotNull(ScalarDouble);
            Assert.NotNull(VectorFloat);
            Assert.NotNull(VectorDoulbe);
            for (int i = 0; i < 4; i++)
            {
                Assert.True(!ScalarFloat[i] && !ScalarDouble[i]);
                Assert.NotNull(VectorFloat[i]);
                Assert.NotNull(VectorDoulbe[i]);
                for (int j = 0; j < 4; j++)
                    Assert.True(!VectorFloat[i][j] && !VectorDoulbe[i][j]);
            }
        }
        
        [Fact]
        public void TextNormalizeStatic()
        {
            var env = new ConsoleEnvironment(seed: 0);
            var dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var reader = TextLoader.CreateReader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);
            var dataSource = new MultiFileSource(dataPath);
            var data = reader.Read(dataSource);

            var est = data.MakeNewEstimator()
                .Append(r => (
                    r.label,
                    norm: r.text.NormalizeText(),
                    norm_Upper: r.text.NormalizeText(textCase: TextNormalizerEstimator.CaseNormalizationMode.Upper),
                    norm_KeepDiacritics: r.text.NormalizeText(keepDiacritics: true),
                    norm_NoPuctuations: r.text.NormalizeText(keepPunctuations: false),
                    norm_NoNumbers: r.text.NormalizeText(keepNumbers: false)));
            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;

            Assert.True(schema.TryGetColumnIndex("norm", out int norm));
            var type = schema.GetColumnType(norm);
            Assert.True(!type.IsVector && type.ItemType.IsText);

            Assert.True(schema.TryGetColumnIndex("norm_Upper", out int normUpper));
            type = schema.GetColumnType(normUpper);
            Assert.True(!type.IsVector && type.ItemType.IsText);
            Assert.True(schema.TryGetColumnIndex("norm_KeepDiacritics", out int diacritics));
            type = schema.GetColumnType(diacritics);
            Assert.True(!type.IsVector && type.ItemType.IsText);
            Assert.True(schema.TryGetColumnIndex("norm_NoPuctuations", out int punct));
            type = schema.GetColumnType(punct);
            Assert.True(!type.IsVector && type.ItemType.IsText);
            Assert.True(schema.TryGetColumnIndex("norm_NoNumbers", out int numbers));
            type = schema.GetColumnType(numbers);
            Assert.True(!type.IsVector && type.ItemType.IsText);
        }

        [Fact]
        public void TestPcaStatic()
        {
            var env = new ConsoleEnvironment(seed: 1);
            var dataSource = GetDataPath("generated_regression_dataset.csv");
            var reader = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);
            var data = reader.Read(dataSource);
            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, pca: r.features.ToPrincipalComponents(rank: 5)));
            var tdata = est.Fit(data).Transform(data);
            var schema = tdata.AsDynamic.Schema;

            Assert.True(schema.TryGetColumnIndex("pca", out int pca));
            var type = schema[pca].Type;
            Assert.True(type.IsVector && type.ItemType.RawKind == DataKind.R4);
            Assert.True(type.VectorSize == 5);
        }

        [Fact]
        public void UseSameColumnNamesForAppend()
        {
            var env = new ConsoleEnvironment(seed: 1, conc: 0);
            var dataView = env.CreateDataView(new[]{new {A = "A", B = "B", C = "C"}}).AssertStatic(env, c => (
                A: c.Text.Scalar,
                B: c.Text.Scalar,
                C: c.Text.Scalar));

            var dataPipeline= dataView.MakeNewEstimator().Append(row => (row.A,row.B));
            
            var transformedData = dataPipeline.Fit(dataView).Transform(dataView);
            Assert.Equal("B",transformedData.GetColumn(r => r.B).First());            
        }
    }
}