// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(TokenizingByCharactersTransformer.Summary, typeof(IDataTransform), typeof(TokenizingByCharactersTransformer), typeof(TokenizingByCharactersTransformer.Options), typeof(SignatureDataTransform),
    TokenizingByCharactersTransformer.UserName, "CharTokenize", TokenizingByCharactersTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IDataTransform), typeof(TokenizingByCharactersTransformer), null, typeof(SignatureLoadDataTransform),
    TokenizingByCharactersTransformer.UserName, TokenizingByCharactersTransformer.LoaderSignature)]

[assembly: LoadableClass(TokenizingByCharactersTransformer.Summary, typeof(TokenizingByCharactersTransformer), null, typeof(SignatureLoadModel),
    TokenizingByCharactersTransformer.UserName, TokenizingByCharactersTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TokenizingByCharactersTransformer), null, typeof(SignatureLoadRowMapper),
    TokenizingByCharactersTransformer.UserName, TokenizingByCharactersTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// Character-oriented tokenizer where text is considered a sequence of characters.
    /// </summary>
    public sealed class TokenizingByCharactersTransformer : OneToOneTransformerBase
    {
        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.Multiple, HelpText = "Whether to mark the beginning/end of each row/slot with start of text character (0x02)/end of text character (0x03)",
                ShortName = "mark", SortOrder = 2)]
            public bool UseMarkerChars = TokenizingByCharactersEstimator.Defaults.UseMarkerCharacters;

            // REVIEW: support UTF-32 encoding through an argument option?

            // REVIEW: support encoding surrogate pairs in UTF-16?
        }

        internal const string Summary = "Character-oriented tokenizer where text is considered a sequence of characters.";

        internal const string LoaderSignature = "CharToken";
        internal const string UserName = "Character Tokenizer Transform";

        // Keep track of the model that was saved with ver:0x00010001
        private readonly bool _isSeparatorStartEnd;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CHARTOKN",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002,  // Updated to use UnitSeparator <US> character instead of using <ETX><STX> for vector inputs.
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TokenizingByCharactersTransformer).Assembly.FullName);
        }

        // Controls whether to mark the beginning/end of each row/slot with TextStartMarker/TextEndMarker.
        private readonly bool _useMarkerChars;

        private const ushort UnitSeparator = 0x1f;
        private const ushort TextStartMarker = 0x02;
        private const ushort TextEndMarker = 0x03;
        private const int TextMarkersCount = 2;

        // For now, this transform supports input text formatted as UTF-16 only.
        // Note: Null-char is mapped to NA. Therefore, we have UInt16.MaxValue unique key values.
        internal const int CharsCount = ushort.MaxValue;
        private const string RegistrationName = "CharTokenizer";

        /// <summary>
        /// Tokenize incoming text in input columns and output the tokens as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="useMarkerCharacters">Whether to prepend a marker character, <see langword="0x02"/>, to the beginning,
        /// and append another marker character, <see langword="0x03"/>, to the end of the output vector of characters.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        internal TokenizingByCharactersTransformer(IHostEnvironment env, bool useMarkerCharacters = TokenizingByCharactersEstimator.Defaults.UseMarkerCharacters,
            params (string outputColumnName, string inputColumnName)[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
            _useMarkerChars = useMarkerCharacters;
        }

        /// <summary>
        /// The names of the output and input column pairs on which the transformation is applied.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema[srcCol].Type;
            if (!TokenizingByCharactersEstimator.IsColumnTypeValid(type))
                throw Host.ExceptParam(nameof(inputSchema), TokenizingByCharactersEstimator.ExpectedColumnType);
        }

        private TokenizingByCharactersTransformer(IHost host, ModelLoadContext ctx) :
          base(host, ctx)
        {
            // *** Binary format ***
            // <base>
            // byte: _useMarkerChars value.
            _useMarkerChars = ctx.Reader.ReadBoolByte();
            _isSeparatorStartEnd = ctx.Header.ModelVerReadable < 0x00010002 || ctx.Reader.ReadBoolByte();
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // byte: _useMarkerChars value.
            SaveColumns(ctx);
            ctx.Writer.WriteBoolByte(_useMarkerChars);
            ctx.Writer.WriteBoolByte(_isSeparatorStartEnd);
        }

        // Factory method for SignatureLoadModel.
        private static TokenizingByCharactersTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new TokenizingByCharactersTransformer(host, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new (string outputColumnName, string inputColumnName)[options.Columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = options.Columns[i];
                cols[i] = (item.Name, item.Source ?? item.Name);
            }
            return new TokenizingByCharactersTransformer(env, options.UseMarkerChars, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly DataViewType _type;
            private readonly TokenizingByCharactersTransformer _parent;
            private readonly bool[] _isSourceVector;
            // Constructed and cached the first time it is needed.
            private volatile string _keyValuesStr;
            private volatile int[] _keyValuesBoundaries;

            public Mapper(TokenizingByCharactersTransformer parent, DataViewSchema inputSchema)
             : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                var keyType = new KeyType(typeof(ushort), CharsCount);
                _type = new VectorType(keyType);
                _isSourceVector = new bool[_parent.ColumnPairs.Length];
                for (int i = 0; i < _isSourceVector.Length; i++)
                    _isSourceVector[i] = inputSchema[_parent.ColumnPairs[i].inputColumnName].Type is VectorType;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var builder = new DataViewSchema.Annotations.Builder();
                    AddMetadata(i, builder);
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _type, builder.ToAnnotations());
                }
                return result;
            }

            private void AddMetadata(int iinfo, DataViewSchema.Annotations.Builder builder)
            {
                builder.Add(InputSchema[_parent.ColumnPairs[iinfo].inputColumnName].Annotations, name => name == AnnotationUtils.Kinds.SlotNames);
                ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter =
                       (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                       {
                           GetKeyValues(iinfo, ref dst);
                       };
                builder.AddKeyValues(CharsCount, TextDataViewType.Instance, getter);
            }

            /// <summary>
            /// Get the key values (chars) corresponding to keys in the output columns.
            /// </summary>
            private void GetKeyValues(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                if (_keyValuesStr == null)
                {
                    // Create key values corresponding to the character. This will
                    // often just be the character itself, but sometimes (control characters,
                    // illegal codepoints, spaces, etc.) it is better to use something else
                    // to represent the character.
                    int[] boundaries = new int[CharsCount + 1];
                    var bldr = new StringBuilder();
                    for (int i = 1; i <= CharsCount; i++)
                    {
                        AppendCharRepr((char)i, bldr);
                        boundaries[i] = bldr.Length;
                    }

                    Host.Assert(bldr.Length == boundaries[boundaries.Length - 1]);
                    Interlocked.CompareExchange(ref _keyValuesBoundaries, boundaries, null);
                    Interlocked.CompareExchange(ref _keyValuesStr, bldr.ToString(), null);
                    bldr.Length = 0;
                }

                var keyValuesStr = _keyValuesStr;
                var keyValuesBoundaries = _keyValuesBoundaries;
                Host.AssertValue(keyValuesBoundaries);

                var editor = VBufferEditor.Create(ref dst, CharsCount);
                for (int i = 0; i < CharsCount; i++)
                    editor.Values[i] = keyValuesStr.AsMemory().Slice(keyValuesBoundaries[i], keyValuesBoundaries[i + 1] - keyValuesBoundaries[i]);
                dst = editor.Commit();
            }

            private void AppendCharRepr(char c, StringBuilder bldr)
            {
                // Special handling of characters identified in https://en.wikipedia.org/wiki/Unicode_control_characters,
                // as well as space, using the control pictures.
                if (c <= 0x20)
                {
                    // Use the control pictures unicode code block.
                    bldr.Append('<');
                    bldr.Append((char)(c + '\u2400'));
                    bldr.Append('>');
                    return;
                }
                if ('\uD800' <= c && c <= '\uDFFF')
                {
                    // These aren't real characters, and so will cause an exception
                    // when we try to write them to the file.
                    bldr.AppendFormat("\\u{0:4X}", (int)c);
                    return;
                }

                switch (c)
                {
                    case '\u007f':
                        bldr.Append("<\u2421>");
                        return; // DEL
                    case '\u0080':
                        bldr.Append("<PAD>");
                        return;
                    case '\u0081':
                        bldr.Append("<HOP>");
                        return;
                    case '\u0082':
                        bldr.Append("<BPH>");
                        return;
                    case '\u0083':
                        bldr.Append("<NBH>");
                        return;
                    case '\u0084':
                        bldr.Append("<IND>");
                        return;
                    case '\u0085':
                        bldr.Append("<NEL>");
                        return;
                    case '\u0086':
                        bldr.Append("<SSA>");
                        return;
                    case '\u0087':
                        bldr.Append("<ESA>");
                        return;
                    case '\u0088':
                        bldr.Append("<HTS>");
                        return;
                    case '\u0089':
                        bldr.Append("<HTJ>");
                        return;
                    case '\u008a':
                        bldr.Append("<VTS>");
                        return;
                    case '\u008b':
                        bldr.Append("<PLD>");
                        return;
                    case '\u008c':
                        bldr.Append("<PLU>");
                        return;
                    case '\u008d':
                        bldr.Append("<RI>");
                        return;
                    case '\u008e':
                        bldr.Append("<SS2>");
                        return;
                    case '\u008f':
                        bldr.Append("<SS3>");
                        return;
                    case '\u0090':
                        bldr.Append("<DCS>");
                        return;
                    case '\u0091':
                        bldr.Append("<PU1>");
                        return;
                    case '\u0092':
                        bldr.Append("<PU2>");
                        return;
                    case '\u0093':
                        bldr.Append("<STS>");
                        return;
                    case '\u0094':
                        bldr.Append("<CCH>");
                        return;
                    case '\u0095':
                        bldr.Append("<MW>");
                        return;
                    case '\u0096':
                        bldr.Append("<SPA>");
                        return;
                    case '\u0097':
                        bldr.Append("<EPA>");
                        return;
                    case '\u0098':
                        bldr.Append("<SOS>");
                        return;
                    case '\u0099':
                        bldr.Append("<SGCI>");
                        return;
                    case '\u009a':
                        bldr.Append("<SCI>");
                        return;
                    case '\u009b':
                        bldr.Append("<CSI>");
                        return;
                    case '\u009c':
                        bldr.Append("<ST>");
                        return;
                    case '\u009d':
                        bldr.Append("<OSC>");
                        return;
                    case '\u009e':
                        bldr.Append("<PM>");
                        return;
                    case '\u009f':
                        bldr.Append("<APC>");
                        return;
                    case '\u2028':
                        bldr.Append("<LSEP>");
                        return;
                    case '\u2029':
                        bldr.Append("<PSEP>");
                        return;
                    default:
                        bldr.Append(c);
                        return;
                }
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                if (!(input.Schema[_parent.ColumnPairs[iinfo].inputColumnName].Type is VectorType))
                    return MakeGetterOne(input, iinfo);
                return MakeGetterVec(input, iinfo);
            }

            private ValueGetter<VBuffer<ushort>> MakeGetterOne(DataViewRow input, int iinfo)
            {
                Host.AssertValue(input);
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[ColMapNewToOld[iinfo]]);
                var src = default(ReadOnlyMemory<char>);
                return
                    (ref VBuffer<ushort> dst) =>
                    {
                        getSrc(ref src);

                        var len = !src.IsEmpty ? (_parent._useMarkerChars ? src.Length + TextMarkersCount : src.Length) : 0;
                        var editor = VBufferEditor.Create(ref dst, len);
                        if (len > 0)
                        {
                            int index = 0;
                            if (_parent._useMarkerChars)
                                editor.Values[index++] = TextStartMarker;
                            var span = src.Span;
                            for (int ich = 0; ich < src.Length; ich++)
                                editor.Values[index++] = span[ich];
                            if (_parent._useMarkerChars)
                                editor.Values[index++] = TextEndMarker;
                            Contracts.Assert(index == len);
                        }

                        dst = editor.Commit();
                    };
            }

            private ValueGetter<VBuffer<ushort>> MakeGetterVec(DataViewRow input, int iinfo)
            {
                Host.AssertValue(input);

                int cv = input.Schema[ColMapNewToOld[iinfo]].Type.GetVectorSize();
                Contracts.Assert(cv >= 0);

                var getSrc = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(input.Schema[ColMapNewToOld[iinfo]]);
                var src = default(VBuffer<ReadOnlyMemory<char>>);

                ValueGetter<VBuffer<ushort>> getterWithStartEndSep = (ref VBuffer<ushort> dst) =>
                {
                    getSrc(ref src);

                    int len = 0;
                    var srcValues = src.GetValues();
                    for (int i = 0; i < srcValues.Length; i++)
                    {
                        if (!srcValues[i].IsEmpty)
                        {
                            len += srcValues[i].Length;
                            if (_parent._useMarkerChars)
                                len += TextMarkersCount;
                        }
                    }

                    var editor = VBufferEditor.Create(ref dst, len);
                    if (len > 0)
                    {
                        int index = 0;
                        for (int i = 0; i < srcValues.Length; i++)
                        {
                            if (srcValues[i].IsEmpty)
                                continue;
                            if (_parent._useMarkerChars)
                                editor.Values[index++] = TextStartMarker;
                            var span = srcValues[i].Span;
                            for (int ich = 0; ich < srcValues[i].Length; ich++)
                                editor.Values[index++] = span[ich];
                            if (_parent._useMarkerChars)
                                editor.Values[index++] = TextEndMarker;
                        }
                        Contracts.Assert(index == len);
                    }

                    dst = editor.Commit();
                };

                ValueGetter<VBuffer<ushort>> getterWithUnitSep = (ref VBuffer<ushort> dst) =>
                {
                    getSrc(ref src);

                    int len = 0;

                    var srcValues = src.GetValues();
                    for (int i = 0; i < srcValues.Length; i++)
                    {
                        if (!srcValues[i].IsEmpty)
                        {
                            len += srcValues[i].Length;

                            if (i > 0)
                                len += 1;  // add UnitSeparator character to len that will be added
                        }
                    }

                    if (_parent._useMarkerChars)
                        len += TextMarkersCount;

                    var editor = VBufferEditor.Create(ref dst, len);
                    if (len > 0)
                    {
                        int index = 0;

                        // ReadOnlyMemory can be a result of either concatenating text columns together
                        // or application of word tokenizer before char tokenizer in TextFeaturizingEstimator.
                        //
                        // Considering VBuffer<ReadOnlyMemory> as a single text stream.
                        // Therefore, prepend and append start and end markers only once i.e. at the start and at end of vector.
                        // Insert UnitSeparator after every piece of text in the vector.
                        if (_parent._useMarkerChars)
                            editor.Values[index++] = TextStartMarker;

                        for (int i = 0; i < srcValues.Length; i++)
                        {
                            if (srcValues[i].IsEmpty)
                                continue;

                            if (i > 0)
                                editor.Values[index++] = UnitSeparator;

                            var span = srcValues[i].Span;
                            for (int ich = 0; ich < srcValues[i].Length; ich++)
                                editor.Values[index++] = span[ich];
                        }

                        if (_parent._useMarkerChars)
                            editor.Values[index++] = TextEndMarker;

                        Contracts.Assert(index == len);
                    }

                    dst = editor.Commit();
                };
                return _parent._isSeparatorStartEnd ? getterWithStartEndSep : getterWithUnitSep;
            }
        }
    }

    /// <summary>
    /// Character tokenizer splits text into sequences of characters using a sliding window.
    /// </summary>
    public sealed class TokenizingByCharactersEstimator : TrivialEstimator<TokenizingByCharactersTransformer>
    {
        internal static class Defaults
        {
            public const bool UseMarkerCharacters = true;
        }

        internal static bool IsColumnTypeValid(DataViewType type) => type.GetItemType() is TextDataViewType;

        internal const string ExpectedColumnType = "Text";

        /// <summary>
        /// Tokenize incoming text in <paramref name="inputColumnName"/> and output the tokens as <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="useMarkerCharacters">Whether to prepend a marker character, <see langword="0x02"/>, to the beginning,
        /// and append another marker character, <see langword="0x03"/>, to the end of the output vector of characters.</param>
        internal TokenizingByCharactersEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
            bool useMarkerCharacters = Defaults.UseMarkerCharacters)
            : this(env, useMarkerCharacters, new[] { (outputColumnName, inputColumnName ?? outputColumnName) })
        {
        }

        /// <summary>
        /// Tokenize incoming text in input columns and output the tokens as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="useMarkerCharacters">Whether to prepend a marker character, <see langword="0x02"/>, to the beginning,
        /// and append another marker character, <see langword="0x03"/>, to the end of the output vector of characters.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>

        internal TokenizingByCharactersEstimator(IHostEnvironment env, bool useMarkerCharacters = Defaults.UseMarkerCharacters,
            params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TokenizingByCharactersEstimator)), new TokenizingByCharactersTransformer(env, useMarkerCharacters, columns))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.inputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName);
                if (!IsColumnTypeValid(col.ItemType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName, ExpectedColumnType, col.ItemType.ToString());
                var metadata = new List<SchemaShape.Column>();
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, slotMeta.ItemType, false));
                metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false));
                result[colInfo.outputColumnName] = new SchemaShape.Column(colInfo.outputColumnName, SchemaShape.Column.VectorKind.VariableVector, NumberDataViewType.UInt16, true, new SchemaShape(metadata.ToArray()));
            }

            return new SchemaShape(result.Values);
        }
    }
}
