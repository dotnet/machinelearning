// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TextAnalytics;

[assembly: LoadableClass(CharTokenizeTransform.Summary, typeof(CharTokenizeTransform), typeof(CharTokenizeTransform.Arguments), typeof(SignatureDataTransform),
    CharTokenizeTransform.UserName, "CharTokenize", CharTokenizeTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(CharTokenizeTransform), null, typeof(SignatureLoadDataTransform),
    CharTokenizeTransform.UserName, CharTokenizeTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.TextAnalytics
{
    /// <summary>
    /// Character-oriented tokenizer where text is considered a sequence of characters.
    /// </summary>
    public sealed class CharTokenizeTransform : OneToOneTransformBase
    {
        public sealed class Column : OneToOneColumn
        {
            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.Multiple, HelpText = "Whether to mark the beginning/end of each row/slot with start of text character (0x02)/end of text character (0x03)",
                ShortName = "mark", SortOrder = 2)]
            public bool UseMarkerChars = true;

            // REVIEW: support UTF-32 encoding through an argument option?

            // REVIEW: support encoding surrogate pairs in UTF-16?
        }

        public const string Summary = "Character-oriented tokenizer where text is considered a sequence of characters.";

        public const string LoaderSignature = "CharToken";
        public const string UserName = "Character Tokenizer Transform";

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
                loaderAssemblyName: typeof(CharTokenizeTransform).Assembly.FullName);
        }

        // Controls whether to mark the beginning/end of each row/slot with TextStartMarker/TextEndMarker.
        private readonly bool _useMarkerChars;

        // Cached type of the output columns.
        private readonly ColumnType _type;

        // Constructed and cached the first time it is needed.
        private volatile string _keyValuesStr;
        private volatile int[] _keyValuesBoundaries;

        private const ushort UnitSeparator = 0x1f;
        private const ushort TextStartMarker = 0x02;
        private const ushort TextEndMarker = 0x03;
        private const int TextMarkersCount = 2;

        // For now, this transform supports input text formatted as UTF-16 only.
        // Note: Null-char is mapped to NA. Therefore, we have UInt16.MaxValue unique key values.
        private const int CharsCount = ushort.MaxValue;

        public CharTokenizeTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, LoaderSignature, Contracts.CheckRef(args, nameof(args)).Column, input, TestIsTextItem)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _useMarkerChars = args.UseMarkerChars;

            _type = GetOutputColumnType();
            SetMetadata();
        }

        private static ColumnType GetOutputColumnType()
        {
            var keyType = new KeyType(DataKind.U2, 1, CharsCount);
            return new VectorType(keyType);
        }

        private CharTokenizeTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsTextItem)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <base>
            // byte: _useMarkerChars value.
            _useMarkerChars = ctx.Reader.ReadBoolByte();

            _isSeparatorStartEnd = ctx.Header.ModelVerReadable < 0x00010002 || ctx.Reader.ReadBoolByte();

            _type = GetOutputColumnType();
            SetMetadata();
        }

        public static CharTokenizeTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new CharTokenizeTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // byte: _useMarkerChars value.
            SaveBase(ctx);
            ctx.Writer.WriteBoolByte(_useMarkerChars);
            ctx.Writer.WriteBoolByte(_isSeparatorStartEnd);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.AssertValue(_type);
            Host.Assert(_type != null);
            return _type;
        }

        private void SetMetadata()
        {
            var md = Metadata;
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                var info = Infos[iinfo];
                // Slot names should propagate.
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, info.Source, MetadataUtils.Kinds.SlotNames))
                {
                    bldr.AddGetter<VBuffer<ReadOnlyMemory<char>>>(MetadataUtils.Kinds.KeyValues,
                        MetadataUtils.GetNamesType(_type.ItemType.KeyCount), GetKeyValues);
                }
            }
            md.Seal();
        }

        /// <summary>
        /// Get the key values (chars) corresponding to keys in the output columns.
        /// </summary>
        private void GetKeyValues(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);

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

            var values = dst.Values;
            if (Utils.Size(values) < CharsCount)
                values = new ReadOnlyMemory<char>[CharsCount];
            for (int i = 0; i < CharsCount; i++)
                values[i] = keyValuesStr.AsMemory().Slice(keyValuesBoundaries[i], keyValuesBoundaries[i + 1] - keyValuesBoundaries[i]);
            dst = new VBuffer<ReadOnlyMemory<char>>(CharsCount, values, dst.Indices);
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

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var info = Infos[iinfo];
            if (!info.TypeSrc.IsVector)
                return MakeGetterOne(input, iinfo);
            return MakeGetterVec(input, iinfo);
        }

        private ValueGetter<VBuffer<ushort>> MakeGetterOne(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(Infos[iinfo].TypeSrc.IsText);

            var getSrc = GetSrcGetter<ReadOnlyMemory<char>>(input, iinfo);
            var src = default(ReadOnlyMemory<char>);
            return
                (ref VBuffer<ushort> dst) =>
                {
                    getSrc(ref src);

                    var len = !src.IsEmpty ? (_useMarkerChars ? src.Length + TextMarkersCount : src.Length) : 0;
                    var values = dst.Values;
                    if (len > 0)
                    {
                        if (Utils.Size(values) < len)
                            values = new ushort[len];

                        int index = 0;
                        if (_useMarkerChars)
                            values[index++] = TextStartMarker;
                        var span = src.Span;
                        for (int ich = 0; ich < src.Length; ich++)
                            values[index++] = span[ich];
                        if (_useMarkerChars)
                            values[index++] = TextEndMarker;
                        Contracts.Assert(index == len);
                    }

                    dst = new VBuffer<ushort>(len, values, dst.Indices);
                };
        }

        private ValueGetter<VBuffer<ushort>> MakeGetterVec(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(Infos[iinfo].TypeSrc.ItemType.IsText);

            int cv = Infos[iinfo].TypeSrc.VectorSize;
            Contracts.Assert(cv >= 0);

            var getSrc = GetSrcGetter<VBuffer<ReadOnlyMemory<char>>>(input, iinfo);
            var src = default(VBuffer<ReadOnlyMemory<char>>);

            ValueGetter<VBuffer<ushort>> getterWithStartEndSep = (ref VBuffer<ushort> dst) =>
                {
                    getSrc(ref src);

                    int len = 0;
                    for (int i = 0; i < src.Count; i++)
                    {
                        if (!src.Values[i].IsEmpty)
                        {
                            len += src.Values[i].Length;
                            if (_useMarkerChars)
                                len += TextMarkersCount;
                        }
                    }

                    var values = dst.Values;
                    if (len > 0)
                    {
                        if (Utils.Size(values) < len)
                            values = new ushort[len];

                        int index = 0;
                        for (int i = 0; i < src.Count; i++)
                        {
                            if (src.Values[i].IsEmpty)
                                continue;
                            if (_useMarkerChars)
                                values[index++] = TextStartMarker;
                            var span = src.Values[i].Span;
                            for (int ich = 0; ich < src.Values[i].Length; ich++)
                                values[index++] = span[ich];
                            if (_useMarkerChars)
                                values[index++] = TextEndMarker;
                        }
                        Contracts.Assert(index == len);
                    }

                    dst = new VBuffer<ushort>(len, values, dst.Indices);
                };

            ValueGetter < VBuffer<ushort> > getterWithUnitSep = (ref VBuffer<ushort> dst) =>
                {
                    getSrc(ref src);

                    int len = 0;

                    for (int i = 0; i < src.Count; i++)
                    {
                        if (!src.Values[i].IsEmpty)
                        {
                            len += src.Values[i].Length;

                            if (i > 0)
                                len += 1;  // add UnitSeparator character to len that will be added
                        }
                    }

                    if (_useMarkerChars)
                        len += TextMarkersCount;

                    var values = dst.Values;
                    if (len > 0)
                    {
                        if (Utils.Size(values) < len)
                            values = new ushort[len];

                        int index = 0;

                        // ReadOnlyMemory can be a result of either concatenating text columns together
                        // or application of word tokenizer before char tokenizer in FeaturizeTextEstimator.
                        //
                        // Considering VBuffer<ReadOnlyMemory> as a single text stream.
                        // Therefore, prepend and append start and end markers only once i.e. at the start and at end of vector.
                        // Insert UnitSeparator after every piece of text in the vector.
                        if (_useMarkerChars)
                            values[index++] = TextStartMarker;

                        for (int i = 0; i < src.Count; i++)
                        {
                            if (src.Values[i].IsEmpty)
                                continue;

                            if (i > 0)
                                values[index++] = UnitSeparator;

                            var span = src.Values[i].Span;
                            for (int ich = 0; ich < src.Values[i].Length; ich++)
                                values[index++] = span[ich];
                        }

                        if (_useMarkerChars)
                            values[index++] = TextEndMarker;

                        Contracts.Assert(index == len);
                    }

                    dst = new VBuffer<ushort>(len, values, dst.Indices);
                };
            return _isSeparatorStartEnd ? getterWithStartEndSep : getterWithUnitSep;
        }
    }
}
