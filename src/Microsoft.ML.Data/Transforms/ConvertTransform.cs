// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(ConvertTransform.Summary, typeof(ConvertTransform), typeof(ConvertTransform.Arguments), typeof(SignatureDataTransform),
    ConvertTransform.UserName, ConvertTransform.ShortName, "ConvertTransform", DocName = "transform/ConvertTransform.md")]

[assembly: LoadableClass(ConvertTransform.Summary, typeof(ConvertTransform), null, typeof(SignatureLoadDataTransform),
    ConvertTransform.UserName, ConvertTransform.LoaderSignature, ConvertTransform.LoaderSignatureOld)]

[assembly: LoadableClass(typeof(ConvertTransform.TypeInfoCommand), typeof(ConvertTransform.TypeInfoCommand.Arguments), typeof(SignatureCommand),
    "", ConvertTransform.TypeInfoCommand.LoadName)]

[assembly: EntryPointModule(typeof(TypeConversion))]

namespace Microsoft.ML.Transforms
{
    public sealed class ConvertTransform : OneToOneTransformBase
    {
        public class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The result type", ShortName = "type")]
            public DataKind? ResultType;

            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the range of values", ShortName = "key", Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public KeyRange KeyRange;

            [Argument(ArgumentType.AtMostOnce, HelpText = "For a key column, this defines the range of values", ShortName = "key", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string Range;

            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // We accept N:T:S where N is the new column name, T is the new type,
                // and S is source column names.
                string extra;
                if (!base.TryParse(str, out extra))
                    return false;
                if (extra == null)
                    return true;

                DataKind kind;
                if (!TypeParsingUtils.TryParseDataKind(extra, out kind, out KeyRange))
                    return false;
                ResultType = kind == default(DataKind) ? default(DataKind?) : kind;
                return true;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (ResultType == null && KeyRange == null)
                    return TryUnparseCore(sb);

                if (!TrySanitize())
                    return false;
                if (CmdQuoter.NeedsQuoting(Name) || CmdQuoter.NeedsQuoting(Source))
                    return false;

                int ich = sb.Length;
                sb.Append(Name);
                sb.Append(':');
                if (ResultType != null)
                    sb.Append(ResultType.Value.GetString());
                if (KeyRange != null)
                {
                    sb.Append('[');
                    if (!KeyRange.TryUnparse(sb))
                    {
                        sb.Length = ich;
                        return false;
                    }
                    sb.Append(']');
                }
                else if (!string.IsNullOrEmpty(Range))
                    sb.Append(Range);
                sb.Append(':');
                sb.Append(Source);
                return true;
            }
        }

        public class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:type:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The result type", ShortName = "type", SortOrder = 2)]
            public DataKind? ResultType;

            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the range of values", ShortName = "key", Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public KeyRange KeyRange;

            // REVIEW: Consider supporting KeyRange type in entrypoints. This may require moving the KeyRange class to MLCore.
            [Argument(ArgumentType.AtMostOnce, HelpText = "For a key column, this defines the range of values", ShortName = "key", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string Range;
        }

        /// <summary>
        /// Extra information for each column (in addition to ColumnInfo).
        /// </summary>
        private sealed class ColInfoEx
        {
            public readonly DataKind Kind;
            // HasKeyRange indicates whether key type range information should be persisted.
            public readonly bool HasKeyRange;
            public readonly ColumnType TypeDst;
            public readonly VectorType SlotTypeDst;

            public ColInfoEx(DataKind kind, bool hasKeyRange, ColumnType type, VectorType slotType)
            {
                Contracts.AssertValue(type);
                Contracts.AssertValueOrNull(slotType);
                Contracts.Assert(slotType == null || type.ItemType.Equals(slotType.ItemType));

                Kind = kind;
                HasKeyRange = hasKeyRange;
                TypeDst = type;
                SlotTypeDst = slotType;
            }
        }

        internal const string Summary = "Converts a column to a different type, using standard conversions.";
        internal const string UserName = "Convert Transform";
        internal const string ShortName = "Convert";

        public const string LoaderSignature = "ConvertTransform";
        internal const string LoaderSignatureOld = "ConvertFunction";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CONVERTF",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Added support for keyRange
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld,
                loaderAssemblyName: typeof(ConvertTransform).Assembly.FullName);
        }

        private const string RegistrationName = "Convert";

        // This is parallel to Infos.
        private readonly ColInfoEx[] _exes;

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="resultType">The expected type of the converted column.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be converted.  If this is null '<paramref name="name"/>' will be used.</param>
        public ConvertTransform(IHostEnvironment env,
            IDataView input,
            DataKind resultType,
            string name,
            string source = null)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source ?? name, Name = name } }, ResultType = resultType }, input)
        {
        }

        public ConvertTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, env.CheckRef(args, nameof(args)).Column,
                input, null)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
            {
                DataKind kind;
                KeyRange range;
                var col = args.Column[i];
                if (col.ResultType != null)
                {
                    kind = col.ResultType.Value;
                    range = !string.IsNullOrEmpty(col.Range) ? KeyRange.Parse(col.Range) : col.KeyRange;
                }
                else if (col.KeyRange != null || !string.IsNullOrEmpty(col.Range))
                {
                    kind = Infos[i].TypeSrc.IsKey ? Infos[i].TypeSrc.RawKind : DataKind.U4;
                    range = col.KeyRange ?? KeyRange.Parse(col.Range);
                }
                else if (args.ResultType != null)
                {
                    kind = args.ResultType.Value;
                    range = !string.IsNullOrEmpty(args.Range) ? KeyRange.Parse(args.Range) : args.KeyRange;
                }
                else if (args.KeyRange != null || !string.IsNullOrEmpty(args.Range))
                {
                    kind = Infos[i].TypeSrc.IsKey ? Infos[i].TypeSrc.RawKind : DataKind.U4;
                    range = args.KeyRange ?? KeyRange.Parse(args.Range);
                }
                else
                {
                    kind = DataKind.Num;
                    range = null;
                }
                Host.CheckUserArg(Enum.IsDefined(typeof(DataKind), kind), nameof(args.ResultType));

                PrimitiveType itemType;
                if (!TryCreateEx(Host, Infos[i], kind, range, out itemType, out _exes[i]))
                {
                    throw Host.ExceptUserArg(nameof(args.Column),
                        "source column '{0}' with item type '{1}' is not compatible with destination type '{2}'",
                        input.Schema.GetColumnName(Infos[i].Source), Infos[i].TypeSrc.ItemType, itemType);
                }
            }
            SetMetadata();
        }

        private void SetMetadata()
        {
            var md = Metadata;
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                var info = Infos[iinfo];
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, info.Source, PassThrough))
                {
                    if (info.TypeSrc.IsBool && _exes[iinfo].TypeDst.ItemType.IsNumber)
                        bldr.AddPrimitive(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, true);
                }
            }
            md.Seal();
        }

        /// <summary>
        /// Returns whether metadata of the indicated kind should be passed through from the source column.
        /// </summary>
        private bool PassThrough(string kind, int iinfo)
        {
            var typeSrc = Infos[iinfo].TypeSrc;
            var typeDst = _exes[iinfo].TypeDst;
            switch (kind)
            {
                case MetadataUtils.Kinds.SlotNames:
                    Host.Assert(typeSrc.VectorSize == typeDst.VectorSize);
                    return typeDst.IsKnownSizeVector;
                case MetadataUtils.Kinds.KeyValues:
                    return typeSrc.ItemType.IsKey && typeDst.ItemType.IsKey && typeSrc.ItemType.KeyCount > 0 &&
                        typeSrc.ItemType.KeyCount == typeDst.ItemType.KeyCount;
                case MetadataUtils.Kinds.IsNormalized:
                    return typeSrc.ItemType.IsNumber && typeDst.ItemType.IsNumber;
            }
            return false;
        }

        private ConvertTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, null)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // for each added column
            //   byte: data kind, with high bit set if there is a key range
            //   if there is a key range
            //     ulong: min
            //     int: count (0 for unspecified)
            //     byte: contiguous

            Host.AssertNonEmpty(Infos);
            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
            {
                byte b = ctx.Reader.ReadByte();
                var kind = (DataKind)(b & 0x7F);
                Host.CheckDecode(Enum.IsDefined(typeof(DataKind), kind));
                KeyRange range = null;
                if ((b & 0x80) != 0)
                {
                    range = new KeyRange();
                    range.Min = ctx.Reader.ReadUInt64();
                    int count = ctx.Reader.ReadInt32();
                    if (count != 0)
                    {
                        if (count < 0 || (ulong)(count - 1) > ulong.MaxValue - range.Min)
                            throw Host.ExceptDecode();
                        range.Max = range.Min + (ulong)(count - 1);
                    }
                    range.Contiguous = ctx.Reader.ReadBoolByte();
                }

                PrimitiveType itemType;
                if (!TryCreateEx(Host, Infos[i], kind, range, out itemType, out _exes[i]))
                    throw Host.ExceptParam(nameof(input), "source column '{0}' is not of compatible type", input.Schema.GetColumnName(Infos[i].Source));
            }
            SetMetadata();
        }

        public static ConvertTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model",
                ch =>
                {
                    // *** Binary format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    ch.CheckDecode(cbFloat == sizeof(Float));
                    return new ConvertTransform(h, ctx, input);
                });
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <base>
            // for each added column
            //   byte: data kind, with high bit set if there is a key range
            //   if there is a key range
            //     ulong: min
            //     int: count (0 for unspecified)
            //     byte: contiguous
            ctx.Writer.Write(sizeof(Float));
            SaveBase(ctx);

            for (int i = 0; i < _exes.Length; i++)
            {
                var ex = _exes[i];
                Host.Assert((DataKind)(byte)ex.Kind == ex.Kind);
                if (!ex.HasKeyRange)
                    ctx.Writer.Write((byte)ex.Kind);
                else
                {
                    Host.Assert(ex.TypeDst.ItemType.IsKey);
                    var key = ex.TypeDst.ItemType.AsKey;
                    byte b = (byte)ex.Kind;
                    b |= 0x80;
                    ctx.Writer.Write(b);
                    ctx.Writer.Write(key.Min);
                    ctx.Writer.Write(key.Count);
                    ctx.Writer.WriteBoolByte(key.Contiguous);
                }
            }
        }

        public override bool CanSaveOnnx(OnnxContext ctx) => ctx.GetOnnxVersion() == OnnxVersion.Experimental;

        protected override bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColInfo info, string srcVariableName, string dstVariableName)
        {
            var opType = "CSharp";

            for (int i = 0; i < _exes.Length; i++)
            {
                var ex = _exes[i];
                var node = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
                node.AddAttribute("type", LoaderSignature);
                node.AddAttribute("to", (byte)ex.Kind);
                if (ex.HasKeyRange)
                {
                    var key = ex.TypeDst.ItemType.AsKey;
                    node.AddAttribute("min", key.Min);
                    node.AddAttribute("max", key.Count);
                    node.AddAttribute("contiguous", key.Contiguous);
                }
            }

            return true;
        }

        private static bool TryCreateEx(IExceptionContext ectx, ColInfo info, DataKind kind, KeyRange range, out PrimitiveType itemType, out ColInfoEx ex)
        {
            ectx.AssertValue(info);
            ectx.Assert(Enum.IsDefined(typeof(DataKind), kind));

            ex = null;

            var typeSrc = info.TypeSrc;
            if (range != null)
            {
                itemType = TypeParsingUtils.ConstructKeyType(kind, range);
                if (!typeSrc.ItemType.IsKey && !typeSrc.ItemType.IsText)
                    return false;
            }
            else if (!typeSrc.ItemType.IsKey)
                itemType = PrimitiveType.FromKind(kind);
            else if (!KeyType.IsValidDataKind(kind))
            {
                itemType = PrimitiveType.FromKind(kind);
                return false;
            }
            else
            {
                var key = typeSrc.ItemType.AsKey;
                ectx.Assert(KeyType.IsValidDataKind(key.RawKind));
                int count = key.Count;
                // Technically, it's an error for the counts not to match, but we'll let the Conversions
                // code return false below. There's a possibility we'll change the standard conversions to
                // map out of bounds values to zero, in which case, this is the right thing to do.
                ulong max = kind.ToMaxInt();
                if ((ulong)count > max)
                    count = (int)max;
                itemType = new KeyType(kind, key.Min, count, key.Contiguous);
            }

            // Ensure that the conversion is legal. We don't actually cache the delegate here. It will get
            // re-fetched by the utils code when needed.
            bool identity;
            Delegate del;
            if (!Conversions.Instance.TryGetStandardConversion(typeSrc.ItemType, itemType, out del, out identity))
                return false;

            ColumnType typeDst = itemType;
            if (typeSrc.IsVector)
                typeDst = new VectorType(itemType, typeSrc.AsVector);

            // An output column is transposable iff the input column was transposable.
            VectorType slotType = null;
            if (info.SlotTypeSrc != null)
                slotType = new VectorType(itemType, info.SlotTypeSrc);

            ex = new ColInfoEx(kind, range != null, typeDst, slotType);
            return true;
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            Host.Assert(_exes.Length == Infos.Length);
            return _exes[iinfo].TypeDst;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var typeSrc = Infos[iinfo].TypeSrc;
            var typeDst = _exes[iinfo].TypeDst;

            if (!typeDst.IsVector)
                return RowCursorUtils.GetGetterAs(typeDst, input, Infos[iinfo].Source);
            return RowCursorUtils.GetVecGetterAs(typeDst.AsVector.ItemType, input, Infos[iinfo].Source);
        }

        protected override VectorType GetSlotTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            return _exes[iinfo].SlotTypeDst;
        }

        protected override ISlotCursor GetSlotCursorCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.AssertValue(Infos[iinfo].SlotTypeSrc);
            Host.AssertValue(_exes[iinfo].SlotTypeDst);

            ISlotCursor cursor = InputTranspose.GetSlotCursor(Infos[iinfo].Source);
            return new SlotCursor(Host, cursor, _exes[iinfo].SlotTypeDst);
        }

        private sealed class SlotCursor : SynchronizedCursorBase<IRowCursor>, ISlotCursor
        {
            private readonly Delegate _getter;
            private readonly VectorType _type;

            public SlotCursor(IChannelProvider provider, ISlotCursor cursor, VectorType typeDst)
                : base(provider, TransposerUtils.GetRowCursorShim(provider, cursor))
            {
                Ch.Assert(Input.Schema.ColumnCount == 1);
                Ch.Assert(Input.Schema.GetColumnType(0) == cursor.GetSlotType());
                Ch.AssertValue(typeDst);
                _getter = RowCursorUtils.GetVecGetterAs(typeDst.ItemType, Input, 0);
                _type = typeDst;
            }

            public VectorType GetSlotType()
            {
                return _type;
            }

            public ValueGetter<VBuffer<TValue>> GetGetter<TValue>()
            {
                ValueGetter<VBuffer<TValue>> getter = _getter as ValueGetter<VBuffer<TValue>>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }
        }

        public sealed class TypeInfoCommand : ICommand
        {
            public const string LoadName = "TypeInfo";
            public const string Summary = "Displays information about the standard primitive " +
                "non-key types, and conversions among them.";

            public sealed class Arguments
            {
            }

            private readonly IHost _host;

            public TypeInfoCommand(IHostEnvironment env, Arguments args)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register(LoadName);
                _host.CheckValue(args, nameof(args));
            }

            private struct TypeNaInfo
            {
                public readonly bool HasNa;
                public readonly bool DefaultIsNa;

                public TypeNaInfo(bool hasNa, bool defaultIsNa)
                {
                    HasNa = hasNa;
                    DefaultIsNa = defaultIsNa;
                }
            }

            private sealed class KindSetComparer : IEqualityComparer<ISet<DataKind>>
            {
                public bool Equals(ISet<DataKind> x, ISet<DataKind> y)
                {
                    Contracts.AssertValueOrNull(x);
                    Contracts.AssertValueOrNull(y);
                    if (x == null || y == null)
                        return (x == null) && (y == null);
                    return x.SetEquals(y);
                }

                public int GetHashCode(ISet<DataKind> obj)
                {
                    Contracts.AssertValueOrNull(obj);
                    int hash = 0;
                    if (obj != null)
                    {
                        foreach (var kind in obj.OrderBy(x => x))
                            hash = Hashing.CombineHash(hash, kind.GetHashCode());
                    }
                    return hash;
                }
            }

            public void Run()
            {
                using (var ch = _host.Start("Run"))
                {
                    var conv = Conversions.Instance;
                    var comp = new KindSetComparer();
                    var dstToSrcMap = new Dictionary<HashSet<DataKind>, HashSet<DataKind>>(comp);
                    var srcToDstMap = new Dictionary<DataKind, HashSet<DataKind>>();

                    var kinds = Enum.GetValues(typeof(DataKind)).Cast<DataKind>().Distinct().OrderBy(k => k).ToArray();
                    var types = kinds.Select(kind => PrimitiveType.FromKind(kind)).ToArray();

                    HashSet<DataKind> nonIdentity = null;
                    // For each kind and its associated type.
                    for (int i = 0; i < types.Length; ++i)
                    {
                        ch.AssertValue(types[i]);
                        var info = Utils.MarshalInvoke(KindReport<int>, types[i].RawType, ch, types[i]);

                        var dstKinds = new HashSet<DataKind>();
                        Delegate del;
                        bool isIdentity;
                        for (int j = 0; j < types.Length; ++j)
                        {
                            if (conv.TryGetStandardConversion(types[i], types[j], out del, out isIdentity))
                                dstKinds.Add(types[j].RawKind);
                        }
                        if (!conv.TryGetStandardConversion(types[i], types[i], out del, out isIdentity))
                            Utils.Add(ref nonIdentity, types[i].RawKind);
                        else
                            ch.Assert(isIdentity);

                        srcToDstMap[types[i].RawKind] = dstKinds;
                        HashSet<DataKind> srcKinds;
                        if (!dstToSrcMap.TryGetValue(dstKinds, out srcKinds))
                            dstToSrcMap[dstKinds] = srcKinds = new HashSet<DataKind>();
                        srcKinds.Add(types[i].RawKind);
                    }

                    // Now perform the final outputs.
                    for (int i = 0; i < kinds.Length; ++i)
                    {
                        var dsts = srcToDstMap[kinds[i]];
                        HashSet<DataKind> srcs;
                        if (!dstToSrcMap.TryGetValue(dsts, out srcs))
                            continue;
                        ch.Assert(Utils.Size(dsts) >= 1);
                        ch.Assert(Utils.Size(srcs) >= 1);
                        string srcStrings = string.Join(", ", srcs.OrderBy(k => k).Select(k => '`' + k.GetString() + '`'));
                        string dstStrings = string.Join(", ", dsts.OrderBy(k => k).Select(k => '`' + k.GetString() + '`'));
                        dstToSrcMap.Remove(dsts);
                        ch.Info(srcStrings + " | " + dstStrings);
                    }

                    if (Utils.Size(nonIdentity) > 0)
                    {
                        ch.Warning("The following kinds did not have an identity conversion: {0}",
                            string.Join(", ", nonIdentity.OrderBy(k => k).Select(DataKindExtensions.GetString)));
                    }
                }
            }

            private TypeNaInfo KindReport<T>(IChannel ch, PrimitiveType type)
            {
                Contracts.AssertValue(ch);
                ch.AssertValue(type);
                ch.Assert(type.IsStandardScalar);

                var conv = Conversions.Instance;
                RefPredicate<T> isNaDel;
                bool hasNaPred = conv.TryGetIsNAPredicate(type, out isNaDel);
                bool defaultIsNa = false;
                if (hasNaPred)
                {
                    T def = default(T);
                    defaultIsNa = isNaDel(ref def);
                }
                return new TypeNaInfo(hasNaPred, defaultIsNa);
            }
        }
    }

    public static class TypeConversion
    {
        [TlcModule.EntryPoint(Name = "Transforms.ColumnTypeConverter", Desc = ConvertTransform.Summary, UserName = ConvertTransform.UserName, ShortName = ConvertTransform.ShortName)]
        public static CommonOutputs.TransformOutput Convert(IHostEnvironment env, ConvertTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "Convert", input);
            var view = new ConvertTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }

    public sealed class ConvertTransformer : OneToOneTransformerBase
    {
        internal const string Summary = "Converts a column to a different type, using standard conversions.";
        internal const string UserName = "Convert Transform";
        internal const string ShortName = "Convert";

        public const string LoaderSignature = "ConvertTransform";
        internal const string LoaderSignatureOld = "ConvertFunction";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CONVERTF",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Added support for keyRange
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld,
                loaderAssemblyName: typeof(ConvertTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "Convert";

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly DataKind ResultType;
            public readonly KeyRange KeyRange;

            public ColumnInfo(string input, string output, DataKind resultType, KeyRange keyRange = null)
            {

            }
        }

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckNonEmpty(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        public ConvertTransformer(IHostEnvironment env, IDataView input, ColumnInfo[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            for (int i = 0; i < columns.Length; i++)
            {
                if (!input.Schema.TryGetColumnIndex(ColumnPairs[i].input, out int srcCol))
                    throw Host.ExceptSchemaMismatch(nameof(input), "input", ColumnPairs[i].input);
                if (!CheckIsConvertable(input.Schema, srcCol, columns[i].ResultType, columns[i].KeyRange, out PrimitiveType itemType))
                {
                    throw Host.ExceptParam(nameof(input), "Source column '{0}' is not of compatible type", input.Schema.GetColumnName(srcCol));
                }
                var srcType = input.Schema.GetColumnType(srcCol);
                ColumnType typeDst = itemType;
                if (srcType.IsVector)
                    typeDst = new VectorType(itemType, srcType.AsVector);

                /*// An output column is transposable iff the input column was transposable.
                VectorType slotType = null;
                if (info.SlotTypeSrc != null)
                    slotType = new VectorType(itemType, info.SlotTypeSrc);*/
            }
        }

        private bool CheckIsConvertable(ISchema schema, int srcCol, DataKind resultType, KeyRange range, out PrimitiveType itemType)
        {
            var srcType = schema.GetColumnType(srcCol);
            if (range != null)
            {
                itemType = TypeParsingUtils.ConstructKeyType(resultType, range);
                if (!srcType.ItemType.IsKey && !srcType.ItemType.IsText)
                    return false;
            }
            else if (!srcType.ItemType.IsKey)
                itemType = PrimitiveType.FromKind(resultType);
            else if (!KeyType.IsValidDataKind(resultType))
            {
                itemType = PrimitiveType.FromKind(resultType);
                return false;
            }
            else
            {
                var key = srcType.ItemType.AsKey;
                Host.Assert(KeyType.IsValidDataKind(key.RawKind));
                int count = key.Count;
                // Technically, it's an error for the counts not to match, but we'll let the Conversions
                // code return false below. There's a possibility we'll change the standard conversions to
                // map out of bounds values to zero, in which case, this is the right thing to do.
                ulong max = resultType.ToMaxInt();
                if ((ulong)count > max)
                    count = (int)max;
                itemType = new KeyType(resultType, key.Min, count, key.Contiguous);
            }

            // Ensure that the conversion is legal. We don't actually cache the delegate here. It will get
            // re-fetched by the utils code when needed.
            return Conversions.Instance.TryGetStandardConversion(srcType.ItemType, itemType, out Delegate del, out bool identity);
        }

        public override void Save(ModelSaveContext ctx)
        {
            throw new NotImplementedException();
        }

        protected override IRowMapper MakeRowMapper(ISchema schema)
        {
            throw new NotImplementedException();
        }
    }

    public sealed class ConvertEstimator : IEstimator<ConvertTransformer>
    {
        private readonly IHost _host;
        private readonly ConvertTransformer.ColumnInfo[] _columns;

        /// <summary>
        /// Convinence constructor for simple one column case
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the output column.</param>
        /// <param name="outputColumn">Name of the column to be transformed. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="resultType">The expected type of the converted column.</param>
        public ConvertEstimator(IHostEnvironment env,
            string inputColumn, string outputColumn,
            DataKind resultType)
            : this(env, new ConvertTransformer.ColumnInfo(inputColumn, outputColumn ?? inputColumn, resultType))
        {
        }

        public ConvertEstimator(IHostEnvironment env, params ConvertTransformer.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ConvertEstimator));
            _columns = columns.ToArray();
        }

        public ConvertTransformer Fit(IDataView input) => new ConvertTransformer(_host, input, _columns);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                var metadata = new List<SchemaShape.Column>();
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, col.Kind, col.ItemType, false, col.Metadata);
            }
            return new SchemaShape(result.Values);
        }
    }

}
