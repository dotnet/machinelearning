// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(ConcatTransform.Summary, typeof(ConcatTransform), typeof(ConcatTransform.TaggedArguments), typeof(SignatureDataTransform),
    ConcatTransform.UserName, ConcatTransform.LoadName, "ConcatTransform", DocName = "transform/ConcatTransform.md")]

[assembly: LoadableClass(ConcatTransform.Summary, typeof(ConcatTransform), null, typeof(SignatureLoadDataTransform),
    ConcatTransform.UserName, ConcatTransform.LoaderSignature, ConcatTransform.LoaderSignatureOld)]

namespace Microsoft.ML.Runtime.Data
{
    using T = PfaUtils.Type;

    public sealed class ConcatTransform : RowToRowMapperTransformBase, ITransformCanSavePfa, ITransformCanSaveOnnx
    {
        public sealed class Column : ManyToOneColumn
        {
            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);
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

        public sealed class TaggedColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the new column", ShortName = "name")]
            public string Name;

            // The tag here (the key of the KeyValuePair) is the string that will be the prefix of the slot name
            // in the output column. For non-vector columns, the slot name will be either the column name or the 
            // tag if it is non empty. For vector columns, the slot names will be 'ColumnName.SlotName' if the
            // tag is empty, 'Tag.SlotName' if tag is non empty, and simply the slot name if tag is non empty
            // and equal to the column name.
            [Argument(ArgumentType.Multiple, HelpText = "Name of the source column", ShortName = "src")]
            public KeyValuePair<string, string>[] Source;

            public static TaggedColumn Parse(string str)
            {
                Contracts.AssertNonEmpty(str);
                // REVIEW: Support a short form for aliases.
                var res = Column.Parse(str);
                if (res == null)
                    return null;
                Contracts.AssertValue(res.Source);
                var taggedColumn = new TaggedColumn();
                taggedColumn.Name = res.Name;
                taggedColumn.Source = res.Source.Select(s => new KeyValuePair<string, string>(null, s)).ToArray();
                return taggedColumn;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Source == null || Source.Any(kvp => !string.IsNullOrEmpty(kvp.Key)))
                    return false;
                var column = new Column();
                column.Name = Name;
                column.Source = Source.Select(kvp => kvp.Value).ToArray();
                return column.TryUnparse(sb);
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            public Arguments()
            {
            }

            public Arguments(string name, params string[] source)
            {
                Column = new[] { new Column()
                {
                    Name = name,
                    Source = source
                }};
            }

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:srcs)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        public sealed class TaggedArguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:srcs)", ShortName = "col", SortOrder = 1)]
            public TaggedColumn[] Column;
        }

        private sealed class Bindings : ManyToOneColumnBindingsBase
        {
            public readonly bool[] EchoSrc;

            private readonly ColumnType[] _types;
            private readonly ColumnType[] _typesSlotNames;
            private readonly ColumnType[] _typesCategoricals;
            private readonly bool[] _isNormalized;
            private readonly string[][] _aliases;

            private readonly MetadataUtils.MetadataGetter<VBuffer<DvText>> _getSlotNames;

            public Bindings(Column[] columns, TaggedColumn[] taggedColumns, ISchema schemaInput)
                : base(columns, schemaInput, TestTypes)
            {
                Contracts.Assert(taggedColumns == null || columns.Length == taggedColumns.Length);
                _aliases = new string[columns.Length][];
                for (int i = 0; i < columns.Length; i++)
                {
                    _aliases[i] = new string[columns[i].Source.Length];
                    if (taggedColumns != null)
                    {
                        var column = taggedColumns[i];
                        Contracts.Assert(columns[i].Name == column.Name);
                        Contracts.AssertValue(columns[i].Source);
                        Contracts.AssertValue(column.Source);
                        Contracts.Assert(columns[i].Source.Length == column.Source.Length);
                        for (int j = 0; j < column.Source.Length; j++)
                        {
                            var kvp = column.Source[j];
                            Contracts.Assert(columns[i].Source[j] == kvp.Value);
                            if (!string.IsNullOrEmpty(kvp.Key))
                                _aliases[i][j] = kvp.Key;
                        }
                    }
                }

                CacheTypes(out _types, out _typesSlotNames, out EchoSrc, out _isNormalized, out _typesCategoricals);
                _getSlotNames = GetSlotNames;
            }

            public Bindings(ModelLoadContext ctx, ISchema schemaInput)
                : base(ctx, schemaInput, TestTypes)
            {
                // *** Binary format ***
                // (base fields)
                // if version >= VersionAddedAliases
                //   foreach column:
                //      foreach non-null alias
                //          int: index of the alias
                //          int: string id of the alias
                //      int: -1, marks the end of the list
                _aliases = new string[Infos.Length][];
                for (int i = 0; i < Infos.Length; i++)
                {
                    var length = Infos[i].SrcIndices.Length;
                    _aliases[i] = new string[length];
                    if (ctx.Header.ModelVerReadable >= VersionAddedAliases)
                    {
                        for (; ; )
                        {
                            var j = ctx.Reader.ReadInt32();
                            if (j == -1)
                                break;
                            Contracts.CheckDecode(0 <= j && j < length);
                            Contracts.CheckDecode(_aliases[i][j] == null);
                            _aliases[i][j] = ctx.LoadNonEmptyString();
                        }
                    }
                }

                CacheTypes(out _types, out _typesSlotNames, out EchoSrc, out _isNormalized, out _typesCategoricals);
                _getSlotNames = GetSlotNames;
            }

            public override void Save(ModelSaveContext ctx)
            {
                // *** Binary format ***
                // (base fields)
                // if version >= VersionAddedAliases
                //   foreach column:
                //      foreach non-null alias
                //          int: index of the alias
                //          int: string id of the alias
                //      int: -1, marks the end of the list
                base.Save(ctx);
                Contracts.Assert(_aliases.Length == Infos.Length);
                for (int i = 0; i < Infos.Length; i++)
                {
                    Contracts.Assert(_aliases[i].Length == Infos[i].SrcIndices.Length);
                    for (int j = 0; j < _aliases[i].Length; j++)
                    {
                        if (!string.IsNullOrEmpty(_aliases[i][j]))
                        {
                            ctx.Writer.Write(j);
                            ctx.SaveNonEmptyString(_aliases[i][j]);
                        }
                    }
                    ctx.Writer.Write(-1);
                }
            }

            private static string TestTypes(ColumnType[] types)
            {
                Contracts.AssertNonEmpty(types);
                var type = types[0].ItemType;
                if (!type.IsPrimitive)
                    return "Expected primitive type";
                if (!types.All(t => type.Equals(t.ItemType)))
                    return "All source columns must have the same type";

                return null;
            }

            private void CacheTypes(out ColumnType[] types, out ColumnType[] typesSlotNames, out bool[] echoSrc,
                out bool[] isNormalized, out ColumnType[] typesCategoricals)
            {
                Contracts.AssertNonEmpty(Infos);
                echoSrc = new bool[Infos.Length];
                isNormalized = new bool[Infos.Length];
                types = new ColumnType[Infos.Length];
                typesSlotNames = new ColumnType[Infos.Length];
                typesCategoricals = new ColumnType[Infos.Length];

                for (int i = 0; i < Infos.Length; i++)
                {
                    var info = Infos[i];
                    // REVIEW: Add support for implicit conversions?
                    if (info.SrcTypes.Length == 1 && info.SrcTypes[0].IsVector)
                    {
                        // All meta-data is passed through in this case, so don't need the slot names type.
                        echoSrc[i] = true;
                        DvBool b = DvBool.False;
                        isNormalized[i] =
                            info.SrcTypes[0].ItemType.IsNumber &&
                            Input.TryGetMetadata(BoolType.Instance, MetadataUtils.Kinds.IsNormalized, info.SrcIndices[0], ref b) &&
                            b.IsTrue;
                        types[i] = info.SrcTypes[0];
                        continue;
                    }

                    // The single scalar and multiple vector case.
                    isNormalized[i] = info.SrcTypes[0].ItemType.IsNumber;
                    if (isNormalized[i])
                    {
                        foreach (var srcCol in info.SrcIndices)
                        {
                            DvBool b = DvBool.False;
                            if (!Input.TryGetMetadata(BoolType.Instance, MetadataUtils.Kinds.IsNormalized, srcCol, ref b) ||
                                !b.IsTrue)
                            {
                                isNormalized[i] = false;
                                break;
                            }
                        }
                    }

                    types[i] = new VectorType(info.SrcTypes[0].ItemType.AsPrimitive, info.SrcSize);
                    if (info.SrcSize == 0)
                        continue;

                    bool hasCategoricals = false;
                    int catCount = 0;
                    for (int j = 0; j < info.SrcTypes.Length; j++)
                    {
                        if (info.SrcTypes[j].ValueCount == 0)
                        {
                            hasCategoricals = false;
                            break;
                        }

                        if (MetadataUtils.TryGetCategoricalFeatureIndices(Input, info.SrcIndices[j], out int[] typeCat))
                        {
                            Contracts.Assert(typeCat.Length > 0);
                            catCount += typeCat.Length;
                            hasCategoricals = true;
                        }
                    }

                    if (hasCategoricals)
                    {
                        Contracts.Assert(catCount % 2 == 0);
                        typesCategoricals[i] = MetadataUtils.GetCategoricalType(catCount / 2);
                    }

                    bool hasSlotNames = false;
                    for (int j = 0; j < info.SrcTypes.Length; j++)
                    {
                        var type = info.SrcTypes[j];
                        // For non-vector source column, we use the column name as the slot name.
                        if (!type.IsVector)
                        {
                            hasSlotNames = true;
                            break;
                        }
                        // The vector has known length since the result length is known.
                        Contracts.Assert(type.IsKnownSizeVector);
                        var typeNames = Input.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, info.SrcIndices[j]);
                        if (typeNames != null && typeNames.VectorSize == type.VectorSize && typeNames.ItemType.IsText)
                        {
                            hasSlotNames = true;
                            break;
                        }
                    }

                    if (hasSlotNames)
                        typesSlotNames[i] = MetadataUtils.GetNamesType(info.SrcSize);
                }
            }

            protected override ColumnType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < Infos.Length);

                Contracts.Assert(_types[iinfo] != null);
                return _types[iinfo];
            }

            protected override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypesCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);

                if (EchoSrc[iinfo])
                {
                    // All meta-data stuff is passed through.
                    Contracts.Assert(Infos[iinfo].SrcIndices.Length == 1);
                    return Input.GetMetadataTypes(Infos[iinfo].SrcIndices[0]);
                }

                var items = base.GetMetadataTypesCore(iinfo);

                var typeNames = _typesSlotNames[iinfo];
                if (typeNames != null)
                    items = items.Prepend(typeNames.GetPair(MetadataUtils.Kinds.SlotNames));

                var typeCategoricals = _typesCategoricals[iinfo];
                if (typeCategoricals != null)
                    items = items.Prepend(typeCategoricals.GetPair(MetadataUtils.Kinds.CategoricalSlotRanges));

                if (_isNormalized[iinfo])
                    items = items.Prepend(BoolType.Instance.GetPair(MetadataUtils.Kinds.IsNormalized));

                return items;
            }

            protected override ColumnType GetMetadataTypeCore(string kind, int iinfo)
            {
                Contracts.AssertNonEmpty(kind);
                Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);

                if (EchoSrc[iinfo])
                {
                    // All meta-data stuff is passed through.
                    Contracts.Assert(Infos[iinfo].SrcIndices.Length == 1);
                    return Input.GetMetadataTypeOrNull(kind, Infos[iinfo].SrcIndices[0]);
                }

                switch (kind)
                {
                    case MetadataUtils.Kinds.SlotNames:
                        return _typesSlotNames[iinfo];
                    case MetadataUtils.Kinds.CategoricalSlotRanges:
                        return _typesCategoricals[iinfo];
                    case MetadataUtils.Kinds.IsNormalized:
                        if (_isNormalized[iinfo])
                            return BoolType.Instance;
                        return null;
                    default:
                        return base.GetMetadataTypeCore(kind, iinfo);
                }
            }

            protected override void GetMetadataCore<TValue>(string kind, int iinfo, ref TValue value)
            {
                Contracts.AssertNonEmpty(kind);
                Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);

                if (EchoSrc[iinfo])
                {
                    // All meta-data stuff is passed through.
                    Contracts.Assert(Infos[iinfo].SrcIndices.Length == 1);
                    Input.GetMetadata(kind, Infos[iinfo].SrcIndices[0], ref value);
                    return;
                }

                switch (kind)
                {
                    case MetadataUtils.Kinds.SlotNames:
                        if (_typesSlotNames[iinfo] == null)
                            throw MetadataUtils.ExceptGetMetadata();
                        _getSlotNames.Marshal(iinfo, ref value);
                        break;
                    case MetadataUtils.Kinds.CategoricalSlotRanges:
                        if (_typesCategoricals[iinfo] == null)
                            throw MetadataUtils.ExceptGetMetadata();

                        MetadataUtils.Marshal<VBuffer<DvInt4>, TValue>(GetCategoricalSlotRanges, iinfo, ref value);
                        break;
                    case MetadataUtils.Kinds.IsNormalized:
                        if (!_isNormalized[iinfo])
                            throw MetadataUtils.ExceptGetMetadata();
                        MetadataUtils.Marshal<DvBool, TValue>(IsNormalized, iinfo, ref value);
                        break;
                    default:
                        base.GetMetadataCore(kind, iinfo, ref value);
                        break;
                }
            }

            private void GetCategoricalSlotRanges(int iiinfo, ref VBuffer<DvInt4> dst)
            {
                List<DvInt4> allValues = new List<DvInt4>();
                int slotCount = 0;
                for (int i = 0; i < Infos[iiinfo].SrcIndices.Length; i++)
                {

                    Contracts.Assert(Infos[iiinfo].SrcTypes[i].ValueCount > 0);

                    if (i > 0)
                        slotCount += Infos[iiinfo].SrcTypes[i - 1].ValueCount;

                    if (MetadataUtils.TryGetCategoricalFeatureIndices(Input, Infos[iiinfo].SrcIndices[i], out int[] values))
                    {
                        Contracts.Assert(values.Length > 0 && values.Length % 2 == 0);

                        for (int j = 0; j < values.Length; j++)
                            allValues.Add(values[j] + slotCount);
                    }
                }

                Contracts.Assert(allValues.Count > 0);

                dst = new VBuffer<DvInt4>(allValues.Count, allValues.ToArray());
            }

            private void IsNormalized(int iinfo, ref DvBool dst)
            {
                dst = DvBool.True;
            }

            private void GetSlotNames(int iinfo, ref VBuffer<DvText> dst)
            {
                Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);
                Contracts.Assert(!EchoSrc[iinfo]);
                Contracts.Assert(_types[iinfo].VectorSize > 0);

                var type = _typesSlotNames[iinfo];
                Contracts.AssertValue(type);
                Contracts.Assert(type.VectorSize == _types[iinfo].VectorSize);

                var bldr = BufferBuilder<DvText>.CreateDefault();
                bldr.Reset(type.VectorSize, dense: false);

                var sb = new StringBuilder();
                var names = default(VBuffer<DvText>);
                var info = Infos[iinfo];
                var aliases = _aliases[iinfo];
                int slot = 0;
                for (int i = 0; i < info.SrcTypes.Length; i++)
                {
                    int colSrc = info.SrcIndices[i];
                    var typeSrc = info.SrcTypes[i];
                    Contracts.Assert(aliases[i] != "");
                    var colName = Input.GetColumnName(colSrc);
                    var nameSrc = aliases[i] ?? colName;
                    if (!typeSrc.IsVector)
                    {
                        bldr.AddFeature(slot++, new DvText(nameSrc));
                        continue;
                    }

                    Contracts.Assert(typeSrc.IsKnownSizeVector);
                    var typeNames = Input.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, colSrc);
                    if (typeNames != null && typeNames.VectorSize == typeSrc.VectorSize && typeNames.ItemType.IsText)
                    {
                        Input.GetMetadata(MetadataUtils.Kinds.SlotNames, colSrc, ref names);
                        sb.Clear();
                        if (aliases[i] != colName)
                            sb.Append(nameSrc).Append(".");
                        int len = sb.Length;
                        foreach (var kvp in names.Items())
                        {
                            if (!kvp.Value.HasChars)
                                continue;
                            sb.Length = len;
                            kvp.Value.AddToStringBuilder(sb);
                            bldr.AddFeature(slot + kvp.Key, new DvText(sb.ToString()));
                        }
                    }
                    slot += info.SrcTypes[i].VectorSize;
                }
                Contracts.Assert(slot == _types[iinfo].VectorSize);

                bldr.GetResult(ref dst);
            }
        }

        public const string Summary = "Concatenates two columns of the same item type.";
        public const string UserName = "Concat Transform";
        public const string LoadName = "Concat";

        internal const string LoaderSignature = "ConcatTransform";
        internal const string LoaderSignatureOld = "ConcatFunction";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CONCAT F",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Added aliases
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld);
        }

        private const int VersionAddedAliases = 0x00010002;

        private readonly Bindings _bindings;

        private const string RegistrationName = "Concat";

        public bool CanSavePfa => true;

        public bool CanSaveOnnx => true;

        public override ISchema Schema => _bindings;

        public ConcatTransform(IHostEnvironment env, IDataView input, string name, params string[] source)
            : this(env, new Arguments(name, source), input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public ConcatTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));
            for (int i = 0; i < args.Column.Length; i++)
                Host.CheckUserArg(Utils.Size(args.Column[i].Source) > 0, nameof(args.Column));

            _bindings = new Bindings(args.Column, null, Source.Schema);
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public ConcatTransform(IHostEnvironment env, TaggedArguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));
            for (int i = 0; i < args.Column.Length; i++)
                Host.CheckUserArg(Utils.Size(args.Column[i].Source) > 0, nameof(args.Column));

            var columns = args.Column
                .Select(c => new Column() { Name = c.Name, Source = c.Source.Select(kvp => kvp.Value).ToArray() })
                .ToArray();
            _bindings = new Bindings(columns, args.Column, Source.Schema);
        }

        private ConcatTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            int cbFloat = ctx.Reader.ReadInt32();
            Host.CheckDecode(cbFloat == sizeof(Float));
            _bindings = new Bindings(ctx, Source.Schema);
        }

        public static ConcatTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoadName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ConcatTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            ctx.Writer.Write(sizeof(Float));
            _bindings.Save(ctx);
        }

        private KeyValuePair<string, JToken> SavePfaInfoCore(BoundPfaContext ctx, int iinfo)
        {
            Host.AssertValue(ctx);
            Host.Assert(0 <= iinfo && iinfo < _bindings.InfoCount);

            var info = _bindings.Infos[iinfo];
            int outIndex = _bindings.MapIinfoToCol(iinfo);
            string outName = _bindings.GetColumnName(outIndex);
            if (info.SrcSize == 0) // Do not attempt variable length.
                return new KeyValuePair<string, JToken>(outName, null);

            string[] srcTokens = new string[info.SrcIndices.Length];
            bool[] srcPrimitive = new bool[info.SrcIndices.Length];
            for (int i = 0; i < info.SrcIndices.Length; ++i)
            {
                int srcIndex = info.SrcIndices[i];
                var srcName = Source.Schema.GetColumnName(srcIndex);
                if ((srcTokens[i] = ctx.TokenOrNullForName(srcName)) == null)
                    return new KeyValuePair<string, JToken>(outName, null);
                srcPrimitive[i] = info.SrcTypes[i].IsPrimitive;
            }
            Host.Assert(srcTokens.All(tok => tok != null));
            var itemColumnType = _bindings.GetColumnType(outIndex).ItemType;
            var itemType = T.PfaTypeOrNullForColumnType(itemColumnType);
            if (itemType == null)
                return new KeyValuePair<string, JToken>(outName, null);
            JObject jobj = null;
            var arrType = T.Array(itemType);

            // The "root" object will be the concatenation of all the initial scalar objects into an
            // array, or else, if the first object is not scalar, just that first object.
            JToken result;
            int min;
            if (srcPrimitive[0])
            {
                JArray rootObjects = new JArray();
                for (int i = 0; i < srcTokens.Length && srcPrimitive[i]; ++i)
                    rootObjects.Add(srcTokens[i]);
                result = jobj.AddReturn("type", arrType).AddReturn("new", new JArray(rootObjects));
                min = rootObjects.Count;
            }
            else
            {
                result = srcTokens[0];
                min = 1;
            }

            for (int i = min; i < srcTokens.Length; ++i)
                result = PfaUtils.Call(srcPrimitive[i] ? "a.append" : "a.concat", result, srcTokens[i]);

            Host.AssertValue(result);
            return new KeyValuePair<string, JToken>(outName, result);
        }

        public void SaveAsPfa(BoundPfaContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            var toHide = new List<string>();
            var toDeclare = new List<KeyValuePair<string, JToken>>();

            for (int iinfo = 0; iinfo < _bindings.InfoCount; ++iinfo)
            {
                var toSave = SavePfaInfoCore(ctx, iinfo);
                if (toSave.Value == null)
                    toHide.Add(toSave.Key);
                else
                    toDeclare.Add(toSave);
            }
            ctx.Hide(toHide.ToArray());
            ctx.DeclareVar(toDeclare.ToArray());
        }

        public void SaveAsOnnx(OnnxContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(CanSaveOnnx);

            string opType = "FeatureVectorizer";
            for (int iinfo = 0; iinfo < _bindings.InfoCount; ++iinfo)
            {
                var info = _bindings.Infos[iinfo];
                int outIndex = _bindings.MapIinfoToCol(iinfo);
                string outName = _bindings.GetColumnName(outIndex);
                var outColType = _bindings.GetColumnType(outIndex);
                if (info.SrcSize == 0)
                {
                    ctx.RemoveColumn(outName, false);
                    continue;
                }

                List<KeyValuePair<string, long>> inputList = new List<KeyValuePair<string, long>>();
                for (int i = 0; i < info.SrcIndices.Length; ++i)
                {
                    int srcIndex = info.SrcIndices[i];
                    var srcName = Source.Schema.GetColumnName(srcIndex);
                    if (!ctx.ContainsColumn(srcName))
                    {
                        ctx.RemoveColumn(outName, false);
                        return;
                    }

                    inputList.Add(new KeyValuePair<string, long>(ctx.GetVariableName(srcName),
                        Source.Schema.GetColumnType(srcIndex).ValueCount));
                }

                var node = OnnxUtils.MakeNode(opType, new List<string>(inputList.Select(t => t.Key)),
                    new List<string> { ctx.AddIntermediateVariable(outColType, outName) }, ctx.GetNodeName(opType));

                ctx.AddNode(node);

                OnnxUtils.NodeAddAttributes(node, "inputList", inputList.Select(x => x.Key));
                OnnxUtils.NodeAddAttributes(node, "inputdimensions", inputList.Select(x => x.Value));
            }
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");

            // Prefer parallel cursors iff some of our columns are active, otherwise, don't care.
            if (_bindings.AnyNewColumnsActive(predicate))
                return true;
            return null;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var input = Source.GetRowCursor(inputPred, rand);
            return new RowCursor(Host, this, input, active);
        }

        public sealed override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n, rand);
            Host.AssertNonEmpty(inputs);

            if (inputs.Length == 1 && n > 1 && _bindings.AnyNewColumnsActive(predicate))
                inputs = DataViewUtils.CreateSplitCursors(out consolidator, Host, inputs[0], n);
            Host.AssertNonEmpty(inputs);

            var cursors = new IRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new RowCursor(Host, this, inputs[i], active);
            return cursors;
        }

        protected override int MapColumnIndex(out bool isSrc, int col)
        {
            return _bindings.MapColumnIndex(out isSrc, col);
        }

        protected override Func<int, bool> GetDependenciesCore(Func<int, bool> predicate)
        {
            return _bindings.GetDependencies(predicate);
        }

        protected override Delegate[] CreateGetters(IRow input, Func<int, bool> active, out Action disposer)
        {
            Func<int, bool> activeInfos =
                iinfo =>
                {
                    int col = _bindings.MapIinfoToCol(iinfo);
                    return active(col);
                };

            var getters = new Delegate[_bindings.InfoCount];
            disposer = null;
            using (var ch = Host.Start("CreateGetters"))
            {
                for (int iinfo = 0; iinfo < _bindings.InfoCount; iinfo++)
                {
                    if (!activeInfos(iinfo))
                        continue;
                    getters[iinfo] = MakeGetter(ch, input, iinfo);
                }
                ch.Done();
                return getters;
            }
        }

        private ValueGetter<T> GetSrcGetter<T>(IRow input, int iinfo, int isrc)
        {
            return input.GetGetter<T>(_bindings.Infos[iinfo].SrcIndices[isrc]);
        }

        private Delegate MakeGetter(IChannel ch, IRow input, int iinfo)
        {
            var info = _bindings.Infos[iinfo];
            MethodInfo meth;
            if (_bindings.EchoSrc[iinfo])
            {
                Func<IRow, int, int, ValueGetter<int>> srcDel = GetSrcGetter<int>;
                meth = srcDel.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(info.SrcTypes[0].RawType);
                return (Delegate)meth.Invoke(this, new object[] { input, iinfo, 0 });
            }

            Func<IChannel, IRow, int, ValueGetter<VBuffer<int>>> del = MakeGetter<int>;
            meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(info.SrcTypes[0].ItemType.RawType);
            return (Delegate)meth.Invoke(this, new object[] { ch, input, iinfo });
        }

        private ValueGetter<VBuffer<T>> MakeGetter<T>(IChannel ch, IRow input, int iinfo)
        {
            var info = _bindings.Infos[iinfo];
            var srcGetterOnes = new ValueGetter<T>[info.SrcIndices.Length];
            var srcGetterVecs = new ValueGetter<VBuffer<T>>[info.SrcIndices.Length];
            for (int j = 0; j < info.SrcIndices.Length; j++)
            {
                if (info.SrcTypes[j].IsVector)
                    srcGetterVecs[j] = GetSrcGetter<VBuffer<T>>(input, iinfo, j);
                else
                    srcGetterOnes[j] = GetSrcGetter<T>(input, iinfo, j);
            }

            T tmp = default(T);
            VBuffer<T>[] tmpBufs = new VBuffer<T>[info.SrcIndices.Length];
            return
                (ref VBuffer<T> dst) =>
                {
                    int dstLength = 0;
                    int dstCount = 0;
                    for (int i = 0; i < info.SrcIndices.Length; i++)
                    {
                        var type = info.SrcTypes[i];
                        if (type.IsVector)
                        {
                            srcGetterVecs[i](ref tmpBufs[i]);
                            if (type.VectorSize != 0 && type.VectorSize != tmpBufs[i].Length)
                            {
                                throw ch.Except("Column '{0}': expected {1} slots, but got {2}",
                                    input.Schema.GetColumnName(info.SrcIndices[i]), type.VectorSize, tmpBufs[i].Length)
                                    .MarkSensitive(MessageSensitivity.Schema);
                            }
                            dstLength = checked(dstLength + tmpBufs[i].Length);
                            dstCount = checked(dstCount + tmpBufs[i].Count);
                        }
                        else
                        {
                            dstLength = checked(dstLength + 1);
                            dstCount = checked(dstCount + 1);
                        }
                    }

                    var values = dst.Values;
                    var indices = dst.Indices;
                    if (dstCount <= dstLength / 2)
                    {
                        // Concatenate into a sparse representation.
                        if (Utils.Size(values) < dstCount)
                            values = new T[dstCount];
                        if (Utils.Size(indices) < dstCount)
                            indices = new int[dstCount];

                        int offset = 0;
                        int count = 0;
                        for (int j = 0; j < info.SrcIndices.Length; j++)
                        {
                            ch.Assert(offset < dstLength);
                            if (info.SrcTypes[j].IsVector)
                            {
                                var buffer = tmpBufs[j];
                                ch.Assert(buffer.Count <= dstCount - count);
                                ch.Assert(buffer.Length <= dstLength - offset);
                                if (buffer.IsDense)
                                {
                                    for (int i = 0; i < buffer.Length; i++)
                                    {
                                        values[count] = buffer.Values[i];
                                        indices[count++] = offset + i;
                                    }
                                }
                                else
                                {
                                    for (int i = 0; i < buffer.Count; i++)
                                    {
                                        values[count] = buffer.Values[i];
                                        indices[count++] = offset + buffer.Indices[i];
                                    }
                                }
                                offset += buffer.Length;
                            }
                            else
                            {
                                ch.Assert(count < dstCount);
                                srcGetterOnes[j](ref tmp);
                                values[count] = tmp;
                                indices[count++] = offset;
                                offset++;
                            }
                        }
                        ch.Assert(count <= dstCount);
                        ch.Assert(offset == dstLength);
                        dst = new VBuffer<T>(dstLength, count, values, indices);
                    }
                    else
                    {
                        // Concatenate into a dense representation.
                        if (Utils.Size(values) < dstLength)
                            values = new T[dstLength];

                        int offset = 0;
                        for (int j = 0; j < info.SrcIndices.Length; j++)
                        {
                            ch.Assert(tmpBufs[j].Length <= dstLength - offset);
                            if (info.SrcTypes[j].IsVector)
                            {
                                tmpBufs[j].CopyTo(values, offset);
                                offset += tmpBufs[j].Length;
                            }
                            else
                            {
                                srcGetterOnes[j](ref tmp);
                                values[offset++] = tmp;
                            }
                        }
                        ch.Assert(offset == dstLength);
                        dst = new VBuffer<T>(dstLength, values, indices);
                    }
                };
        }

        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active;
            private readonly Delegate[] _getters;

            public RowCursor(IChannelProvider provider, ConcatTransform parent, IRowCursor input, bool[] active)
                : base(provider, input)
            {
                Ch.AssertValue(parent);
                Ch.Assert(active == null || active.Length == parent._bindings.ColumnCount);

                _bindings = parent._bindings;
                _active = active;

                _getters = new Delegate[_bindings.Infos.Length];
                for (int i = 0; i < _bindings.Infos.Length; i++)
                {
                    if (IsIndexActive(i))
                        _getters[i] = parent.MakeGetter(Ch, Input, i);
                }
            }

            public ISchema Schema { get { return _bindings; } }

            private bool IsIndexActive(int iinfo)
            {
                Ch.Assert(0 <= iinfo & iinfo < _bindings.Infos.Length);
                return _active == null || _active[_bindings.MapIinfoToCol(iinfo)];
            }

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.ColumnCount);
                return _active == null || _active[col];
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));

                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return Input.GetGetter<TValue>(index);

                Ch.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }
        }
    }
}
