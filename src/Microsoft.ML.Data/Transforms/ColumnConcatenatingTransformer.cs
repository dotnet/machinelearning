// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(ColumnConcatenatingTransformer.Summary, typeof(IDataTransform), typeof(ColumnConcatenatingTransformer), typeof(ColumnConcatenatingTransformer.TaggedOptions), typeof(SignatureDataTransform),
    ColumnConcatenatingTransformer.UserName, ColumnConcatenatingTransformer.LoadName, "ConcatTransform", DocName = "transform/ConcatTransform.md")]

[assembly: LoadableClass(ColumnConcatenatingTransformer.Summary, typeof(IDataTransform), typeof(ColumnConcatenatingTransformer), null, typeof(SignatureLoadDataTransform),
    ColumnConcatenatingTransformer.UserName, ColumnConcatenatingTransformer.LoaderSignature, ColumnConcatenatingTransformer.LoaderSignatureOld)]

[assembly: LoadableClass(typeof(ColumnConcatenatingTransformer), null, typeof(SignatureLoadModel),
    ColumnConcatenatingTransformer.UserName, ColumnConcatenatingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ColumnConcatenatingTransformer), null, typeof(SignatureLoadRowMapper),
    ColumnConcatenatingTransformer.UserName, ColumnConcatenatingTransformer.LoaderSignature)]

namespace Microsoft.ML.Data
{
    using PfaType = PfaUtils.Type;

    /// <summary>
    /// Concatenates columns in an <see cref="IDataView"/> into one single column. Please see <see cref="ColumnConcatenatingEstimator"/> for
    /// constructing <see cref="ColumnConcatenatingTransformer"/>.
    /// </summary>
    public sealed class ColumnConcatenatingTransformer : RowToRowTransformerBase
    {
        internal const string Summary = "Concatenates one or more columns of the same item type.";
        internal const string UserName = "Concat Transform";
        internal const string LoadName = "Concat";

        internal const string LoaderSignature = "ConcatTransform";
        internal const string LoaderSignatureOld = "ConcatFunction";

        internal sealed class Column : ManyToOneColumn
        {
            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);
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

        [BestFriend]
        internal sealed class TaggedColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the new column", ShortName = "name")]
            public string Name;

            // The tag here (the key of the KeyValuePair) is the string that will be the prefix of the slot name
            // in the output column. For non-vector columns, the slot name will be either the column name or the
            // tag if it is non empty. For vector columns, the slot names will be 'ColumnName.SlotName' if the
            // tag is empty, 'Tag.SlotName' if tag is non empty, and simply the slot name if tag is non empty
            // and equal to the column name.
            [Argument(ArgumentType.Multiple, HelpText = "Names of the source columns", ShortName = "src")]
            public KeyValuePair<string, string>[] Source;

            internal static TaggedColumn Parse(string str)
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

            internal bool TryUnparse(StringBuilder sb)
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

        internal sealed class Options : TransformInputBase
        {
            public Options()
            {
            }

            public Options(string name, params string[] source)
            {
                Columns = new[] { new Column()
                {
                    Name = name,
                    Source = source
                }};
            }

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:srcs)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        [BestFriend]
        internal sealed class TaggedOptions
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:srcs)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public TaggedColumn[] Columns;
        }

        [BestFriend]
        internal sealed class ColumnOptions
        {
            public readonly string Name;
            private readonly (string name, string alias)[] _sources;
            public IReadOnlyList<(string name, string alias)> Sources => _sources.AsReadOnly();

            /// <summary>
            /// This denotes a concatenation of all <paramref name="inputColumnNames"/> into column called <paramref name="name"/>.
            /// </summary>
            public ColumnOptions(string name, params string[] inputColumnNames)
                : this(name, GetPairs(inputColumnNames))
            {
            }

            private static IEnumerable<(string name, string alias)> GetPairs(string[] inputColumnNames)
            {
                Contracts.CheckValue(inputColumnNames, nameof(inputColumnNames));
                return inputColumnNames.Select(name => (name, (string)null));
            }

            /// <summary>
            /// This denotes a concatenation of input columns into one column called <paramref name="name"/>.
            /// For each input column, an 'alias' can be specified, to be used in constructing the resulting slot names.
            /// If the alias is not specified, it defaults to be column name.
            /// </summary>
            public ColumnOptions(string name, IEnumerable<(string name, string alias)> inputColumnNames)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckValue(inputColumnNames, nameof(inputColumnNames));
                Contracts.CheckParam(inputColumnNames.Any(), nameof(inputColumnNames), "Can not be empty");

                foreach (var (output, alias) in inputColumnNames)
                {
                    Contracts.CheckNonEmpty(output, nameof(inputColumnNames));
                    Contracts.CheckValueOrNull(alias);
                }

                Name = name;
                _sources = inputColumnNames.ToArray();
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);
                // *** Binary format ***
                // int: id of output
                // int: number of inputs
                // for each input
                //   int: id of name
                //   int: id of alias

                ctx.SaveNonEmptyString(Name);
                Contracts.Assert(_sources.Length > 0);
                ctx.Writer.Write(_sources.Length);
                foreach (var (name, alias) in _sources)
                {
                    ctx.SaveNonEmptyString(name);
                    ctx.SaveStringOrNull(alias);
                }
            }

            internal ColumnOptions(ModelLoadContext ctx)
            {
                Contracts.AssertValue(ctx);
                // *** Binary format ***
                // int: id of output
                // int: number of inputs
                // for each input
                //   int: id of name
                //   int: id of alias

                Name = ctx.LoadNonEmptyString();
                int n = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(n > 0);
                _sources = new (string name, string alias)[n];
                for (int i = 0; i < n; i++)
                {
                    var name = ctx.LoadNonEmptyString();
                    var alias = ctx.LoadStringOrNull();
                    _sources[i] = (name, alias);
                }
            }
        }

        private readonly ColumnOptions[] _columns;

        /// <summary>
        /// The names of the output and input column pairs for the transformation.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string[] inputColumnNames)> Columns
            => _columns.Select(col => (outputColumnName: col.Name, inputColumnNames: col.Sources.Select(source => source.name).ToArray())).ToArray().AsReadOnly();

        /// <summary>
        /// Concatename columns in <paramref name="inputColumnNames"/> into one column <paramref name="outputColumnName"/>.
        /// Original columns are also preserved.
        /// The column types must match, and the output column type is always a vector.
        /// </summary>
        internal ColumnConcatenatingTransformer(IHostEnvironment env, string outputColumnName, params string[] inputColumnNames)
            : this(env, new ColumnOptions(outputColumnName, inputColumnNames))
        {
        }

        /// <summary>
        /// Concatenates multiple groups of columns, each group is denoted by one of <paramref name="columns"/>.
        /// </summary>
        internal ColumnConcatenatingTransformer(IHostEnvironment env, params ColumnOptions[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ColumnConcatenatingTransformer)))
        {
            Contracts.CheckValue(columns, nameof(columns));
            _columns = columns.ToArray();
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CONCAT F",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Added aliases
                verWrittenCur: 0x00010003, // Converted to transformer
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld,
                loaderAssemblyName: typeof(ColumnConcatenatingTransformer).Assembly.FullName);
        }

        private const int VersionAddedAliases = 0x00010002;
        private const int VersionTransformer = 0x00010003;

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of columns
            // for each column:
            //    columnOptions

            Contracts.Assert(_columns.Length > 0);
            ctx.Writer.Write(_columns.Length);
            foreach (var col in _columns)
                col.Save(ctx);
        }

        /// <summary>
        /// Factory method for SignatureLoadModel.
        /// </summary>
        private ColumnConcatenatingTransformer(IHostEnvironment env, ModelLoadContext ctx) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ColumnConcatenatingTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten >= VersionTransformer)
            {
                // *** Binary format ***
                // int: number of columns
                // for each column:
                //    columnOptions
                int n = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(n > 0);
                _columns = new ColumnOptions[n];
                for (int i = 0; i < n; i++)
                    _columns[i] = new ColumnOptions(ctx);
            }
            else
                _columns = LoadLegacy(ctx);
        }

        private ColumnOptions[] LoadLegacy(ModelLoadContext ctx)
        {
            // *** Legacy binary format ***
            // int: sizeof(Float).
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: number of input column names
            //   int[]: ids of input column names
            // if version >= VersionAddedAliases
            //   foreach column:
            //      foreach non-null alias
            //          int: index of the alias
            //          int: string id of the alias
            //      int: -1, marks the end of the list

            var sizeofFloat = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(sizeofFloat == sizeof(float));

            int n = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(n > 0);
            var names = new string[n];
            var inputs = new string[n][];
            for (int i = 0; i < n; i++)
            {
                names[i] = ctx.LoadNonEmptyString();
                int numSources = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(numSources > 0);
                inputs[i] = new string[numSources];
                for (int j = 0; j < numSources; j++)
                    inputs[i][j] = ctx.LoadNonEmptyString();
            }

            var aliases = new string[n][];
            if (ctx.Header.ModelVerReadable >= VersionAddedAliases)
            {
                for (int i = 0; i < n; i++)
                {
                    var length = inputs[i].Length;
                    aliases[i] = new string[length];
                    if (ctx.Header.ModelVerReadable >= VersionAddedAliases)
                    {
                        for (; ; )
                        {
                            var j = ctx.Reader.ReadInt32();
                            if (j == -1)
                                break;
                            Contracts.CheckDecode(0 <= j && j < length);
                            Contracts.CheckDecode(aliases[i][j] == null);
                            aliases[i][j] = ctx.LoadNonEmptyString();
                        }
                    }
                }
            }

            var result = new ColumnOptions[n];
            for (int i = 0; i < n; i++)
                result[i] = new ColumnOptions(names[i],
                    inputs[i].Zip(aliases[i], (name, alias) => (name, alias)));
            return result;
        }

        ///<summary>
        /// Factory method for SignatureDataTransform.
        /// </summary>
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns));

            for (int i = 0; i < options.Columns.Length; i++)
                env.CheckUserArg(Utils.Size(options.Columns[i].Source) > 0, nameof(options.Columns));

            var cols = options.Columns
                .Select(c => new ColumnOptions(c.Name, c.Source))
                .ToArray();
            var transformer = new ColumnConcatenatingTransformer(env, cols);
            return transformer.MakeDataTransform(input);
        }
        /// <summary>
        /// Factory method corresponding to SignatureDataTransform.
        /// </summary>
        [BestFriend]
        internal static IDataTransform Create(IHostEnvironment env, TaggedOptions options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns));

            for (int i = 0; i < options.Columns.Length; i++)
                env.CheckUserArg(Utils.Size(options.Columns[i].Source) > 0, nameof(options.Columns));

            var cols = options.Columns
                .Select(c => new ColumnOptions(c.Name, c.Source.Select(kvp => (kvp.Value, kvp.Key != "" ? kvp.Key : null))))
                .ToArray();
            var transformer = new ColumnConcatenatingTransformer(env, cols);
            return transformer.MakeDataTransform(input);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema) => new Mapper(this, inputSchema);

        /// <summary>
        /// Factory method for SignatureLoadDataTransform.
        /// </summary>
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => new ColumnConcatenatingTransformer(env, ctx).MakeDataTransform(input);

        /// <summary>
        /// Factory method for SignatureLoadRowMapper.
        /// </summary>
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => new ColumnConcatenatingTransformer(env, ctx).MakeRowMapper(inputSchema);

        private sealed class Mapper : MapperBase, ISaveAsOnnx, ISaveAsPfa
        {
            private readonly ColumnConcatenatingTransformer _parent;
            private readonly BoundColumn[] _columns;

            public bool CanSaveOnnx(OnnxContext ctx) => true;
            public bool CanSavePfa => true;

            public Mapper(ColumnConcatenatingTransformer parent, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;

                _columns = new BoundColumn[_parent._columns.Length];
                for (int i = 0; i < _parent._columns.Length; i++)
                {
                    _columns[i] = MakeColumn(inputSchema, i);
                }
            }

            private BoundColumn MakeColumn(DataViewSchema inputSchema, int iinfo)
            {
                Contracts.AssertValue(inputSchema);
                Contracts.Assert(0 <= iinfo && iinfo < _parent._columns.Length);

                DataViewType itemType = null;
                int[] sources = new int[_parent._columns[iinfo].Sources.Count];
                // Go through the columns, and establish the following:
                // - indices of input columns in the input schema. Throw if they are not there.
                // - output type. Throw if the types of inputs are not the same.
                // - how many slots are there in the output vector (or variable). Denoted by totalSize.
                // - total size of CategoricalSlotRanges metadata, if present. Denoted by catCount.
                // - whether the column is normalized.
                //      It is true when ALL inputs are normalized (and of numeric type).
                // - whether the column has slot names.
                //      It is true if ANY input is a scalar, or has slot names.
                // - whether the column has categorical slot ranges.
                //      It is true if ANY input has this metadata.
                int totalSize = 0;
                int catCount = 0;
                bool isNormalized = true;
                bool hasSlotNames = false;
                bool hasCategoricals = false;
                for (int i = 0; i < _parent._columns[iinfo].Sources.Count; i++)
                {
                    var (srcName, srcAlias) = _parent._columns[iinfo].Sources[i];
                    if (!inputSchema.TryGetColumnIndex(srcName, out int srcCol))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", srcName);
                    sources[i] = srcCol;

                    var curType = inputSchema[srcCol].Type;
                    VectorType curVectorType = curType as VectorType;

                    DataViewType currentItemType = curVectorType?.ItemType ?? curType;
                    int currentValueCount = curVectorType?.Size ?? 1;

                    if (itemType == null)
                    {
                        itemType = currentItemType;
                        totalSize = currentValueCount;
                    }
                    else if (currentItemType.Equals(itemType))
                    {
                        // If any one input is variable length, then the output is variable length.
                        if (totalSize == 0 || currentValueCount == 0)
                            totalSize = 0;
                        else
                            totalSize += currentValueCount;
                    }
                    else
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", srcName, itemType.ToString(), curType.ToString());

                    if (isNormalized && !inputSchema[srcCol].IsNormalized())
                        isNormalized = false;

                    if (AnnotationUtils.TryGetCategoricalFeatureIndices(inputSchema, srcCol, out int[] typeCat))
                    {
                        Contracts.Assert(typeCat.Length > 0);
                        catCount += typeCat.Length;
                        hasCategoricals = true;
                    }

                    if ((!hasSlotNames && curVectorType == null)
                        || (curVectorType != null && inputSchema[srcCol].HasSlotNames(curVectorType.Size)))
                        hasSlotNames = true;
                }

                if (!(itemType is NumberDataViewType))
                    isNormalized = false;
                if (totalSize == 0)
                {
                    hasCategoricals = false;
                    hasSlotNames = false;
                }

                return new BoundColumn(InputSchema, _parent._columns[iinfo], sources, new VectorType((PrimitiveDataViewType)itemType, totalSize),
                    isNormalized, hasSlotNames, hasCategoricals, totalSize, catCount);
            }

            /// <summary>
            /// This represents the column information bound to the schema.
            /// </summary>
            private sealed class BoundColumn
            {
                public readonly int[] SrcIndices;

                private readonly ColumnOptions _columnOptions;
                private readonly DataViewType[] _srcTypes;

                public readonly VectorType OutputType;

                // Fields pertaining to column metadata.
                private readonly bool _isIdentity;
                private readonly bool _isNormalized;
                private readonly bool _hasSlotNames;
                private readonly bool _hasCategoricals;

                private readonly VectorType _slotNamesType;
                private readonly DataViewType _categoricalRangeType;

                private readonly DataViewSchema _inputSchema;

                public BoundColumn(DataViewSchema inputSchema, ColumnOptions columnOptions, int[] sources, VectorType outputType,
                    bool isNormalized, bool hasSlotNames, bool hasCategoricals, int slotCount, int catCount)
                {
                    _columnOptions = columnOptions;
                    SrcIndices = sources;
                    _srcTypes = sources.Select(c => inputSchema[c].Type).ToArray();

                    OutputType = outputType;

                    _inputSchema = inputSchema;

                    _isIdentity = SrcIndices.Length == 1 && _inputSchema[SrcIndices[0]].Type is VectorType;
                    _isNormalized = isNormalized;

                    _hasSlotNames = hasSlotNames;
                    if (_hasSlotNames)
                        _slotNamesType = AnnotationUtils.GetNamesType(slotCount);

                    _hasCategoricals = hasCategoricals;
                    if (_hasCategoricals)
                        _categoricalRangeType = AnnotationUtils.GetCategoricalType(catCount / 2);
                }

                public DataViewSchema.DetachedColumn MakeSchemaColumn()
                {
                    if (_isIdentity)
                    {
                        var inputCol = _inputSchema[SrcIndices[0]];
                        return new DataViewSchema.DetachedColumn(_columnOptions.Name, inputCol.Type, inputCol.Annotations);
                    }

                    var metadata = new DataViewSchema.Annotations.Builder();
                    if (_isNormalized)
                        metadata.Add(AnnotationUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, (ValueGetter<bool>)GetIsNormalized);
                    if (_hasSlotNames)
                        metadata.AddSlotNames(_slotNamesType.Size, GetSlotNames);
                    if (_hasCategoricals)
                        metadata.Add(AnnotationUtils.Kinds.CategoricalSlotRanges, _categoricalRangeType, (ValueGetter<VBuffer<int>>)GetCategoricalSlotRanges);

                    return new DataViewSchema.DetachedColumn(_columnOptions.Name, OutputType, metadata.ToAnnotations());
                }

                private void GetIsNormalized(ref bool value) => value = _isNormalized;

                private void GetCategoricalSlotRanges(ref VBuffer<int> dst)
                {
                    List<int> allValues = new List<int>();
                    int slotCount = 0;
                    for (int i = 0; i < SrcIndices.Length; i++)
                    {

                        Contracts.Assert(_srcTypes[i].GetValueCount() > 0);

                        if (i > 0)
                            slotCount += _srcTypes[i - 1].GetValueCount();

                        if (AnnotationUtils.TryGetCategoricalFeatureIndices(_inputSchema, SrcIndices[i], out int[] values))
                        {
                            Contracts.Assert(values.Length > 0 && values.Length % 2 == 0);

                            for (int j = 0; j < values.Length; j++)
                                allValues.Add(values[j] + slotCount);
                        }
                    }

                    Contracts.Assert(allValues.Count > 0);

                    dst = new VBuffer<int>(allValues.Count, allValues.ToArray());
                }

                private void GetSlotNames(ref VBuffer<ReadOnlyMemory<char>> dst)
                {
                    Contracts.Assert(!_isIdentity);
                    Contracts.Assert(OutputType.Size > 0);

                    Contracts.AssertValue(_slotNamesType);
                    Contracts.Assert(_slotNamesType.Size == OutputType.Size);

                    var bldr = BufferBuilder<ReadOnlyMemory<char>>.CreateDefault();
                    bldr.Reset(_slotNamesType.Size, dense: false);

                    var sb = new StringBuilder();
                    var names = default(VBuffer<ReadOnlyMemory<char>>);
                    int slot = 0;
                    for (int i = 0; i < _srcTypes.Length; i++)
                    {
                        int colSrc = SrcIndices[i];
                        var typeSrc = _srcTypes[i];
                        Contracts.Assert(_columnOptions.Sources[i].alias != "");
                        var colName = _inputSchema[colSrc].Name;
                        var nameSrc = _columnOptions.Sources[i].alias ?? colName;
                        if (!(typeSrc is VectorType vectorTypeSrc))
                        {
                            bldr.AddFeature(slot++, nameSrc.AsMemory());
                            continue;
                        }

                        Contracts.Assert(vectorTypeSrc.IsKnownSize);
                        VectorType typeNames = null;

                        var inputMetadata = _inputSchema[colSrc].Annotations;
                        if (inputMetadata != null && inputMetadata.Schema.TryGetColumnIndex(AnnotationUtils.Kinds.SlotNames, out int idx))
                            typeNames = inputMetadata.Schema[idx].Type as VectorType;

                        if (typeNames != null && typeNames.Size == vectorTypeSrc.Size && typeNames.ItemType is TextDataViewType)
                        {
                            inputMetadata.GetValue(AnnotationUtils.Kinds.SlotNames, ref names);
                            sb.Clear();
                            if (_columnOptions.Sources[i].alias != colName)
                                sb.Append(nameSrc).Append(".");
                            int len = sb.Length;
                            foreach (var kvp in names.Items())
                            {
                                if (kvp.Value.IsEmpty)
                                    continue;
                                sb.Length = len;
                                sb.AppendMemory(kvp.Value);
                                bldr.AddFeature(slot + kvp.Key, sb.ToString().AsMemory());
                            }
                        }
                        slot += vectorTypeSrc.Size;
                    }
                    Contracts.Assert(slot == OutputType.Size);

                    bldr.GetResult(ref dst);
                }

                public Delegate MakeGetter(DataViewRow input)
                {
                    if (_isIdentity)
                        return Utils.MarshalInvoke(MakeIdentityGetter<int>, OutputType.RawType, input);

                    return Utils.MarshalInvoke(MakeGetter<int>, OutputType.ItemType.RawType, input);
                }

                private Delegate MakeIdentityGetter<T>(DataViewRow input)
                {
                    Contracts.Assert(SrcIndices.Length == 1);
                    return input.GetGetter<T>(input.Schema[SrcIndices[0]]);
                }

                private Delegate MakeGetter<T>(DataViewRow input)
                {
                    var srcGetterOnes = new ValueGetter<T>[SrcIndices.Length];
                    var srcGetterVecs = new ValueGetter<VBuffer<T>>[SrcIndices.Length];

                    for (int j = 0; j < SrcIndices.Length; j++)
                    {
                        var column = input.Schema[SrcIndices[j]];

                        if (_srcTypes[j] is VectorType)
                            srcGetterVecs[j] = input.GetGetter<VBuffer<T>>(column);
                        else
                            srcGetterOnes[j] = input.GetGetter<T>(column);
                    }

                    T tmp = default(T);
                    VBuffer<T>[] tmpBufs = new VBuffer<T>[SrcIndices.Length];
                    ValueGetter<VBuffer<T>> result = (ref VBuffer<T> dst) =>
                    {
                        int dstLength = 0;
                        int dstCount = 0;
                        for (int i = 0; i < SrcIndices.Length; i++)
                        {
                            var type = _srcTypes[i];
                            if (type is VectorType vectorType)
                            {
                                srcGetterVecs[i](ref tmpBufs[i]);
                                if (vectorType.Size != 0 && vectorType.Size != tmpBufs[i].Length)
                                {
                                    throw Contracts.Except("Column '{0}': expected {1} slots, but got {2}",
                                        input.Schema[SrcIndices[i]].Name, vectorType.Size, tmpBufs[i].Length)
                                        .MarkSensitive(MessageSensitivity.Schema);
                                }
                                dstLength = checked(dstLength + tmpBufs[i].Length);
                                dstCount = checked(dstCount + tmpBufs[i].GetValues().Length);
                            }
                            else
                            {
                                dstLength = checked(dstLength + 1);
                                dstCount = checked(dstCount + 1);
                            }
                        }

                        if (dstCount <= dstLength / 2)
                        {
                            // Concatenate into a sparse representation.
                            var editor = VBufferEditor.Create(ref dst, dstLength, dstCount);

                            int offset = 0;
                            int count = 0;
                            for (int j = 0; j < SrcIndices.Length; j++)
                            {
                                Contracts.Assert(offset < dstLength);
                                if (_srcTypes[j] is VectorType)
                                {
                                    var buffer = tmpBufs[j];
                                    var bufferValues = buffer.GetValues();
                                    Contracts.Assert(bufferValues.Length <= dstCount - count);
                                    Contracts.Assert(buffer.Length <= dstLength - offset);
                                    if (buffer.IsDense)
                                    {
                                        for (int i = 0; i < bufferValues.Length; i++)
                                        {
                                            editor.Values[count] = bufferValues[i];
                                            editor.Indices[count++] = offset + i;
                                        }
                                    }
                                    else
                                    {
                                        var bufferIndices = buffer.GetIndices();
                                        for (int i = 0; i < bufferValues.Length; i++)
                                        {
                                            editor.Values[count] = bufferValues[i];
                                            editor.Indices[count++] = offset + bufferIndices[i];
                                        }
                                    }
                                    offset += buffer.Length;
                                }
                                else
                                {
                                    Contracts.Assert(count < dstCount);
                                    srcGetterOnes[j](ref tmp);
                                    editor.Values[count] = tmp;
                                    editor.Indices[count++] = offset;
                                    offset++;
                                }
                            }
                            Contracts.Assert(count <= dstCount);
                            Contracts.Assert(offset == dstLength);
                            dst = editor.CommitTruncated(count);
                        }
                        else
                        {
                            // Concatenate into a dense representation.
                            var editor = VBufferEditor.Create(ref dst, dstLength);

                            int offset = 0;
                            for (int j = 0; j < SrcIndices.Length; j++)
                            {
                                Contracts.Assert(tmpBufs[j].Length <= dstLength - offset);
                                if (_srcTypes[j] is VectorType)
                                {
                                    tmpBufs[j].CopyTo(editor.Values, offset);
                                    offset += tmpBufs[j].Length;
                                }
                                else
                                {
                                    srcGetterOnes[j](ref tmp);
                                    editor.Values[offset++] = tmp;
                                }
                            }
                            Contracts.Assert(offset == dstLength);
                            dst = editor.Commit();
                        }
                    };
                    return result;
                }

                public KeyValuePair<string, JToken> SavePfaInfo(BoundPfaContext ctx)
                {
                    Contracts.AssertValue(ctx);
                    string outName = _columnOptions.Name;
                    if (!OutputType.IsKnownSize) // Do not attempt variable length.
                        return new KeyValuePair<string, JToken>(outName, null);

                    string[] srcTokens = new string[SrcIndices.Length];
                    bool[] srcPrimitive = new bool[SrcIndices.Length];
                    for (int i = 0; i < SrcIndices.Length; ++i)
                    {
                        var srcName = _columnOptions.Sources[i].name;
                        if ((srcTokens[i] = ctx.TokenOrNullForName(srcName)) == null)
                            return new KeyValuePair<string, JToken>(outName, null);
                        srcPrimitive[i] = _srcTypes[i] is PrimitiveDataViewType;
                    }
                    Contracts.Assert(srcTokens.All(tok => tok != null));
                    var itemColumnType = OutputType.ItemType;
                    var itemType = PfaType.PfaTypeOrNullForColumnType(itemColumnType);
                    if (itemType == null)
                        return new KeyValuePair<string, JToken>(outName, null);
                    JObject jobj = null;
                    var arrType = PfaType.Array(itemType);

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

                    Contracts.AssertValue(result);
                    return new KeyValuePair<string, JToken>(outName, result);
                }
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.Count];
                for (int i = 0; i < _columns.Length; i++)
                {
                    if (activeOutput(i))
                    {
                        foreach (var src in _columns[i].SrcIndices)
                            active[src] = true;
                    }
                }
                return col => active[col];
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore() => _columns.Select(x => x.MakeSchemaColumn()).ToArray();

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                return _columns[iinfo].MakeGetter(input);
            }

            public void SaveAsPfa(BoundPfaContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                var toHide = new List<string>();
                var toDeclare = new List<KeyValuePair<string, JToken>>();

                for (int iinfo = 0; iinfo < _columns.Length; ++iinfo)
                {
                    var toSave = _columns[iinfo].SavePfaInfo(ctx);
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
                Contracts.Assert(CanSaveOnnx(ctx));

                string opType = "FeatureVectorizer";
                for (int iinfo = 0; iinfo < _columns.Length; ++iinfo)
                {
                    var colInfo = _parent._columns[iinfo];
                    var boundCol = _columns[iinfo];

                    string outName = colInfo.Name;
                    var outColType = boundCol.OutputType;
                    if (!outColType.IsKnownSize)
                    {
                        ctx.RemoveColumn(outName, false);
                        continue;
                    }

                    List<KeyValuePair<string, long>> inputList = new List<KeyValuePair<string, long>>();
                    for (int i = 0; i < boundCol.SrcIndices.Length; ++i)
                    {
                        var srcName = colInfo.Sources[i].name;
                        if (!ctx.ContainsColumn(srcName))
                        {
                            ctx.RemoveColumn(outName, false);
                            return;
                        }

                        var srcIndex = boundCol.SrcIndices[i];
                        inputList.Add(new KeyValuePair<string, long>(ctx.GetVariableName(srcName),
                            InputSchema[srcIndex].Type.GetValueCount()));
                    }

                    var node = ctx.CreateNode(opType, inputList.Select(t => t.Key),
                        new[] { ctx.AddIntermediateVariable(outColType, outName) }, ctx.GetNodeName(opType));

                    node.AddAttribute("inputdimensions", inputList.Select(x => x.Value));
                }
            }
        }
    }
}