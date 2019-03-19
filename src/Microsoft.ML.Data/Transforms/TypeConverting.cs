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
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(TypeConvertingTransformer.Summary, typeof(IDataTransform), typeof(TypeConvertingTransformer), typeof(TypeConvertingTransformer.Options), typeof(SignatureDataTransform),
    TypeConvertingTransformer.UserName, TypeConvertingTransformer.ShortName, "ConvertTransform", DocName = "transform/ConvertTransform.md")]

[assembly: LoadableClass(TypeConvertingTransformer.Summary, typeof(IDataTransform), typeof(TypeConvertingTransformer), null, typeof(SignatureLoadDataTransform),
    TypeConvertingTransformer.UserName, TypeConvertingTransformer.LoaderSignature, TypeConvertingTransformer.LoaderSignatureOld)]

[assembly: LoadableClass(TypeConvertingTransformer.Summary, typeof(TypeConvertingTransformer), null, typeof(SignatureLoadModel),
    TypeConvertingTransformer.UserName, TypeConvertingTransformer.LoaderSignature)]

[assembly: LoadableClass(TypeConvertingTransformer.Summary, typeof(IRowMapper), typeof(TypeConvertingTransformer), null, typeof(SignatureLoadRowMapper),
    TypeConvertingTransformer.UserName, TypeConvertingTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(TypeConversion))]

namespace Microsoft.ML.Transforms
{
    internal static class TypeConversion
    {
        [TlcModule.EntryPoint(Name = "Transforms.ColumnTypeConverter", Desc = TypeConvertingTransformer.Summary, UserName = TypeConvertingTransformer.UserName, ShortName = TypeConvertingTransformer.ShortName)]
        public static CommonOutputs.TransformOutput Convert(IHostEnvironment env, TypeConvertingTransformer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "Convert", input);
            var view = TypeConvertingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {

                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }
    }

    /// <summary>
    /// <see cref="TypeConvertingTransformer"/> converts underlying column types.
    /// The source and destination column types need to be compatible.
    /// </summary>
    public sealed class TypeConvertingTransformer : OneToOneTransformerBase
    {
        [BestFriend]
        internal class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The result type", ShortName = "type")]
            public InternalDataKind? ResultType;

            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the cardinality/count of valid key values", ShortName = "key", Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public KeyCount KeyCount;

            [Argument(ArgumentType.AtMostOnce, HelpText = "For a key column, this defines the range of values", ShortName = "key", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string Range;

            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // We accept N:T:S where N is the new column name, T is the new type,
                // and S is source column names.
                if (!base.TryParse(str, out string extra))
                    return false;
                if (extra == null)
                    return true;

                if (!TypeParsingUtils.TryParseDataKind(extra, out InternalDataKind kind, out KeyCount))
                    return false;
                ResultType = kind == default ? default(InternalDataKind?) : kind;
                return true;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (ResultType == null && KeyCount == null)
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
                if (KeyCount != null)
                {
                    sb.Append('[');
                    if (!KeyCount.TryUnparse(sb))
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

        [BestFriend]
        internal class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:type:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The result type", ShortName = "type", SortOrder = 2)]
            public InternalDataKind? ResultType;

            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the range of values", ShortName = "key", Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public KeyCount KeyCount;

            [Argument(ArgumentType.AtMostOnce, HelpText = "For a key column, this defines the range of values", ShortName = "key", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string Range;
        }

        internal const string Summary = "Converts a column to a different type, using standard conversions.";
        internal const string UserName = "Convert Transform";
        internal const string ShortName = "Convert";

        internal const string LoaderSignature = "ConvertTransform";
        internal const string LoaderSignatureOld = "ConvertFunction";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CONVERTF",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // Added support for keyRange
                //verWrittenCur: 0x00010003, // Change to transformer leads to change of saving objects.
                verWrittenCur: 0x00010004, // Removed Min and Contiguous from KeyCount.
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld,
                loaderAssemblyName: typeof(TypeConvertingTransformer).Assembly.FullName);
        }

        private const uint VersionNoMinCount = 0x00010004;
        private const int VersionTransformer = 0x00010003;

        private const string RegistrationName = "Convert";

        /// <summary>
        /// A collection of <see cref="TypeConvertingEstimator.ColumnOptions"/> describing the settings of the transformation.
        /// </summary>
        internal IReadOnlyCollection<TypeConvertingEstimator.ColumnOptions> Columns => _columns.AsReadOnly();

        private readonly TypeConvertingEstimator.ColumnOptions[] _columns;

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(TypeConvertingEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckNonEmpty(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        /// <summary>
        /// Convinence constructor for simple one column case.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the output column.</param>
        /// <param name="inputColumnName">Name of the column to be transformed. If this is null '<paramref name="outputColumnName"/>' will be used.</param>
        /// <param name="outputKind">The expected type of the converted column.</param>
        /// <param name="outputKeyCount">New key count if we work with key type.</param>
        internal TypeConvertingTransformer(IHostEnvironment env, string outputColumnName, DataKind outputKind, string inputColumnName = null, KeyCount outputKeyCount = null)
            : this(env, new TypeConvertingEstimator.ColumnOptions(outputColumnName, outputKind, inputColumnName ?? outputColumnName, outputKeyCount))
        {
        }

        /// <summary>
        /// Create a <see cref="TypeConvertingTransformer"/> that takes multiple pairs of columns.
        /// </summary>
        internal TypeConvertingTransformer(IHostEnvironment env, params TypeConvertingEstimator.ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TypeConvertingTransformer)), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // for each added column
            //   byte: data kind, with high bit set if there is a keyCout
            //   if there is a keyCount
            //     ulong: keyCount (0 for unspecified)
            SaveColumns(ctx);

            for (int i = 0; i < _columns.Length; i++)
            {
                Host.Assert((InternalDataKind)(byte)_columns[i].OutputKind.ToInternalDataKind() == _columns[i].OutputKind.ToInternalDataKind());
                if (_columns[i].OutputKeyCount != null)
                {
                    byte b = (byte)_columns[i].OutputKind;
                    b |= 0x80;
                    ctx.Writer.Write(b);
                    ctx.Writer.Write(_columns[i].OutputKeyCount.Count ?? _columns[i].OutputKind.ToInternalDataKind().ToMaxInt());
                }
                else
                    ctx.Writer.Write((byte)_columns[i].OutputKind);
            }
        }

        // Factory method for SignatureLoadModel.
        private static TypeConvertingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten < VersionTransformer)
            {
                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));
            }
            return new TypeConvertingTransformer(host, ctx);
        }

        private TypeConvertingTransformer(IHost host, ModelLoadContext ctx)
        : base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            // *** Binary format ***
            // <base>
            // for each added column
            //   byte: data kind, with high bit set if there is a keyCount
            //   if there is a keyCount
            //     ulong: keyCount (0 for unspecified)

            _columns = new TypeConvertingEstimator.ColumnOptions[columnsLength];
            for (int i = 0; i < columnsLength; i++)
            {
                byte b = ctx.Reader.ReadByte();
                var kind = (InternalDataKind)(b & 0x7F);
                Host.CheckDecode(Enum.IsDefined(typeof(InternalDataKind), kind));
                KeyCount keyCount = null;
                ulong count = 0;
                if ((b & 0x80) != 0)
                {
                    // Special treatment for versions that had Min and Contiguous fields in KeyType.
                    if (ctx.Header.ModelVerWritten < VersionNoMinCount)
                    {
                        // We no longer support non zero Min for KeyType.
                        ulong min = ctx.Reader.ReadUInt64();
                        Host.CheckDecode(min == 0);
                        // KeyRange became KeyCount, and its count is 1 + KeyRange.Max.
                        count = ctx.Reader.ReadUInt64() + 1;
                        // We no longer support non contiguous values for KeyType.
                        bool contiguous = ctx.Reader.ReadBoolByte();
                        Host.CheckDecode(contiguous);
                    }
                    else
                        count = ctx.Reader.ReadUInt64();

                    Host.CheckDecode(0 < count);
                    keyCount = new KeyCount(count);

                }
                _columns[i] = new TypeConvertingEstimator.ColumnOptions(ColumnPairs[i].outputColumnName, kind.ToDataKind(), ColumnPairs[i].inputColumnName, keyCount);
            }
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new TypeConvertingEstimator.ColumnOptions[options.Columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = options.Columns[i];
                var tempResultType = item.ResultType ?? options.ResultType;
                KeyCount keyCount = null;
                // If KeyCount or Range are defined on this column, set keyCount to the appropriate value.
                if (item.KeyCount != null)
                    keyCount = item.KeyCount;
                else if (item.Range != null)
                    keyCount = KeyCount.Parse(item.Range);
                // If KeyCount and Range are not defined for this column, we set keyCount to the value
                // defined in the Arguments object only in case the ResultType is not defined on the column.
                else if (item.ResultType == null)
                {
                    if (options.KeyCount != null)
                        keyCount = options.KeyCount;
                    else if (options.Range != null)
                        keyCount = KeyCount.Parse(options.Range);
                }

                InternalDataKind kind;
                if (tempResultType == null)
                {
                    if (keyCount == null)
                        kind = InternalDataKind.Num;
                    else
                    {
                        var srcType = input.Schema[item.Source ?? item.Name].Type;
                        kind = srcType is KeyType ? srcType.GetRawKind() : InternalDataKind.U8;
                    }
                }
                else
                {
                    kind = tempResultType.Value;
                }
                cols[i] = new TypeConvertingEstimator.ColumnOptions(item.Name, kind.ToDataKind(), item.Source ?? item.Name, keyCount);
            };
            return new TypeConvertingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        internal static bool GetNewType(IExceptionContext ectx, DataViewType srcType, InternalDataKind kind, KeyCount keyCount, out PrimitiveDataViewType itemType)
        {
            if (keyCount != null)
            {
                itemType = TypeParsingUtils.ConstructKeyType(kind, keyCount);
                DataViewType srcItemType = srcType.GetItemType();
                if (!(srcItemType is KeyType) && !(srcItemType is TextDataViewType))
                    return false;
            }
            else if (!(srcType.GetItemType() is KeyType key))
                itemType = ColumnTypeExtensions.PrimitiveTypeFromKind(kind);
            else if (!KeyType.IsValidDataType(kind.ToType()))
            {
                itemType = ColumnTypeExtensions.PrimitiveTypeFromKind(kind);
                return false;
            }
            else
            {
                ectx.Assert(KeyType.IsValidDataType(key.RawType));
                ulong count = key.Count;
                // Technically, it's an error for the counts not to match, but we'll let the Conversions
                // code return false below. There's a possibility we'll change the standard conversions to
                // map out of bounds values to zero, in which case, this is the right thing to do.
                ulong max = kind.ToMaxInt();
                if (count > max)
                    count = max;
                itemType = new KeyType(kind.ToType(), count);
            }
            return true;
        }

        private sealed class Mapper : OneToOneMapperBase, ICanSaveOnnx
        {
            private readonly TypeConvertingTransformer _parent;
            private readonly DataViewType[] _types;
            private readonly int[] _srcCols;

            public bool CanSaveOnnx(OnnxContext ctx) => ctx.GetOnnxVersion() == OnnxVersion.Experimental;

            public Mapper(TypeConvertingTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new DataViewType[_parent._columns.Length];
                _srcCols = new int[_parent._columns.Length];
                for (int i = 0; i < _parent._columns.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out _srcCols[i]);
                    var srcCol = inputSchema[_srcCols[i]];
                    if (!CanConvertToType(Host, srcCol.Type, _parent._columns[i].OutputKind.ToInternalDataKind(), _parent._columns[i].OutputKeyCount,
                        out PrimitiveDataViewType itemType, out _types[i]))
                    {
                        throw Host.ExceptParam(nameof(inputSchema),
                        "source column '{0}' with item type '{1}' is not compatible with destination type '{2}'",
                        _parent._columns[i].InputColumnName, srcCol.Type, itemType);
                    }
                }
            }

            private static bool CanConvertToType(IExceptionContext ectx, DataViewType srcType, InternalDataKind kind, KeyCount keyCount,
                out PrimitiveDataViewType itemType, out DataViewType typeDst)
            {
                ectx.AssertValue(srcType);
                ectx.Assert(Enum.IsDefined(typeof(InternalDataKind), kind));

                typeDst = null;
                if (!GetNewType(ectx, srcType, kind, keyCount, out itemType))
                    return false;

                // Ensure that the conversion is legal. We don't actually cache the delegate here. It will get
                // re-fetched by the utils code when needed.
                if (!Data.Conversion.Conversions.Instance.TryGetStandardConversion(srcType.GetItemType(), itemType, out Delegate del, out bool identity))
                    return false;

                typeDst = itemType;
                if (srcType is VectorType vectorType)
                    typeDst = new VectorType(itemType, vectorType);

                return true;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent._columns.Length];
                for (int i = 0; i < _parent._columns.Length; i++)
                {
                    var builder = new DataViewSchema.Annotations.Builder();
                    var srcType = InputSchema[_srcCols[i]].Type;
                    if (_types[i].IsKnownSizeVector())
                        builder.Add(InputSchema[ColMapNewToOld[i]].Annotations, name => name == AnnotationUtils.Kinds.SlotNames);

                    DataViewType srcItemType = srcType.GetItemType();
                    DataViewType currentItemType = _types[i].GetItemType();

                    KeyType srcItemKeyType = srcItemType as KeyType;
                    KeyType currentItemKeyType = currentItemType as KeyType;
                    if (srcItemKeyType != null && currentItemKeyType != null &&
                        srcItemKeyType.Count > 0 && srcItemKeyType.Count == currentItemKeyType.Count)
                    {
                        builder.Add(InputSchema[ColMapNewToOld[i]].Annotations, name => name == AnnotationUtils.Kinds.KeyValues);
                    }

                    if (srcItemType is NumberDataViewType && currentItemType is NumberDataViewType)
                        builder.Add(InputSchema[ColMapNewToOld[i]].Annotations, name => name == AnnotationUtils.Kinds.IsNormalized);
                    if (srcType is BooleanDataViewType && currentItemType is NumberDataViewType)
                    {
                        ValueGetter<bool> getter = (ref bool dst) => dst = true;
                        builder.Add(AnnotationUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, getter);
                    }
                    result[i] = new DataViewSchema.DetachedColumn(_parent._columns[i].Name, _types[i], builder.ToAnnotations());
                }
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;
                if (!(_types[iinfo] is VectorType vectorType))
                    return RowCursorUtils.GetGetterAs(_types[iinfo], input, _srcCols[iinfo]);
                return RowCursorUtils.GetVecGetterAs(vectorType.ItemType, input, _srcCols[iinfo]);
            }

            public void SaveAsOnnx(OnnxContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                for (int iinfo = 0; iinfo < _parent._columns.Length; ++iinfo)
                {
                    string inputColumnName = _parent._columns[iinfo].InputColumnName;
                    if (!ctx.ContainsColumn(inputColumnName))
                    {
                        ctx.RemoveColumn(_parent._columns[iinfo].Name, false);
                        continue;
                    }

                    if (!SaveAsOnnxCore(ctx, iinfo, ctx.GetVariableName(inputColumnName),
                        ctx.AddIntermediateVariable(_types[iinfo], _parent._columns[iinfo].Name)))
                    {
                        ctx.RemoveColumn(_parent._columns[iinfo].Name, true);
                    }
                }
            }

            private bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, string srcVariableName, string dstVariableName)
            {
                var opType = "CSharp";
                var node = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
                node.AddAttribute("type", LoaderSignature);
                node.AddAttribute("to", (byte)_parent._columns[iinfo].OutputKind);
                if (_parent._columns[iinfo].OutputKeyCount != null)
                {
                    var key = (KeyType)_types[iinfo].GetItemType();
                    node.AddAttribute("max", key.Count);
                }
                return true;
            }
        }
    }

    /// <summary>
    /// <see cref="TypeConvertingEstimator"/> converts underlying column types.
    /// The source and destination column types need to be compatible.
    /// </summary>
    public sealed class TypeConvertingEstimator : TrivialEstimator<TypeConvertingTransformer>
    {
        internal sealed class Defaults
        {
            public const DataKind DefaultOutputKind = DataKind.Single;
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary>
            /// Name of the column resulting from the transformation of <see cref="InputColumnName"/>.
            /// </summary>
            public readonly string Name;
            /// <summary>
            /// Name of column to transform. If set to <see langword="null"/>, the value of the <see cref="Name"/> will be used as source.
            /// </summary>
            public readonly string InputColumnName;
            /// <summary>
            /// The expected kind of the converted column.
            /// </summary>
            public readonly DataKind OutputKind;
            /// <summary>
            /// New key count, if we work with key type.
            /// </summary>
            public readonly KeyCount OutputKeyCount;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="outputKind">The expected kind of the converted column.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="outputKeyCount">New key count, if we work with key type.</param>
            public ColumnOptions(string name, DataKind outputKind, string inputColumnName, KeyCount outputKeyCount = null)
            {
                Name = name;
                InputColumnName = inputColumnName ?? name;
                OutputKind = outputKind;
                OutputKeyCount = outputKeyCount;
            }

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="type">The expected kind of the converted column.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="outputKeyCount">New key count, if we work with key type.</param>
            public ColumnOptions(string name, Type type, string inputColumnName, KeyCount outputKeyCount = null)
            {
                Name = name;
                InputColumnName = inputColumnName ?? name;
                if (!type.TryGetDataKind(out InternalDataKind OutputKind))
                    throw Contracts.ExceptUserArg(nameof(type), $"Unsupported type {type}.");
                this.OutputKind = OutputKind.ToDataKind();
                OutputKeyCount = outputKeyCount;
            }
        }

        /// <summary>
        /// Convinence constructor for simple one column case.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="outputKind">The expected kind of the converted column.</param>
        internal TypeConvertingEstimator(IHostEnvironment env,
            string outputColumnName, string inputColumnName = null,
            DataKind outputKind = Defaults.DefaultOutputKind)
            : this(env, new ColumnOptions(outputColumnName, outputKind, inputColumnName ?? outputColumnName))
        {
        }

        /// <summary>
        /// Create a <see cref="TypeConvertingEstimator"/> that takes multiple pairs of columns.
        /// </summary>
        internal TypeConvertingEstimator(IHostEnvironment env, params ColumnOptions[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TypeConvertingEstimator)), new TypeConvertingTransformer(env, columns))
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
                if (!inputSchema.TryFindColumn(colInfo.InputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);
                if (!TypeConvertingTransformer.GetNewType(Host, col.ItemType, colInfo.OutputKind.ToInternalDataKind(), colInfo.OutputKeyCount, out PrimitiveDataViewType newType))
                    throw Host.ExceptParam(nameof(inputSchema), $"Can't convert {colInfo.InputColumnName} into {newType.ToString()}");
                if (!Data.Conversion.Conversions.Instance.TryGetStandardConversion(col.ItemType, newType, out Delegate del, out bool identity))
                    throw Host.ExceptParam(nameof(inputSchema), $"Don't know how to convert {colInfo.InputColumnName} into {newType.ToString()}");
                var metadata = new List<SchemaShape.Column>();
                if (col.ItemType is BooleanDataViewType && newType is NumberDataViewType)
                    metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
                    if (col.Kind == SchemaShape.Column.VectorKind.Vector)
                        metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, slotMeta.ItemType, false));
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.KeyValues, out var keyMeta))
                    if (col.ItemType is KeyType)
                        metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector, keyMeta.ItemType, false));
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.IsNormalized, out var normMeta))
                    if (col.ItemType is NumberDataViewType && newType is NumberDataViewType)
                        metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector, normMeta.ItemType, false));
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, col.Kind, newType, false, col.Annotations);
            }
            return new SchemaShape(result.Values);
        }
    }
}
