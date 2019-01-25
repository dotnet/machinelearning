// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.Onnx;
using Microsoft.ML.Transforms.Conversions;

[assembly: LoadableClass(TypeConvertingTransformer.Summary, typeof(IDataTransform), typeof(TypeConvertingTransformer), typeof(TypeConvertingTransformer.Arguments), typeof(SignatureDataTransform),
    TypeConvertingTransformer.UserName, TypeConvertingTransformer.ShortName, "ConvertTransform", DocName = "transform/ConvertTransform.md")]

[assembly: LoadableClass(TypeConvertingTransformer.Summary, typeof(IDataTransform), typeof(TypeConvertingTransformer), null, typeof(SignatureLoadDataTransform),
    TypeConvertingTransformer.UserName, TypeConvertingTransformer.LoaderSignature, TypeConvertingTransformer.LoaderSignatureOld)]

[assembly: LoadableClass(TypeConvertingTransformer.Summary, typeof(TypeConvertingTransformer), null, typeof(SignatureLoadModel),
    TypeConvertingTransformer.UserName, TypeConvertingTransformer.LoaderSignature)]

[assembly: LoadableClass(TypeConvertingTransformer.Summary, typeof(IRowMapper), typeof(TypeConvertingTransformer), null, typeof(SignatureLoadRowMapper),
    TypeConvertingTransformer.UserName, TypeConvertingTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(TypeConversion))]

namespace Microsoft.ML.Transforms.Conversions
{
    public static class TypeConversion
    {
        [TlcModule.EntryPoint(Name = "Transforms.ColumnTypeConverter", Desc = TypeConvertingTransformer.Summary, UserName = TypeConvertingTransformer.UserName, ShortName = TypeConvertingTransformer.ShortName)]
        public static CommonOutputs.TransformOutput Convert(IHostEnvironment env, TypeConvertingTransformer.Arguments input)
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
    /// ConvertTransform allow to change underlying column type as long as we know how to convert types.
    /// </summary>
    public sealed class TypeConvertingTransformer : OneToOneTransformerBase
    {
        public class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The result type", ShortName = "type")]
            public DataKind? ResultType;

            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the cardinality/count of valid key values", ShortName = "key", Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public KeyCount KeyCount;

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
                if (!base.TryParse(str, out string extra))
                    return false;
                if (extra == null)
                    return true;

                if (!TypeParsingUtils.TryParseDataKind(extra, out DataKind kind, out KeyCount))
                    return false;
                ResultType = kind == default ? default(DataKind?) : kind;
                return true;
            }

            public bool TryUnparse(StringBuilder sb)
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

        public class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:type:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The result type", ShortName = "type", SortOrder = 2)]
            public DataKind? ResultType;

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
                verWeCanReadBack: 0x00010003,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld,
                loaderAssemblyName: typeof(TypeConvertingTransformer).Assembly.FullName);
        }
        private const uint VersionNoMinCount = 0x00010004;
        private const string RegistrationName = "Convert";

        public IReadOnlyCollection<ColumnInfo> Columns => _columns.AsReadOnly();

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly DataKind OutputKind;
            public readonly KeyCount OutputKeyCount;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            /// <param name="outputKind">The expected kind of the converted column.</param>
            /// <param name="outputKeyCount">New key count, if we work with key type.</param>
            public ColumnInfo(string input, string output, DataKind outputKind, KeyCount outputKeyCount = null)
            {
                Input = input;
                Output = output;
                OutputKind = outputKind;
                OutputKeyCount = outputKeyCount;
            }
        }

        private readonly ColumnInfo[] _columns;

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckNonEmpty(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        /// <summary>
        /// Convinence constructor for simple one column case.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the output column.</param>
        /// <param name="outputColumn">Name of the column to be transformed. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="outputKind">The expected type of the converted column.</param>
        /// <param name="outputKeyCount">New key count if we work with key type.</param>
        public TypeConvertingTransformer(IHostEnvironment env, string inputColumn, string outputColumn, DataKind outputKind, KeyCount outputKeyCount = null)
            : this(env, new ColumnInfo(inputColumn, outputColumn, outputKind, outputKeyCount))
        {
        }

        /// <summary>
        /// Create a <see cref="TypeConvertingTransformer"/> that takes multiple pairs of columns.
        /// </summary>
        public TypeConvertingTransformer(IHostEnvironment env, params ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TypeConvertingTransformer)), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        public override void Save(ModelSaveContext ctx)
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
                Host.Assert((DataKind)(byte)_columns[i].OutputKind == _columns[i].OutputKind);
                if (_columns[i].OutputKeyCount != null)
                {
                    byte b = (byte)_columns[i].OutputKind;
                    b |= 0x80;
                    ctx.Writer.Write(b);
                    ctx.Writer.Write(_columns[i].OutputKeyCount.Count ?? _columns[i].OutputKind.ToMaxInt());
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

            _columns = new ColumnInfo[columnsLength];
            for (int i = 0; i < columnsLength; i++)
            {
                byte b = ctx.Reader.ReadByte();
                var kind = (DataKind)(b & 0x7F);
                Host.CheckDecode(Enum.IsDefined(typeof(DataKind), kind));
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
                _columns[i] = new ColumnInfo(ColumnPairs[i].input, ColumnPairs[i].output, kind, keyCount);
            }
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new ColumnInfo[args.Column.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = args.Column[i];
                var tempResultType = item.ResultType ?? args.ResultType;
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
                    if (args.KeyCount != null)
                        keyCount = args.KeyCount;
                    else if (args.Range != null)
                        keyCount = KeyCount.Parse(args.Range);
                }

                DataKind kind;
                if (tempResultType == null)
                {
                    if (keyCount == null)
                        kind = DataKind.Num;
                    else
                    {
                        var srcType = input.Schema[item.Source ?? item.Name].Type;
                        kind = srcType is KeyType ? srcType.GetRawKind() : DataKind.U8;
                    }
                }
                else
                {
                    kind = tempResultType.Value;
                }

                cols[i] = new ColumnInfo(item.Source ?? item.Name, item.Name, kind, keyCount);
            };
            return new TypeConvertingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        internal static bool GetNewType(IExceptionContext ectx, ColumnType srcType, DataKind kind, KeyCount keyCount, out PrimitiveType itemType)
        {
            if (keyCount != null)
            {
                itemType = TypeParsingUtils.ConstructKeyType(kind, keyCount);
                ColumnType srcItemType = srcType.GetItemType();
                if (!(srcItemType is KeyType) && !(srcItemType is TextType))
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
            private readonly ColumnType[] _types;
            private readonly int[] _srcCols;

            public bool CanSaveOnnx(OnnxContext ctx) => ctx.GetOnnxVersion() == OnnxVersion.Experimental;

            public Mapper(TypeConvertingTransformer parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new ColumnType[_parent._columns.Length];
                _srcCols = new int[_parent._columns.Length];
                for (int i = 0; i < _parent._columns.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out _srcCols[i]);
                    var srcCol = inputSchema[_srcCols[i]];
                    if (!CanConvertToType(Host, srcCol.Type, _parent._columns[i].OutputKind, _parent._columns[i].OutputKeyCount, out PrimitiveType itemType, out _types[i]))
                    {
                        throw Host.ExceptParam(nameof(inputSchema),
                        "source column '{0}' with item type '{1}' is not compatible with destination type '{2}'",
                        _parent._columns[i].Input, srcCol.Type, itemType);
                    }
                }
            }

            private static bool CanConvertToType(IExceptionContext ectx, ColumnType srcType, DataKind kind, KeyCount keyCount,
                out PrimitiveType itemType, out ColumnType typeDst)
            {
                ectx.AssertValue(srcType);
                ectx.Assert(Enum.IsDefined(typeof(DataKind), kind));

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

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new Schema.DetachedColumn[_parent._columns.Length];
                for (int i = 0; i < _parent._columns.Length; i++)
                {
                    var builder = new MetadataBuilder();
                    var srcType = InputSchema[_srcCols[i]].Type;
                    if (_types[i].IsKnownSizeVector())
                        builder.Add(InputSchema[ColMapNewToOld[i]].Metadata, name => name == MetadataUtils.Kinds.SlotNames);

                    ColumnType srcItemType = srcType.GetItemType();
                    ColumnType currentItemType = _types[i].GetItemType();

                    KeyType srcItemKeyType = srcItemType as KeyType;
                    KeyType currentItemKeyType = currentItemType as KeyType;
                    if (srcItemKeyType != null && currentItemKeyType != null &&
                        srcItemKeyType.Count > 0 && srcItemKeyType.Count == currentItemKeyType.Count)
                    {
                        builder.Add(InputSchema[ColMapNewToOld[i]].Metadata, name => name == MetadataUtils.Kinds.KeyValues);
                    }

                    if (srcItemType is NumberType && currentItemType is NumberType)
                        builder.Add(InputSchema[ColMapNewToOld[i]].Metadata, name => name == MetadataUtils.Kinds.IsNormalized);
                    if (srcType is BoolType && currentItemType is NumberType)
                    {
                        ValueGetter<bool> getter = (ref bool dst) => dst = true;
                        builder.Add(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, getter);
                    }
                    result[i] = new Schema.DetachedColumn(_parent._columns[i].Output, _types[i], builder.GetMetadata());
                }
                return result;
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
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
                    string sourceColumnName = _parent._columns[iinfo].Input;
                    if (!ctx.ContainsColumn(sourceColumnName))
                    {
                        ctx.RemoveColumn(_parent._columns[iinfo].Output, false);
                        continue;
                    }

                    if (!SaveAsOnnxCore(ctx, iinfo, ctx.GetVariableName(sourceColumnName),
                        ctx.AddIntermediateVariable(_types[iinfo], _parent._columns[iinfo].Output)))
                    {
                        ctx.RemoveColumn(_parent._columns[iinfo].Output, true);
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
    /// Convert estimator allow you take column and change it type as long as we know how to do conversion between types.
    /// </summary>
    public sealed class TypeConvertingEstimator : TrivialEstimator<TypeConvertingTransformer>
    {
        internal sealed class Defaults
        {
            public const DataKind DefaultOutputKind = DataKind.R4;
        }

        /// <summary>
        /// Convinence constructor for simple one column case.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the output column.</param>
        /// <param name="outputKind">The expected type of the converted column.</param>
        public TypeConvertingEstimator(IHostEnvironment env,
            string inputColumn, string outputColumn = null,
            DataKind outputKind = Defaults.DefaultOutputKind)
            : this(env, new TypeConvertingTransformer.ColumnInfo(inputColumn, outputColumn ?? inputColumn, outputKind))
        {
        }

        /// <summary>
        /// Create a <see cref="TypeConvertingEstimator"/> that takes multiple pairs of columns.
        /// </summary>
        public TypeConvertingEstimator(IHostEnvironment env, params TypeConvertingTransformer.ColumnInfo[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TypeConvertingEstimator)), new TypeConvertingTransformer(env, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if (!TypeConvertingTransformer.GetNewType(Host, col.ItemType, colInfo.OutputKind, colInfo.OutputKeyCount, out PrimitiveType newType))
                    throw Host.ExceptParam(nameof(inputSchema), $"Can't convert {colInfo.Input} into {newType.ToString()}");
                if (!Data.Conversion.Conversions.Instance.TryGetStandardConversion(col.ItemType, newType, out Delegate del, out bool identity))
                    throw Host.ExceptParam(nameof(inputSchema), $"Don't know how to convert {colInfo.Input} into {newType.ToString()}");
                var metadata = new List<SchemaShape.Column>();
                if (col.ItemType is BoolType && newType is NumberType)
                    metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false));
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    if (col.Kind == SchemaShape.Column.VectorKind.Vector)
                        metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, slotMeta.ItemType, false));
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.KeyValues, out var keyMeta))
                    if (col.ItemType is KeyType)
                        metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector, keyMeta.ItemType, false));
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.IsNormalized, out var normMeta))
                    if (col.ItemType is NumberType && newType is NumberType)
                        metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector, normMeta.ItemType, false));
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, col.Kind, newType, false, col.Metadata);
            }
            return new SchemaShape(result.Values);
        }
    }
}
