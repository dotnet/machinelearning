// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms.Conversions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

[assembly: LoadableClass(KeyToBinaryVectorMappingTransformer.Summary, typeof(IDataTransform), typeof(KeyToBinaryVectorMappingTransformer), typeof(KeyToBinaryVectorMappingTransformer.Arguments), typeof(SignatureDataTransform),
    "Key To Binary Vector Transform", KeyToBinaryVectorMappingTransformer.UserName, "KeyToBinary", "ToBinaryVector", DocName = "transform/KeyToBinaryVectorTransform.md")]

[assembly: LoadableClass(KeyToBinaryVectorMappingTransformer.Summary, typeof(IDataTransform), typeof(KeyToBinaryVectorMappingTransformer), null, typeof(SignatureLoadDataTransform),
    "Key To Binary Vector Transform", KeyToBinaryVectorMappingTransformer.LoaderSignature)]

[assembly: LoadableClass(KeyToBinaryVectorMappingTransformer.Summary, typeof(KeyToBinaryVectorMappingTransformer), null, typeof(SignatureLoadModel),
    KeyToBinaryVectorMappingTransformer.UserName, KeyToBinaryVectorMappingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(KeyToBinaryVectorMappingTransformer), null, typeof(SignatureLoadRowMapper),
   KeyToBinaryVectorMappingTransformer.UserName, KeyToBinaryVectorMappingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Conversions
{
    public sealed class KeyToBinaryVectorMappingTransformer : OneToOneTransformerBase
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
                ShortName = "col", SortOrder = 1)]
            public KeyToVectorMappingTransformer.Column[] Column;
        }
        public class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;

            public ColumnInfo(string input, string output)
            {
                Input = input;
                Output = output;
            }
        }

        internal const string Summary = "Converts a key column to a binary encoded vector.";
        public const string UserName = "KeyToBinaryVectorTransform";
        public const string LoaderSignature = "KeyToBinaryTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "KEY2BINR",
                verWrittenCur: 0x00000001, // Initial
                verReadableCur: 0x00000001,
                verWeCanReadBack: 0x00000001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(KeyToBinaryVectorMappingTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "KeyToBinary";

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        public IReadOnlyCollection<ColumnInfo> Columns => _columns.AsReadOnly();
        private readonly ColumnInfo[] _columns;

        private string TestIsKey(ColumnType type)
        {
            if (type.ItemType.KeyCount > 0)
                return null;
            return "key type of known cardinality";
        }

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema.GetColumnType(srcCol);
            string reason = TestIsKey(type);
            if (reason != null)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, reason, type.ToString());
        }

        public KeyToBinaryVectorMappingTransformer(IHostEnvironment env, params ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();

        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            SaveColumns(ctx);
        }

        // Factory method for SignatureLoadModel.
        private static KeyToBinaryVectorMappingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new KeyToBinaryVectorMappingTransformer(host, ctx);
        }

        private KeyToBinaryVectorMappingTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            _columns = new ColumnInfo[ColumnPairs.Length];
            for (int i = 0; i < ColumnPairs.Length; i++)
                _columns[i] = new ColumnInfo(ColumnPairs[i].input, ColumnPairs[i].output);
        }

        public static IDataTransform Create(IHostEnvironment env, IDataView input, params ColumnInfo[] columns) =>
            new KeyToBinaryVectorMappingTransformer(env, columns).MakeDataTransform(input);

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new ColumnInfo[args.Column.Length];
            using (var ch = env.Start("ValidateArgs"))
            {
                for (int i = 0; i < cols.Length; i++)
                {
                    var item = args.Column[i];
                    cols[i] = new ColumnInfo(item.Source ?? item.Name, item.Name);
                };
            }
            return new KeyToBinaryVectorMappingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(Schema.Create(inputSchema));

        protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        private sealed class Mapper : MapperBase
        {
            private sealed class ColInfo
            {
                public readonly string Name;
                public readonly string Source;
                public readonly ColumnType TypeSrc;

                public ColInfo(string name, string source, ColumnType type)
                {
                    Name = name;
                    Source = source;
                    TypeSrc = type;
                }
            }

            private readonly KeyToBinaryVectorMappingTransformer _parent;
            private readonly ColInfo[] _infos;
            private readonly VectorType[] _types;
            private readonly int[] _bitsPerKey;

            public Mapper(KeyToBinaryVectorMappingTransformer parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _infos = CreateInfos(inputSchema);
                _types = new VectorType[_parent.ColumnPairs.Length];
                _bitsPerKey = new int[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    //Add an additional bit for all 1s to represent missing values.
                    _bitsPerKey[i] = Utils.IbitHigh((uint)_infos[i].TypeSrc.ItemType.KeyCount) + 2;
                    Host.Assert(_bitsPerKey[i] > 0);
                    if (_infos[i].TypeSrc.ValueCount == 1)
                        // Output is a single vector computed as the sum of the output indicator vectors.
                        _types[i] = new VectorType(NumberType.Float, _bitsPerKey[i]);
                    else
                        // Output is the concatenation of the multiple output indicator vectors.
                        _types[i] = new VectorType(NumberType.Float, _infos[i].TypeSrc.ValueCount, _bitsPerKey[i]);
                }
            }
            private ColInfo[] CreateInfos(ISchema inputSchema)
            {
                Host.AssertValue(inputSchema);
                var infos = new ColInfo[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colSrc))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].input);
                    var type = inputSchema.GetColumnType(colSrc);
                    _parent.CheckInputColumn(inputSchema, i, colSrc);
                    infos[i] = new ColInfo(_parent.ColumnPairs[i].output, _parent.ColumnPairs[i].input, type);
                }
                return infos;
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var builder = new Schema.Metadata.Builder();
                    AddMetadata(i, builder);

                    result[i] = new Schema.Column(_parent.ColumnPairs[i].output, _types[i], builder.GetMetadata());
                }
                return result;
            }

            private void AddMetadata(int iinfo, Schema.Metadata.Builder builder)
            {
                InputSchema.TryGetColumnIndex(_infos[iinfo].Source, out int srcCol);
                var inputMetadata = InputSchema[srcCol].Metadata;
                var srcType = _infos[iinfo].TypeSrc;
                // See if the source has key names.

                ColumnType typeNames = null;
                int metaKeyValuesCol = 0;
                if (inputMetadata.Schema.TryGetColumnIndex(MetadataUtils.Kinds.KeyValues, out metaKeyValuesCol))
                    typeNames = inputMetadata.Schema[metaKeyValuesCol].Type;
                if (typeNames == null || !typeNames.IsKnownSizeVector || !typeNames.ItemType.IsText ||
                    typeNames.VectorSize != _infos[iinfo].TypeSrc.ItemType.KeyCount)
                {
                    typeNames = null;
                }

                if (_infos[iinfo].TypeSrc.ValueCount == 1)
                {
                    if (typeNames != null)
                    {
                        ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                        {
                            GenerateBitSlotName(iinfo, ref dst);
                        };

                        var slotNamesType = new VectorType(TextType.Instance, _types[iinfo]);
                        builder.AddSlotNames(slotNamesType.VectorSize, getter);
                    }

                    ValueGetter<bool> normalizeGetter = (ref bool dst) =>
                    {
                        dst = true;
                    };
                    builder.Add(new Schema.Column(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, null), normalizeGetter);
                }
                else
                {
                    if (typeNames != null && _types[iinfo].IsKnownSizeVector)
                    {
                        ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                        {
                            GetSlotNames(iinfo, ref dst);
                        };
                        var slotNamesType = new VectorType(TextType.Instance, _types[iinfo]);
                        builder.Add(new Schema.Column(MetadataUtils.Kinds.SlotNames, slotNamesType, null), getter);
                    }
                }
            }

            private void GenerateBitSlotName(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                const string slotNamePrefix = "Bit";
                var bldr = new BufferBuilder<ReadOnlyMemory<char>>(TextCombiner.Instance);
                bldr.Reset(_bitsPerKey[iinfo], true);
                for (int i = 0; i < _bitsPerKey[iinfo]; i++)
                    bldr.AddFeature(i, (slotNamePrefix + (_bitsPerKey[iinfo] - i - 1)).AsMemory());

                bldr.GetResult(ref dst);
            }

            private void GetSlotNames(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);
                Host.Assert(_types[iinfo].IsKnownSizeVector);

                // Variable size should have thrown (by the caller).
                var typeSrc = _infos[iinfo].TypeSrc;
                Host.Assert(typeSrc.VectorSize > 1);

                // Get the source slot names, defaulting to empty text.
                var namesSlotSrc = default(VBuffer<ReadOnlyMemory<char>>);

                var inputMetadata = InputSchema[_infos[iinfo].Source].Metadata;
                ColumnType typeSlotSrc = null;
                if (inputMetadata != null)
                    typeSlotSrc = inputMetadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.SlotNames)?.Type;
                if (typeSlotSrc != null && typeSlotSrc.VectorSize == typeSrc.VectorSize && typeSlotSrc.ItemType.IsText)
                {
                    inputMetadata.GetValue(MetadataUtils.Kinds.SlotNames, ref namesSlotSrc);
                    Host.Check(namesSlotSrc.Length == typeSrc.VectorSize);
                }
                else
                    namesSlotSrc = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(typeSrc.VectorSize);

                int slotLim = _types[iinfo].VectorSize;
                Host.Assert(slotLim == (long)typeSrc.VectorSize * _bitsPerKey[iinfo]);

                var values = dst.Values;
                if (Utils.Size(values) < slotLim)
                    values = new ReadOnlyMemory<char>[slotLim];

                var sb = new StringBuilder();
                int slot = 0;
                VBuffer<ReadOnlyMemory<char>> bits = default;
                GenerateBitSlotName(iinfo, ref bits);
                foreach (var kvpSlot in namesSlotSrc.Items(all: true))
                {
                    Contracts.Assert(slot == (long)kvpSlot.Key * _bitsPerKey[iinfo]);
                    sb.Clear();
                    if (!kvpSlot.Value.IsEmpty)
                        sb.AppendMemory(kvpSlot.Value);
                    else
                        sb.Append('[').Append(kvpSlot.Key).Append(']');
                    sb.Append('.');

                    int len = sb.Length;
                    foreach (var key in bits.Values)
                    {
                        sb.Length = len;
                        sb.AppendMemory(key);
                        values[slot++] = sb.ToString().AsMemory();
                    }
                }
                Host.Assert(slot == slotLim);

                dst = new VBuffer<ReadOnlyMemory<char>>(slotLim, values, dst.Indices);
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);
                disposer = null;

                var info = _infos[iinfo];
                if (!info.TypeSrc.IsVector)
                    return MakeGetterOne(input, iinfo);
                return MakeGetterInd(input, iinfo);
            }

            /// <summary>
            /// This is for the scalar case.
            /// </summary>
            private ValueGetter<VBuffer<float>> MakeGetterOne(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(_infos[iinfo].TypeSrc.IsKey);

                int bitsPerKey = _bitsPerKey[iinfo];
                Host.Assert(bitsPerKey == _types[iinfo].VectorSize);

                int dstLength = _types[iinfo].VectorSize;
                Host.Assert(dstLength > 0);
                input.Schema.TryGetColumnIndex(_infos[iinfo].Source, out int srcCol);
                Host.Assert(srcCol >= 0);
                var getSrc = RowCursorUtils.GetGetterAs<uint>(NumberType.U4, input, srcCol);
                var src = default(uint);
                var bldr = new BufferBuilder<float>(R4Adder.Instance);
                return
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref src);
                        bldr.Reset(bitsPerKey, false);
                        EncodeValueToBinary(bldr, src, bitsPerKey, 0);
                        bldr.GetResult(ref dst);

                        Contracts.Assert(dst.Length == bitsPerKey);
                    };
            }

            /// <summary>
            /// This is for the indicator case - vector input and outputs should be concatenated.
            /// </summary>
            private ValueGetter<VBuffer<float>> MakeGetterInd(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(_infos[iinfo].TypeSrc.IsVector);
                Host.Assert(_infos[iinfo].TypeSrc.ItemType.IsKey);

                int cv = _infos[iinfo].TypeSrc.VectorSize;
                Host.Assert(cv >= 0);
                input.Schema.TryGetColumnIndex(_infos[iinfo].Source, out int srcCol);
                Host.Assert(srcCol >= 0);
                var getSrc = RowCursorUtils.GetVecGetterAs<uint>(NumberType.U4, input, srcCol);
                var src = default(VBuffer<uint>);
                var bldr = new BufferBuilder<float>(R4Adder.Instance);
                int bitsPerKey = _bitsPerKey[iinfo];
                return
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref src);
                        Host.Check(src.Length == cv || cv == 0);
                        bldr.Reset(src.Length * bitsPerKey, false);

                        int index = 0;
                        foreach (uint value in src.DenseValues())
                        {
                            EncodeValueToBinary(bldr, value, bitsPerKey, index * bitsPerKey);
                            index++;
                        }

                        bldr.GetResult(ref dst);

                        Contracts.Assert(dst.Length == src.Length * bitsPerKey);
                    };
            }

            private void EncodeValueToBinary(BufferBuilder<float> bldr, uint value, int bitsToConsider, int startIndex)
            {
                Contracts.Assert(0 < bitsToConsider && bitsToConsider <= sizeof(uint) * 8);
                Contracts.Assert(startIndex >= 0);

                //Treat missing values, zero, as a special value of all 1s.
                value--;
                while (bitsToConsider > 0)
                    bldr.AddFeature(startIndex++, (value >> --bitsToConsider) & 1U);
            }
        }
    }

    public sealed class KeyToBinaryVectorMappingEstimator : TrivialEstimator<KeyToBinaryVectorMappingTransformer>
    {

        public KeyToBinaryVectorMappingEstimator(IHostEnvironment env, params KeyToBinaryVectorMappingTransformer.ColumnInfo[] columns)
            : this(env, new KeyToBinaryVectorMappingTransformer(env, columns))
        {
        }

        public KeyToBinaryVectorMappingEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null)
            : this(env, new KeyToBinaryVectorMappingTransformer(env, new KeyToBinaryVectorMappingTransformer.ColumnInfo(inputColumn, outputColumn ?? inputColumn)))
        {
        }

        private KeyToBinaryVectorMappingEstimator(IHostEnvironment env, KeyToBinaryVectorMappingTransformer transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(KeyToBinaryVectorMappingEstimator)), transformer)
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if ((col.ItemType.ItemType.RawKind == default) || !(col.ItemType.IsVector || col.ItemType.IsPrimitive))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);

                var metadata = new List<SchemaShape.Column>();
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.KeyValues, out var keyMeta))
                    if (col.Kind != SchemaShape.Column.VectorKind.VariableVector && keyMeta.ItemType.IsText)
                        metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, keyMeta.ItemType, false));
                if (col.Kind == SchemaShape.Column.VectorKind.Scalar)
                    metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false));
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, new SchemaShape(metadata));
            }

            return new SchemaShape(result.Values);
        }
    }
    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class KeyToBinaryVectorExtensions
    {
        private interface IColInput
        {
            PipelineColumn Input { get; }
        }

        private sealed class OutVectorColumn<TKey, TValue> : Vector<float>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVectorColumn(Vector<Key<TKey, TValue>> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }

            public OutVectorColumn(Key<TKey, TValue> input)
              : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVarVectorColumn<TKey, TValue> : VarVector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public OutVarVectorColumn(VarVector<Key<TKey, TValue>> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVectorColumn<TKey> : Vector<float>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVectorColumn(Vector<Key<TKey>> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }

            public OutVectorColumn(Key<TKey> input)
              : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVarVectorColumn<TKey> : VarVector<float>, IColInput
        {
            public PipelineColumn Input { get; }
            public OutVarVectorColumn(VarVector<Key<TKey>> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            public static Reconciler Inst = new Reconciler();

            private Reconciler() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var infos = new KeyToBinaryVectorMappingTransformer.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (IColInput)toOutput[i];
                    infos[i] = new KeyToBinaryVectorMappingTransformer.ColumnInfo(inputNames[col.Input], outputNames[toOutput[i]]);
                }
                return new KeyToBinaryVectorMappingEstimator(env, infos);
            }
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static Vector<float> ToBinaryVector<TKey, TValue>(this Key<TKey, TValue> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static Vector<float> ToBinaryVector<TKey, TValue>(this Vector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static VarVector<float> ToBinaryVector<TKey, TValue>(this VarVector<Key<TKey, TValue>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<TKey, TValue>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static Vector<float> ToBinaryVector<TKey>(this Key<TKey> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static Vector<float> ToBinaryVector<TKey>(this Vector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<TKey>(input);
        }

        /// <summary>
        /// Takes a column of key type of known cardinality and produces a vector of bits representing the key in binary form.
        /// The first value is encoded as all zeros and missing values are encoded as all ones.
        /// In the case where a vector has multiple keys, the encoded values are concatenated.
        /// Number of bits per key is determined as the number of bits needed to represent the cardinality of the keys plus one.
        /// </summary>
        public static VarVector<float> ToBinaryVector<TKey>(this VarVector<Key<TKey>> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<TKey>(input);
        }
    }
}
