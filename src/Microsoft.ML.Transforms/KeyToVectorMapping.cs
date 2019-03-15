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
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(KeyToBinaryVectorMappingTransformer.Summary, typeof(IDataTransform), typeof(KeyToBinaryVectorMappingTransformer), typeof(KeyToBinaryVectorMappingTransformer.Options), typeof(SignatureDataTransform),
    "Key To Binary Vector Transform", KeyToBinaryVectorMappingTransformer.UserName, "KeyToBinary", "ToBinaryVector", DocName = "transform/KeyToBinaryVectorTransform.md")]

[assembly: LoadableClass(KeyToBinaryVectorMappingTransformer.Summary, typeof(IDataTransform), typeof(KeyToBinaryVectorMappingTransformer), null, typeof(SignatureLoadDataTransform),
    "Key To Binary Vector Transform", KeyToBinaryVectorMappingTransformer.LoaderSignature)]

[assembly: LoadableClass(KeyToBinaryVectorMappingTransformer.Summary, typeof(KeyToBinaryVectorMappingTransformer), null, typeof(SignatureLoadModel),
    KeyToBinaryVectorMappingTransformer.UserName, KeyToBinaryVectorMappingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(KeyToBinaryVectorMappingTransformer), null, typeof(SignatureLoadRowMapper),
   KeyToBinaryVectorMappingTransformer.UserName, KeyToBinaryVectorMappingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    ///  Converts the key types back to binary vectors.
    /// </summary>
    public sealed class KeyToBinaryVectorMappingTransformer : OneToOneTransformerBase
    {
        internal sealed class Options
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public KeyToVectorMappingTransformer.Column[] Columns;
        }

        internal const string Summary = "Converts a key column to a binary encoded vector.";
        internal const string UserName = "KeyToBinaryVectorTransform";
        internal const string LoaderSignature = "KeyToBinaryTransform";

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

        /// <summary>
        /// The names of the output and input column pairs on which the transformation is performed.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        private string TestIsKey(DataViewType type)
        {
            if (type.GetItemType().GetKeyCount() > 0)
                return null;
            return "key type of known cardinality";
        }

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema[srcCol].Type;
            string reason = TestIsKey(type);
            if (reason != null)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, reason, type.ToString());
        }

        internal KeyToBinaryVectorMappingTransformer(IHostEnvironment env, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
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
        }

        private static IDataTransform Create(IHostEnvironment env, IDataView input, params (string outputColumnName, string inputColumnName)[] columns) =>
            new KeyToBinaryVectorMappingTransformer(env, columns).MakeDataTransform(input);

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new (string outputColumnName, string inputColumnName)[options.Columns.Length];
            using (var ch = env.Start("ValidateArgs"))
            {
                for (int i = 0; i < cols.Length; i++)
                {
                    var item = options.Columns[i];
                    cols[i] = (item.Name, item.Source ?? item.Name);
                };
            }
            return new KeyToBinaryVectorMappingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private sealed class ColInfo
            {
                public readonly string Name;
                public readonly string InputColumnName;
                public readonly DataViewType TypeSrc;

                public ColInfo(string name, string inputColumnName, DataViewType type)
                {
                    Name = name;
                    InputColumnName = inputColumnName;
                    TypeSrc = type;
                }
            }

            private readonly KeyToBinaryVectorMappingTransformer _parent;
            private readonly ColInfo[] _infos;
            private readonly VectorType[] _types;
            private readonly int[] _bitsPerKey;

            public Mapper(KeyToBinaryVectorMappingTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _infos = CreateInfos(inputSchema);
                _types = new VectorType[_parent.ColumnPairs.Length];
                _bitsPerKey = new int[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    //Add an additional bit for all 1s to represent missing values.
                    _bitsPerKey[i] = Utils.IbitHigh((uint)_infos[i].TypeSrc.GetItemType().GetKeyCount()) + 2;
                    Host.Assert(_bitsPerKey[i] > 0);
                    int srcValueCount = _infos[i].TypeSrc.GetValueCount();
                    if (srcValueCount == 1)
                        // Output is a single vector computed as the sum of the output indicator vectors.
                        _types[i] = new VectorType(NumberDataViewType.Single, _bitsPerKey[i]);
                    else
                        // Output is the concatenation of the multiple output indicator vectors.
                        _types[i] = new VectorType(NumberDataViewType.Single, srcValueCount, _bitsPerKey[i]);
                }
            }
            private ColInfo[] CreateInfos(DataViewSchema inputSchema)
            {
                Host.AssertValue(inputSchema);
                var infos = new ColInfo[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colSrc))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName);
                    var type = inputSchema[colSrc].Type;
                    _parent.CheckInputColumn(inputSchema, i, colSrc);
                    infos[i] = new ColInfo(_parent.ColumnPairs[i].outputColumnName, _parent.ColumnPairs[i].inputColumnName, type);
                }
                return infos;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var builder = new DataViewSchema.Annotations.Builder();
                    AddMetadata(i, builder);

                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _types[i], builder.ToAnnotations());
                }
                return result;
            }

            private void AddMetadata(int iinfo, DataViewSchema.Annotations.Builder builder)
            {
                InputSchema.TryGetColumnIndex(_infos[iinfo].InputColumnName, out int srcCol);
                var inputMetadata = InputSchema[srcCol].Annotations;
                var srcType = _infos[iinfo].TypeSrc;
                // See if the source has key names.

                VectorType typeNames = null;
                int metaKeyValuesCol = 0;
                if (inputMetadata.Schema.TryGetColumnIndex(AnnotationUtils.Kinds.KeyValues, out metaKeyValuesCol))
                    typeNames = inputMetadata.Schema[metaKeyValuesCol].Type as VectorType;
                if (typeNames == null || !typeNames.IsKnownSize || !(typeNames.ItemType is TextDataViewType) ||
                    typeNames.Size != _infos[iinfo].TypeSrc.GetItemType().GetKeyCountAsInt32(Host))
                {
                    typeNames = null;
                }

                if (_infos[iinfo].TypeSrc.GetValueCount() == 1)
                {
                    if (typeNames != null)
                    {
                        ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                        {
                            GenerateBitSlotName(iinfo, ref dst);
                        };

                        var slotNamesType = new VectorType(TextDataViewType.Instance, _types[iinfo]);
                        builder.AddSlotNames(slotNamesType.Size, getter);
                    }

                    ValueGetter<bool> normalizeGetter = (ref bool dst) =>
                    {
                        dst = true;
                    };
                    builder.Add(AnnotationUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, normalizeGetter);
                }
                else
                {
                    if (typeNames != null && _types[iinfo].IsKnownSize)
                    {
                        ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                        {
                            GetSlotNames(iinfo, ref dst);
                        };
                        var slotNamesType = new VectorType(TextDataViewType.Instance, _types[iinfo]);
                        builder.Add(AnnotationUtils.Kinds.SlotNames, slotNamesType, getter);
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
                Host.Assert(_types[iinfo].IsKnownSize);

                // Variable size should have thrown (by the caller).
                var typeSrc = _infos[iinfo].TypeSrc;
                var srcVectorSize = typeSrc.GetVectorSize();
                Host.Assert(srcVectorSize > 1);

                // Get the source slot names, defaulting to empty text.
                var namesSlotSrc = default(VBuffer<ReadOnlyMemory<char>>);

                var inputMetadata = InputSchema[_infos[iinfo].InputColumnName].Annotations;
                VectorType typeSlotSrc = null;
                if (inputMetadata != null)
                    typeSlotSrc = inputMetadata.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames)?.Type as VectorType;
                if (typeSlotSrc != null && typeSlotSrc.Size == srcVectorSize && typeSlotSrc.ItemType is TextDataViewType)
                {
                    inputMetadata.GetValue(AnnotationUtils.Kinds.SlotNames, ref namesSlotSrc);
                    Host.Check(namesSlotSrc.Length == srcVectorSize);
                }
                else
                    namesSlotSrc = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(srcVectorSize);

                int slotLim = _types[iinfo].Size;
                Host.Assert(slotLim == (long)srcVectorSize * _bitsPerKey[iinfo]);

                var editor = VBufferEditor.Create(ref dst, slotLim);

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
                    foreach (var key in bits.GetValues())
                    {
                        sb.Length = len;
                        sb.AppendMemory(key);
                        editor.Values[slot++] = sb.ToString().AsMemory();
                    }
                }
                Host.Assert(slot == slotLim);

                dst = editor.Commit();
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);
                disposer = null;

                var info = _infos[iinfo];
                if (!(info.TypeSrc is VectorType vectorType))
                    return MakeGetterOne(input, iinfo);
                return MakeGetterInd(input, iinfo, vectorType);
            }

            /// <summary>
            /// This is for the scalar case.
            /// </summary>
            private ValueGetter<VBuffer<float>> MakeGetterOne(DataViewRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(_infos[iinfo].TypeSrc is KeyType);

                int bitsPerKey = _bitsPerKey[iinfo];
                Host.Assert(bitsPerKey == _types[iinfo].Size);

                int dstLength = _types[iinfo].Size;
                Host.Assert(dstLength > 0);
                input.Schema.TryGetColumnIndex(_infos[iinfo].InputColumnName, out int srcCol);
                Host.Assert(srcCol >= 0);
                var getSrc = RowCursorUtils.GetGetterAs<uint>(NumberDataViewType.UInt32, input, srcCol);
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
            private ValueGetter<VBuffer<float>> MakeGetterInd(DataViewRow input, int iinfo, VectorType typeSrc)
            {
                Host.AssertValue(input);
                Host.AssertValue(typeSrc);
                Host.Assert(typeSrc.ItemType is KeyType);

                int cv = typeSrc.Size;
                Host.Assert(cv >= 0);
                input.Schema.TryGetColumnIndex(_infos[iinfo].InputColumnName, out int srcCol);
                Host.Assert(srcCol >= 0);
                var getSrc = RowCursorUtils.GetVecGetterAs<uint>(NumberDataViewType.UInt32, input, srcCol);
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

    /// <summary>
    ///  Converts the key types back to binary vectors.
    /// </summary>
    public sealed class KeyToBinaryVectorMappingEstimator : TrivialEstimator<KeyToBinaryVectorMappingTransformer>
    {
        internal KeyToBinaryVectorMappingEstimator(IHostEnvironment env, params (string outputColumnName, string inputColumnName)[] columns)
            : this(env, new KeyToBinaryVectorMappingTransformer(env, columns))
        {
        }

        internal KeyToBinaryVectorMappingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null)
            : this(env, new KeyToBinaryVectorMappingTransformer(env, (outputColumnName, inputColumnName ?? outputColumnName)))
        {
        }

        private KeyToBinaryVectorMappingEstimator(IHostEnvironment env, KeyToBinaryVectorMappingTransformer transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(KeyToBinaryVectorMappingEstimator)), transformer)
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
                if (!(col.ItemType is VectorType || col.ItemType is PrimitiveDataViewType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName);

                var metadata = new List<SchemaShape.Column>();
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.KeyValues, out var keyMeta))
                    if (col.Kind != SchemaShape.Column.VectorKind.VariableVector && keyMeta.ItemType is TextDataViewType)
                        metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, keyMeta.ItemType, false));
                if (col.Kind == SchemaShape.Column.VectorKind.Scalar)
                    metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));
                result[colInfo.outputColumnName] = new SchemaShape.Column(colInfo.outputColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(metadata));
            }

            return new SchemaShape(result.Values);
        }
    }

}
