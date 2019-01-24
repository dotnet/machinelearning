// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Transforms.Conversions;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(IDataTransform), typeof(KeyToValueMappingTransformer), typeof(KeyToValueMappingTransformer.Arguments), typeof(SignatureDataTransform),
    KeyToValueMappingTransformer.UserName, KeyToValueMappingTransformer.LoaderSignature, "KeyToValue", "KeyToVal", "Unterm")]

[assembly: LoadableClass(typeof(IDataTransform), typeof(KeyToValueMappingTransformer), null, typeof(SignatureLoadDataTransform),
    KeyToValueMappingTransformer.UserName, KeyToValueMappingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(KeyToValueMappingTransformer), null, typeof(SignatureLoadModel),
    KeyToValueMappingTransformer.UserName, KeyToValueMappingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(KeyToValueMappingTransformer), null, typeof(SignatureLoadRowMapper),
    KeyToValueMappingTransformer.UserName, KeyToValueMappingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Conversions
{
    /// <summary>
    /// KeyToValueTransform utilizes KeyValues metadata to map key indices to the corresponding values in the KeyValues metadata.
    /// Notes:
    /// * Output columns utilize the KeyValues metadata.
    /// * Maps zero values of the key type to the NA of the output type.
    /// </summary>
    public sealed class KeyToValueMappingTransformer : OneToOneTransformerBase
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
        }

        public const string LoaderSignature = "KeyToValueTransform";
        public const string UserName = "Key To Value Transform";

        public IReadOnlyCollection<(string input, string output)> Columns => ColumnPairs.AsReadOnly();

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "KEY2VALT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(KeyToValueMappingTransformer).Assembly.FullName);
        }

        /// <summary>
        /// Create a <see cref="KeyToValueMappingTransformer"/> that takes and transforms one column.
        /// </summary>
        public KeyToValueMappingTransformer(IHostEnvironment env, string columnName)
            : this(env, (columnName, columnName))
        {
        }

        /// <summary>
        /// Create a <see cref="KeyToValueMappingTransformer"/> that takes multiple pairs of columns.
        /// </summary>
        public KeyToValueMappingTransformer(IHostEnvironment env, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(KeyToValueMappingTransformer)), columns)
        {
        }

        /// <summary>
        /// Factory method for SignatureDataTransform.
        /// </summary>
        [BestFriend]
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));
            env.CheckNonEmpty(args.Column, nameof(args.Column));

            var transformer = new KeyToValueMappingTransformer(env, args.Column.Select(c => (c.Source ?? c.Name, c.Name)).ToArray());
            return transformer.MakeDataTransform(input);
        }

        /// <summary>
        /// Factory method for SignatureLoadModel.
        /// </summary>
        private static KeyToValueMappingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(KeyToValueMappingTransformer));
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new KeyToValueMappingTransformer(host, ctx);
        }

        private KeyToValueMappingTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
        }

        /// <summary>
        /// Factory method for SignatureLoadDataTransform.
        /// </summary>
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        /// <summary>
        /// Factory method for SignatureLoadRowMapper.
        /// </summary>
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>

            SaveColumns(ctx);
        }

        private protected override IRowMapper MakeRowMapper(Schema inputSchema) => new Mapper(this, inputSchema);

        private sealed class Mapper : OneToOneMapperBase, ISaveAsPfa
        {
            private readonly KeyToValueMappingTransformer _parent;
            private readonly ColumnType[] _types;
            private readonly KeyToValueMap[] _kvMaps;

            public Mapper(KeyToValueMappingTransformer parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                ComputeKvMaps(inputSchema, out _types, out _kvMaps);
            }

            public bool CanSavePfa => true;

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new Schema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var meta = new MetadataBuilder();
                    meta.Add(InputSchema[ColMapNewToOld[i]].Metadata, name => name == MetadataUtils.Kinds.SlotNames);
                    result[i] = new Schema.DetachedColumn(_parent.ColumnPairs[i].output, _types[i], meta.GetMetadata());
                }
                return result;
            }

            public void SaveAsPfa(BoundPfaContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                var toHide = new List<string>();
                var toDeclare = new List<KeyValuePair<string, JToken>>();

                for (int iinfo = 0; iinfo < _parent.ColumnPairs.Length; ++iinfo)
                {
                    var info = _parent.ColumnPairs[iinfo];
                    var srcName = info.input;
                    string srcToken = ctx.TokenOrNullForName(srcName);
                    if (srcToken == null)
                    {
                        toHide.Add(info.output);
                        continue;
                    }
                    var result = _kvMaps[iinfo].SavePfa(ctx, srcToken);
                    if (result == null)
                    {
                        toHide.Add(info.output);
                        continue;
                    }
                    toDeclare.Add(new KeyValuePair<string, JToken>(info.output, result));
                }
                ctx.Hide(toHide.ToArray());
                ctx.DeclareVar(toDeclare.ToArray());
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _types.Length);
                disposer = null;
                return _kvMaps[iinfo].GetMappingGetter(input);
            }

            // Computes the types of the columns and constructs the kvMaps.
            private void ComputeKvMaps(Schema schema, out ColumnType[] types, out KeyToValueMap[] kvMaps)
            {
                types = new ColumnType[_parent.ColumnPairs.Length];
                kvMaps = new KeyToValueMap[_parent.ColumnPairs.Length];
                for (int iinfo = 0; iinfo < types.Length; iinfo++)
                {
                    // Construct kvMaps.
                    Contracts.Assert(types[iinfo] == null);
                    var typeSrc = schema[ColMapNewToOld[iinfo]].Type;
                    var typeVals = schema[ColMapNewToOld[iinfo]].Metadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.KeyValues)?.Type;
                    Host.Check(typeVals != null, "Metadata KeyValues does not exist");
                    ColumnType valsItemType = typeVals.GetItemType();
                    ColumnType srcItemType = typeSrc.GetItemType();
                    Host.Check(typeVals.GetVectorSize() == srcItemType.GetKeyCountAsInt32(Host), "KeyValues metadata size does not match column type key count");
                    if (!(typeSrc is VectorType vectorType))
                        types[iinfo] = valsItemType;
                    else
                        types[iinfo] = new VectorType((PrimitiveType)valsItemType, vectorType);

                    // MarshalInvoke with two generic params.
                    Func<int, ColumnType, ColumnType, KeyToValueMap> func = GetKeyMetadata<int, int>;
                    var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(
                        new Type[] { srcItemType.RawType, types[iinfo].GetItemType().RawType });
                    kvMaps[iinfo] = (KeyToValueMap)meth.Invoke(this, new object[] { iinfo, typeSrc, typeVals });
                }
            }

            private KeyToValueMap GetKeyMetadata<TKey, TValue>(int iinfo, ColumnType typeKey, ColumnType typeVal)
            {
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                Host.AssertValue(typeKey);
                Host.AssertValue(typeVal);
                ColumnType keyItemType = typeKey.GetItemType();
                ColumnType valItemType = typeVal.GetItemType();
                Host.Assert(keyItemType.RawType == typeof(TKey));
                Host.Assert(valItemType.RawType == typeof(TValue));

                var keyMetadata = default(VBuffer<TValue>);
                InputSchema[ColMapNewToOld[iinfo]].Metadata.GetValue(MetadataUtils.Kinds.KeyValues, ref keyMetadata);
                Host.Check(keyMetadata.Length == keyItemType.GetKeyCountAsInt32(Host));

                VBufferUtils.Densify(ref keyMetadata);
                return new KeyToValueMap<TKey, TValue>(this, (KeyType)keyItemType, (PrimitiveType)valItemType, keyMetadata, iinfo);
            }
            /// <summary>
            /// A map is an object capable of creating the association from an input type, to an output
            /// type. This mapping is constructed from key metadata, with the input type being the key type
            /// and the output type being the type specified by the key metadata.
            /// </summary>
            private abstract class KeyToValueMap
            {
                /// <summary>
                /// The item type of the output type, that is, either the output type or,
                /// if a vector, the item type of that type.
                /// </summary>
                protected readonly PrimitiveType TypeOutput;

                /// <summary>
                /// The column index in Infos.
                /// </summary>
                protected readonly int InfoIndex;

                /// <summary>
                /// The parent transform.
                /// </summary>
                protected readonly Mapper Parent;

                protected KeyToValueMap(Mapper mapper, PrimitiveType typeVal, int iinfo)
                {
                    // REVIEW: Is there a better way to perform this first assert value?
                    Contracts.AssertValue(mapper);
                    Parent = mapper;
                    Parent.Host.AssertValue(typeVal);
                    Parent.Host.Assert(0 <= iinfo && iinfo < Parent._types.Length);
                    TypeOutput = typeVal;
                    InfoIndex = iinfo;
                }

                public abstract Delegate GetMappingGetter(Row input);

                public abstract JToken SavePfa(BoundPfaContext ctx, JToken srcToken);
            }

            private class KeyToValueMap<TKey, TValue> : KeyToValueMap
            {
                private readonly VBuffer<TValue> _values;
                private readonly TValue _na;

                private readonly bool _naMapsToDefault;
                private readonly InPredicate<TValue> _isDefault;

                private readonly ValueMapper<TKey, UInt32> _convertToUInt;

                public KeyToValueMap(Mapper parent, KeyType typeKey, PrimitiveType typeVal, VBuffer<TValue> values, int iinfo)
                    : base(parent, typeVal, iinfo)
                {
                    Parent.Host.Assert(values.IsDense);
                    Parent.Host.Assert(typeKey.RawType == typeof(TKey));
                    Parent.Host.Assert(TypeOutput.RawType == typeof(TValue));
                    _values = values;

                    // REVIEW: May want to include more specific information about what the specific value is for the default.
                    ColumnType outputItemType = TypeOutput.GetItemType();
                    _na = Data.Conversion.Conversions.Instance.GetNAOrDefault<TValue>(outputItemType, out _naMapsToDefault);

                    if (_naMapsToDefault)
                    {
                        // Only initialize _isDefault if _defaultIsNA is true as this is the only case in which it is used.
                        _isDefault = Data.Conversion.Conversions.Instance.GetIsDefaultPredicate<TValue>(outputItemType);
                    }

                    bool identity;
                    _convertToUInt = Data.Conversion.Conversions.Instance.GetStandardConversion<TKey, UInt32>(typeKey, NumberType.U4, out identity);
                }

                private void MapKey(in TKey src, ref TValue dst)
                {
                    MapKey(in src, _values.GetValues(), ref dst);
                }

                private void MapKey(in TKey src, ReadOnlySpan<TValue> values, ref TValue dst)
                {
                    uint uintSrc = 0;
                    _convertToUInt(in src, ref uintSrc);
                    // Assign to NA if key value is not in valid range.
                    if (0 < uintSrc && uintSrc <= values.Length)
                        dst = values[(int)(uintSrc - 1)];
                    else
                        dst = _na;
                }

                public override Delegate GetMappingGetter(Row input)
                {
                    // When constructing the getter, there are a few cases we have to consider:
                    // If scalar then it's just a straightforward mapping.
                    // If vector, then we have to detect whether the mapping should be mapped to
                    // dense or sparse. Almost all cases will map to dense (as the NA key value
                    // represented by sparsity will map to the NA of the corresponding type) but
                    // if enough key values map to the default value of the output type sparsifying
                    // may be desirable, as is the case when the default value is equal to the
                    // NA value.

                    Parent.Host.AssertValue(input);

                    if (!(Parent._types[InfoIndex] is VectorType))
                    {
                        var src = default(TKey);
                        ValueGetter<TKey> getSrc = input.GetGetter<TKey>(Parent.ColMapNewToOld[InfoIndex]);
                        ValueGetter<TValue> retVal =
                            (ref TValue dst) =>
                            {
                                getSrc(ref src);
                                MapKey(in src, ref dst);
                            };
                        return retVal;
                    }
                    else
                    {
                        var src = default(VBuffer<TKey>);
                        var dstItem = default(TValue);
                        ValueGetter<VBuffer<TKey>> getSrc = input.GetGetter<VBuffer<TKey>>(Parent.ColMapNewToOld[InfoIndex]);
                        ValueGetter<VBuffer<TValue>> retVal =
                            (ref VBuffer<TValue> dst) =>
                            {
                                getSrc(ref src);
                                int srcSize = src.Length;
                                var srcValues = src.GetValues();
                                int srcCount = srcValues.Length;

                                var keyValues = _values.GetValues();
                                if (src.IsDense)
                                {
                                    var editor = VBufferEditor.Create(ref dst, srcSize);
                                    for (int slot = 0; slot < srcSize; ++slot)
                                    {
                                        MapKey(in srcValues[slot], keyValues, ref editor.Values[slot]);

                                        // REVIEW:
                                        // The current implementation always maps dense to dense, even if the resulting columns could benefit from
                                        // sparsity. This would only occur if there are key values that map over half of the keys to the default value.
                                        // One way to rule out the helpfulness of sparsifying is to have a flag that indicates whether any key maps to
                                        // default, still need a good method for discerning when to implement sparsity (would either need precomputation
                                        // of the amount of default values or allow for some dynamic updating to sparsity when the requisite number of
                                        // defaults is hit. We assume that if the user was willing to densify the data into key values that they will
                                        // be fine with this output being dense.
                                    }
                                    dst = editor.Commit();
                                }
                                else if (!_naMapsToDefault)
                                {
                                    // Sparse input will always result in dense output unless the key metadata maps back to key types.
                                    // Currently this always maps sparse to dense, as long as the output type's NA does not equal its default value.
                                    var editor = VBufferEditor.Create(ref dst, srcSize);

                                    var srcIndices = src.GetIndices();
                                    int nextExplicitSlot = srcCount == 0 ? srcSize : srcIndices[0];
                                    int islot = 0;
                                    for (int slot = 0; slot < srcSize; ++slot)
                                    {
                                        if (nextExplicitSlot == slot)
                                        {
                                            // Current slot has an explicitly defined value.
                                            Parent.Host.Assert(islot < srcCount);
                                            MapKey(in srcValues[islot], keyValues, ref editor.Values[slot]);
                                            nextExplicitSlot = ++islot == srcCount ? srcSize : srcIndices[islot];
                                            Parent.Host.Assert(slot < nextExplicitSlot);
                                        }
                                        else
                                        {
                                            Parent.Host.Assert(slot < nextExplicitSlot);
                                            editor.Values[slot] = _na;
                                        }
                                    }
                                    dst = editor.Commit();
                                }
                                else
                                {
                                    // As the default value equals the NA value for the output type, we produce sparse output.
                                    var editor = VBufferEditor.Create(ref dst, srcSize, srcCount);
                                    var srcIndices = src.GetIndices();
                                    var islotDst = 0;
                                    for (int islotSrc = 0; islotSrc < srcCount; ++islotSrc)
                                    {
                                        // Current slot has an explicitly defined value.
                                        Parent.Host.Assert(islotSrc < srcCount);
                                        MapKey(in srcValues[islotSrc], keyValues, ref dstItem);
                                        if (!_isDefault(in dstItem))
                                        {
                                            editor.Values[islotDst] = dstItem;
                                            editor.Indices[islotDst++] = srcIndices[islotSrc];
                                        }
                                    }
                                    dst = editor.CommitTruncated(islotDst);
                                }
                            };
                        return retVal;
                    }
                }

                public override JToken SavePfa(BoundPfaContext ctx, JToken srcToken)
                {
                    Contracts.AssertValue(ctx);
                    Contracts.AssertValue(srcToken);
                    var outType = PfaUtils.Type.PfaTypeOrNullForColumnType(TypeOutput);
                    if (outType == null)
                        return null;

                    // REVIEW: To map the missing key to the *default* value is
                    // wrong, but the alternative is we have a bunch of null unions everywhere
                    // probably, which I am not prepared to do.
                    var defaultToken = PfaUtils.Type.DefaultTokenOrNull(TypeOutput);
                    JArray jsonValues;
                    if (TypeOutput is TextType)
                    {
                        jsonValues = new JArray();
                        var keyValues = _values.GetValues();
                        for (int i = 0; i < keyValues.Length; ++i)
                            jsonValues.Add(keyValues[i].ToString());
                    }
                    else
                        jsonValues = new JArray(_values);

                    string cellName = ctx.DeclareCell("KeyToValueMap", PfaUtils.Type.Array(outType), jsonValues);
                    JObject cellRef = PfaUtils.Cell(cellName);

                    var srcType = Parent.InputSchema[Parent.ColMapNewToOld[InfoIndex]].Type;
                    if (srcType is VectorType)
                    {
                        var funcName = ctx.GetFreeFunctionName("mapKeyToValue");
                        ctx.Pfa.AddFunc(funcName, new JArray(PfaUtils.Param("key", PfaUtils.Type.Int)),
                            outType, PfaUtils.If(PfaUtils.Call("<", "key", 0), defaultToken,
                            PfaUtils.Index(cellRef, "key")));
                        var funcRef = PfaUtils.FuncRef("u." + funcName);
                        return PfaUtils.Call("a.map", srcToken, funcRef);
                    }
                    return PfaUtils.If(PfaUtils.Call("<", srcToken, 0), defaultToken, PfaUtils.Index(cellRef, srcToken));
                }
            }

        }
    }

    public sealed class KeyToValueMappingEstimator : TrivialEstimator<KeyToValueMappingTransformer>
    {
        public KeyToValueMappingEstimator(IHostEnvironment env, string columnName)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(KeyToValueMappingEstimator)), new KeyToValueMappingTransformer(env, columnName))
        {
        }

        public KeyToValueMappingEstimator(IHostEnvironment env, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(KeyToValueMappingEstimator)), new KeyToValueMappingTransformer(env, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.input);
                if (!col.IsKey)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.input, "key type", col.GetTypeString());

                if (!col.Metadata.TryFindColumn(MetadataUtils.Kinds.KeyValues, out var keyMetaCol))
                    throw Host.ExceptParam(nameof(inputSchema), $"Input column '{colInfo.input}' doesn't contain key values metadata");

                SchemaShape metadata = null;
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotCol))
                    metadata = new SchemaShape(new[] { slotCol });

                result[colInfo.output] = new SchemaShape.Column(colInfo.output, col.Kind, keyMetaCol.ItemType, keyMetaCol.IsKey, metadata);
            }

            return new SchemaShape(result.Values);
        }
    }
}
