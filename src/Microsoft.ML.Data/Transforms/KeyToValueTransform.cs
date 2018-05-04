// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(KeyToValueTransform), typeof(KeyToValueTransform.Arguments), typeof(SignatureDataTransform),
    KeyToValueTransform.UserName, KeyToValueTransform.LoaderSignature, "KeyToValue", "KeyToVal", "Unterm")]

[assembly: LoadableClass(typeof(KeyToValueTransform), null, typeof(SignatureLoadDataTransform),
    KeyToValueTransform.UserName, KeyToValueTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// KeyToValueTransform utilizes KeyValues metadata to map key indices to the corresponding values in the KeyValues metadata.
    /// Notes:
    /// * Output columns utilize the KeyValues metadata.
    /// * Maps zero values of the key type to the NA of the output type.
    /// </summary>
    public sealed class KeyToValueTransform : OneToOneTransformBase
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

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "KEY2VALT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public override bool CanSavePfa => true;

        // These are parallel to Infos.
        private readonly ColumnType[] _types;
        private KeyToValueMap[] _kvMaps;

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public KeyToValueTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, LoaderSignature, Contracts.CheckRef(args, nameof(args)).Column,
                input, TestIsKey)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            ComputeKVMapsAndMetadata(out _types, out _kvMaps);
        }

        private KeyToValueTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsKey)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***

            ComputeKVMapsAndMetadata(out _types, out _kvMaps);

        }

        public static KeyToValueTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(input, nameof(input));
            env.CheckValue(env, nameof(env));
            var h = env.Register(LoaderSignature);
            return h.Apply("Loading Model", ch => new KeyToValueTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            SaveBase(ctx);
        }

        protected override JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, ColInfo info, JToken srcToken)
        {
            Contracts.AssertValue(ctx);
            Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);
            Contracts.Assert(Infos[iinfo] == info);
            Contracts.AssertValue(srcToken);
            Contracts.Assert(CanSavePfa);
            return _kvMaps[iinfo].SavePfa(ctx, srcToken);
        }

        // Computes the types of the columns and constructs the kvMaps.
        private void ComputeKVMapsAndMetadata(out ColumnType[] types, out KeyToValueMap[] kvMaps)
        {
            Contracts.Assert(Utils.Size(Infos) > 0);

            var md = Metadata;
            types = new ColumnType[Infos.Length];
            kvMaps = new KeyToValueMap[Infos.Length];
            for (int iinfo = 0; iinfo < types.Length; iinfo++)
            {
                // Inherit slot names from source.
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, Infos[iinfo].Source, MetadataUtils.Kinds.SlotNames))
                {
                }

                // Construct kvMaps.
                Contracts.Assert(types[iinfo] == null);
                var typeSrc = Infos[iinfo].TypeSrc;
                var typeVals = Source.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, Infos[iinfo].Source);
                Host.Check(typeVals != null, "Metadata KeyValues does not exist");
                Host.Check(typeVals.VectorSize == typeSrc.ItemType.KeyCount, "KeyValues metadata size does not match column type key count");
                if (!typeSrc.IsVector)
                    types[iinfo] = typeVals.ItemType;
                else
                    types[iinfo] = new VectorType(typeVals.ItemType.AsPrimitive, typeSrc.AsVector);
                Func<int, ColumnType, ColumnType, KeyToValueMap> func = GetKeyMetadata<int, int>;
                var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(new Type[] { typeSrc.ItemType.RawType,
                    types[iinfo].ItemType.RawType });
                kvMaps[iinfo] = (KeyToValueMap)meth.Invoke(this, new object[] { iinfo, typeSrc, typeVals });
            }
            md.Seal();
        }

        private KeyToValueMap GetKeyMetadata<TKey, TValue>(int iinfo, ColumnType typeKey, ColumnType typeVal)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.AssertValue(typeKey);
            Host.AssertValue(typeVal);
            Host.Assert(typeKey.ItemType.RawType == typeof(TKey));
            Host.Assert(typeVal.ItemType.RawType == typeof(TValue));

            var keyMetadata = default(VBuffer<TValue>);
            Source.Schema.GetMetadata<VBuffer<TValue>>(MetadataUtils.Kinds.KeyValues, Infos[iinfo].Source, ref keyMetadata);
            Host.Check(keyMetadata.Length == typeKey.ItemType.KeyCount);

            VBufferUtils.Densify<TValue>(ref keyMetadata);
            return new KeyToValueMap<TKey, TValue>(this, typeKey.ItemType.AsKey, typeVal.ItemType.AsPrimitive, keyMetadata.Values, iinfo);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < _types.Length);
            var type = _types[iinfo];
            Host.Assert(type != null);
            return type;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            return _kvMaps[iinfo].GetMappingGetter(input);
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
            protected readonly KeyToValueTransform Parent;

            protected KeyToValueMap(KeyToValueTransform trans, PrimitiveType typeVal, int iinfo)
            {
                // REVIEW: Is there a better way to perform this first assert value?
                Contracts.AssertValue(trans);
                Parent = trans;
                Parent.Host.AssertValue(typeVal);
                Parent.Host.Assert(0 <= iinfo && iinfo < Parent.Infos.Length);
                TypeOutput = typeVal;
                InfoIndex = iinfo;
            }

            public abstract Delegate GetMappingGetter(IRow input);

            public abstract JToken SavePfa(BoundPfaContext ctx, JToken srcToken);
        }

        private class KeyToValueMap<TKey, TValue> : KeyToValueMap
        {
            private readonly TValue[] _values;
            private readonly TValue _na;

            private readonly bool _naMapsToDefault;
            private readonly RefPredicate<TValue> _isDefault;

            private readonly ValueMapper<TKey, UInt32> _convertToUInt;

            public KeyToValueMap(KeyToValueTransform trans, KeyType typeKey, PrimitiveType typeVal, TValue[] values, int iinfo)
                : base(trans, typeVal, iinfo)
            {
                Parent.Host.AssertValue(values);
                Parent.Host.Assert(typeKey.RawType == typeof(TKey));
                Parent.Host.Assert(TypeOutput.RawType == typeof(TValue));
                _values = values;

                // REVIEW: May want to include more specific information about what the specific value is for the default.
                using (var ch = Parent.Host.Start("Getting NA Predicate and Value"))
                {
                    _na = Conversions.Instance.GetNAOrDefault<TValue>(TypeOutput.ItemType, out _naMapsToDefault);

                    if (_naMapsToDefault)
                    {
                        // Only initialize _isDefault if _defaultIsNA is true as this is the only case in which it is used.
                        _isDefault = Conversions.Instance.GetIsDefaultPredicate<TValue>(TypeOutput.ItemType);
                        RefPredicate<TValue> del;
                        if (!Conversions.Instance.TryGetIsNAPredicate<TValue>(TypeOutput.ItemType, out del))
                        {
                            ch.Warning("There is no NA value for type '{0}'. The missing key value " +
                                "will be mapped to the default value of '{0}'", TypeOutput.ItemType);
                        }
                    }
                }

                bool identity;
                _convertToUInt = Conversions.Instance.GetStandardConversion<TKey, UInt32>(typeKey, NumberType.U4, out identity);
            }

            private void MapKey(ref TKey src, ref TValue dst)
            {
                uint uintSrc = 0;
                _convertToUInt(ref src, ref uintSrc);
                // Assign to NA if key value is not in valid range.
                if (0 < uintSrc && uintSrc <= _values.Length)
                    dst = _values[uintSrc - 1];
                else
                    dst = _na;
            }

            public override Delegate GetMappingGetter(IRow input)
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

                if (!Parent._types[InfoIndex].IsVector)
                {
                    var src = default(TKey);
                    ValueGetter<TKey> getSrc = Parent.GetSrcGetter<TKey>(input, InfoIndex);
                    ValueGetter<TValue> retVal =
                        (ref TValue dst) =>
                        {
                            getSrc(ref src);
                            MapKey(ref src, ref dst);
                        };
                    return retVal;
                }
                else
                {
                    var src = default(VBuffer<TKey>);
                    var dstItem = default(TValue);
                    int maxSize = TypeOutput.IsKnownSizeVector ? TypeOutput.VectorSize : Utils.ArrayMaxSize;
                    ValueGetter<VBuffer<TKey>> getSrc = Parent.GetSrcGetter<VBuffer<TKey>>(input, InfoIndex);
                    ValueGetter<VBuffer<TValue>> retVal =
                        (ref VBuffer<TValue> dst) =>
                        {
                            getSrc(ref src);
                            int srcSize = src.Length;
                            int srcCount = src.Count;
                            var srcValues = src.Values;
                            var dstValues = dst.Values;
                            var dstIndices = dst.Indices;

                            int islotDst = 0;

                            if (src.IsDense)
                            {
                                Utils.EnsureSize(ref dstValues, srcSize, maxSize, keepOld: false);

                                for (int slot = 0; slot < srcSize; ++slot)
                                {
                                    MapKey(ref srcValues[slot], ref dstValues[slot]);

                                    // REVIEW:
                                    // The current implementation always maps dense to dense, even if the resulting columns could benefit from
                                    // sparsity. This would only occur if there are key values that map over half of the keys to the default value.
                                    // One way to rule out the helpfulness of sparsifying is to have a flag that indicates whether any key maps to
                                    // default, still need a good method for discerning when to implement sparsity (would either need precomputation
                                    // of the amount of default values or allow for some dynamic updating to sparsity when the requisite number of
                                    // defaults is hit. We assume that if the user was willing to densify the data into key values that they will
                                    // be fine with this output being dense.
                                }
                                islotDst = srcSize;
                            }
                            else if (!_naMapsToDefault)
                            {
                                // Sparse input will always result in dense output unless the key metadata maps back to key types.
                                // Currently this always maps sparse to dense, as long as the output type's NA does not equal its default value.
                                Utils.EnsureSize(ref dstValues, srcSize, maxSize, keepOld: false);

                                var srcIndices = src.Indices;
                                int nextExplicitSlot = src.Count == 0 ? srcSize : srcIndices[0];
                                int islot = 0;
                                for (int slot = 0; slot < srcSize; ++slot)
                                {
                                    if (nextExplicitSlot == slot)
                                    {
                                        // Current slot has an explicitly defined value.
                                        Parent.Host.Assert(islot < src.Count);
                                        MapKey(ref srcValues[islot], ref dstValues[slot]);
                                        nextExplicitSlot = ++islot == src.Count ? srcSize : srcIndices[islot];
                                        Parent.Host.Assert(slot < nextExplicitSlot);
                                    }
                                    else
                                    {
                                        Parent.Host.Assert(slot < nextExplicitSlot);
                                        dstValues[slot] = _na;
                                    }
                                }
                                islotDst = srcSize;
                            }
                            else
                            {
                                // As the default value equals the NA value for the output type, we produce sparse output.
                                Utils.EnsureSize(ref dstValues, srcCount, maxSize, keepOld: false);
                                Utils.EnsureSize(ref dstIndices, srcCount, maxSize, keepOld: false);
                                var srcIndices = src.Indices;
                                for (int islotSrc = 0; islotSrc < srcCount; ++islotSrc)
                                {
                                    // Current slot has an explicitly defined value.
                                    Parent.Host.Assert(islotSrc < srcCount);
                                    MapKey(ref srcValues[islotSrc], ref dstItem);
                                    if (!_isDefault(ref dstItem))
                                    {
                                        dstValues[islotDst] = dstItem;
                                        dstIndices[islotDst++] = srcIndices[islotSrc];
                                    }
                                }
                            }
                            dst = new VBuffer<TValue>(srcSize, islotDst, dstValues, dstIndices);
                        };
                    return retVal;
                }
            }

            public override JToken SavePfa(BoundPfaContext ctx, JToken srcToken)
            {
                Contracts.AssertValue(ctx);
                Contracts.AssertValue(srcToken);
                var info = Parent.Infos[InfoIndex];
                var outType = PfaUtils.Type.PfaTypeOrNullForColumnType(TypeOutput);
                if (outType == null)
                    return null;

                // REVIEW: To map the missing key to the *default* value is
                // wrong, but the alternative is we have a bunch of null unions everywhere
                // probably, which I am not prepared to do.
                var defaultToken = PfaUtils.Type.DefaultTokenOrNull(TypeOutput);
                JArray jsonValues;
                if (TypeOutput.IsText)
                {
                    jsonValues = new JArray();
                    for (int i = 0; i < _values.Length; ++i)
                        jsonValues.Add(_values[i].ToString());
                }
                else
                    jsonValues = new JArray(_values);

                string cellName = ctx.DeclareCell("KeyToValueMap", PfaUtils.Type.Array(outType), jsonValues);
                JObject cellRef = PfaUtils.Cell(cellName);

                if (info.TypeSrc.IsVector)
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
