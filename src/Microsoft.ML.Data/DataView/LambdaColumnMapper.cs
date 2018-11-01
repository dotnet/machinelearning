// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This applies the user provided ValueMapper to a column to produce a new column. It automatically
    /// injects a standard conversion from the actual type of the source column to typeSrc (if needed).
    /// </summary>
    public static class LambdaColumnMapper
    {
        // REVIEW: It would be nice to support propagation of select metadata.
        public static IDataView Create<TSrc, TDst>(IHostEnvironment env, string name, IDataView input,
            string src, string dst, ColumnType typeSrc, ColumnType typeDst, ValueMapper<TSrc, TDst> mapper,
            ValueGetter<VBuffer<ReadOnlyMemory<char>>> keyValueGetter = null, ValueGetter<VBuffer<ReadOnlyMemory<char>>> slotNamesGetter = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonEmpty(name, nameof(name));
            env.CheckValue(input, nameof(input));
            env.CheckNonEmpty(src, nameof(src));
            env.CheckNonEmpty(dst, nameof(dst));
            env.CheckValue(typeSrc, nameof(typeSrc));
            env.CheckValue(typeDst, nameof(typeDst));
            env.CheckValue(mapper, nameof(mapper));
            env.Check(keyValueGetter == null || typeDst.ItemType.IsKey);
            env.Check(slotNamesGetter == null || typeDst.IsKnownSizeVector);

            if (typeSrc.RawType != typeof(TSrc))
            {
                throw env.ExceptParam(nameof(mapper),
                    "The source column type '{0}' doesn't match the input type of the mapper", typeSrc);
            }
            if (typeDst.RawType != typeof(TDst))
            {
                throw env.ExceptParam(nameof(mapper),
                    "The destination column type '{0}' doesn't match the output type of the mapper", typeDst);
            }

            bool tmp = input.Schema.TryGetColumnIndex(src, out int colSrc);
            if (!tmp)
                throw env.ExceptParam(nameof(src), "The input data doesn't have a column named '{0}'", src);
            var typeOrig = input.Schema.GetColumnType(colSrc);

            // REVIEW: Ideally this should support vector-type conversion. It currently doesn't.
            bool ident;
            Delegate conv;
            if (typeOrig.SameSizeAndItemType(typeSrc))
            {
                ident = true;
                conv = null;
            }
            else if (!Conversions.Instance.TryGetStandardConversion(typeOrig, typeSrc, out conv, out ident))
            {
                throw env.ExceptParam(nameof(mapper),
                    "The type of column '{0}', '{1}', cannot be converted to the input type of the mapper '{2}'",
                    src, typeOrig, typeSrc);
            }

            var col = new Column(src, dst);
            IDataView impl;
            if (ident)
                impl = new Impl<TSrc, TDst, TDst>(env, name, input, col, typeDst, mapper, keyValueGetter: keyValueGetter, slotNamesGetter: slotNamesGetter);
            else
            {
                Func<IHostEnvironment, string, IDataView, Column, ColumnType, ValueMapper<int, int>,
                    ValueMapper<int, int>, ValueGetter<VBuffer<ReadOnlyMemory<char>>>, ValueGetter<VBuffer<ReadOnlyMemory<char>>>,
                    Impl<int, int, int>> del = CreateImpl<int, int, int>;
                var meth = del.GetMethodInfo().GetGenericMethodDefinition()
                    .MakeGenericMethod(typeOrig.RawType, typeof(TSrc), typeof(TDst));
                impl = (IDataView)meth.Invoke(null, new object[] { env, name, input, col, typeDst, conv, mapper, keyValueGetter, slotNamesGetter });
            }

            return new OpaqueDataView(impl);
        }

        private static Impl<T1, T2, T3> CreateImpl<T1, T2, T3>(
            IHostEnvironment env, string name, IDataView input, Column col,
            ColumnType typeDst, ValueMapper<T1, T2> map1, ValueMapper<T2, T3> map2,
            ValueGetter<VBuffer<ReadOnlyMemory<char>>> keyValueGetter, ValueGetter<VBuffer<ReadOnlyMemory<char>>> slotNamesGetter)
        {
            return new Impl<T1, T2, T3>(env, name, input, col, typeDst, map1, map2, keyValueGetter);
        }

        private sealed class Column : OneToOneColumn
        {
            public Column(string src, string dst)
            {
                Name = dst;
                Source = src;
            }
        }

        private sealed class Impl<T1, T2, T3> : OneToOneTransformBase
        {
            private readonly ColumnType _typeDst;
            private readonly ValueMapper<T1, T2> _map1;
            private readonly ValueMapper<T2, T3> _map2;

            public Impl(IHostEnvironment env, string name, IDataView input, OneToOneColumn col,
                ColumnType typeDst, ValueMapper<T1, T2> map1, ValueMapper<T2, T3> map2 = null,
                ValueGetter<VBuffer<ReadOnlyMemory<char>>> keyValueGetter = null, ValueGetter<VBuffer<ReadOnlyMemory<char>>> slotNamesGetter = null)
                : base(env, name, new[] { col }, input, x => null)
            {
                Host.Assert(typeDst.RawType == typeof(T3));
                Host.AssertValue(map1);
                Host.Assert(map2 != null || typeof(T2) == typeof(T3));

                _typeDst = typeDst;
                _map1 = map1;
                _map2 = map2;

                if (keyValueGetter != null || slotNamesGetter != null)
                {
                    using (var bldr = Metadata.BuildMetadata(0))
                    {
                        if (keyValueGetter != null)
                        {
                            Host.Assert(_typeDst.ItemType.KeyCount > 0);
                            MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>> mdGetter =
                                (int c, ref VBuffer<ReadOnlyMemory<char>> dst) => keyValueGetter(ref dst);
                            bldr.AddGetter(MetadataUtils.Kinds.KeyValues, new VectorType(TextType.Instance, _typeDst.ItemType.KeyCount), mdGetter);
                        }
                        if (slotNamesGetter != null)
                        {
                            Host.Assert(_typeDst.VectorSize > 0);
                            MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>> mdGetter =
                                (int c, ref VBuffer<ReadOnlyMemory<char>> dst) => slotNamesGetter(ref dst);
                            bldr.AddGetter(MetadataUtils.Kinds.SlotNames, new VectorType(TextType.Instance, _typeDst.VectorSize), mdGetter);
                        }
                    }
                }
                Metadata.Seal();
            }

            public override void Save(ModelSaveContext ctx)
            {
                Host.Assert(false, "Shouldn't serialize this!");
                throw Host.ExceptNotSupp("Shouldn't serialize this");
            }

            protected override ColumnType GetColumnTypeCore(int iinfo)
            {
                Host.Assert(iinfo == 0);
                return _typeDst;
            }

            protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValueOrNull(ch);
                Host.AssertValue(input);
                Host.Assert(iinfo == 0);
                disposer = null;

                if (_map2 == null)
                {
                    var getSrc = GetSrcGetter<T1>(input, 0);
                    T1 v1 = default(T1);
                    ValueGetter<T2> getter =
                        (ref T2 v2) =>
                        {
                            getSrc(ref v1);
                            _map1(in v1, ref v2);
                        };
                    return getter;
                }
                else
                {
                    var getSrc = GetSrcGetter<T1>(input, 0);
                    T1 v1 = default(T1);
                    T2 v2 = default(T2);
                    ValueGetter<T3> getter =
                        (ref T3 v3) =>
                        {
                            getSrc(ref v1);
                            _map1(in v1, ref v2);
                            _map2(in v2, ref v3);
                        };
                    return getter;
                }
            }
        }
    }
}
