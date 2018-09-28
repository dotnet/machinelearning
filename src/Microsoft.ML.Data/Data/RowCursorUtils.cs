// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    public static class RowCursorUtils
    {
        /// <summary>
        /// Returns an appropriate <see cref="ValueGetter{T}"/> for a row given an active column
        /// index, but as a delegate. The type parameter for the delegate will correspond to the
        /// raw type of the column.
        /// </summary>
        /// <param name="row">The row to get the getter for</param>
        /// <param name="col">The column index, which must be active on that row</param>
        /// <returns>The getter as a delegate</returns>
        public static Delegate GetGetterAsDelegate(IRow row, int col)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.ColumnCount, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            Func<IRow, int, Delegate> getGetter = GetGetterAsDelegateCore<int>;
            return Utils.MarshalInvoke(getGetter, row.Schema.GetColumnType(col).RawType, row, col);
        }

        private static Delegate GetGetterAsDelegateCore<TValue>(IRow row, int col)
        {
            return row.GetGetter<TValue>(col);
        }

        /// <summary>
        /// Given a destination type, IRow, and column index, return a ValueGetter for the column
        /// with a conversion to typeDst, if needed. This is a weakly typed version of
        /// <see cref="GetGetterAs{TDst}"/>.
        /// </summary>
        /// <seealso cref="GetGetterAs{TDst}"/>
        public static Delegate GetGetterAs(ColumnType typeDst, IRow row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst.IsPrimitive, nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.ColumnCount, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            var typeSrc = row.Schema.GetColumnType(col);
            Contracts.Check(typeSrc.IsPrimitive, "Source column type must be primitive");

            Func<ColumnType, ColumnType, IRow, int, ValueGetter<int>> del = GetGetterAsCore<int, int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType, typeDst.RawType);
            return (Delegate)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, row, col });
        }

        /// <summary>
        /// Given a destination type, IRow, and column index, return a ValueGetter{TDst} for the column
        /// with a conversion to typeDst, if needed.
        /// </summary>
        public static ValueGetter<TDst> GetGetterAs<TDst>(ColumnType typeDst, IRow row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst.IsPrimitive, nameof(typeDst));
            Contracts.CheckParam(typeDst.RawType == typeof(TDst), nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.ColumnCount, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            var typeSrc = row.Schema.GetColumnType(col);
            Contracts.Check(typeSrc.IsPrimitive, "Source column type must be primitive");

            Func<ColumnType, ColumnType, IRow, int, ValueGetter<TDst>> del = GetGetterAsCore<int, TDst>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType, typeof(TDst));
            return (ValueGetter<TDst>)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, row, col });
        }

        private static ValueGetter<TDst> GetGetterAsCore<TSrc, TDst>(ColumnType typeSrc, ColumnType typeDst, IRow row, int col)
        {
            Contracts.Assert(typeof(TSrc) == typeSrc.RawType);
            Contracts.Assert(typeof(TDst) == typeDst.RawType);

            var getter = row.GetGetter<TSrc>(col);
            bool identity;
            var conv = Conversions.Instance.GetStandardConversion<TSrc, TDst>(typeSrc, typeDst, out identity);
            if (identity)
            {
                Contracts.Assert(typeof(TSrc) == typeof(TDst));
                return (ValueGetter<TDst>)(Delegate)getter;
            }

            var src = default(TSrc);
            return
                (ref TDst dst) =>
                {
                    getter(ref src);
                    conv(ref src, ref dst);
                };
        }

        /// <summary>
        /// Given an IRow, and column index, return a function that utilizes the
        /// <see cref="Conversions.GetStringConversion{TSrc}(ColumnType)"/> on the input
        /// rows to map the values in the column, whatever type they may be, into a string
        /// builder. This method will obviously succeed only if there is a string conversion
        /// into the required type. This method can be useful if you want to output a value
        /// as a string in a generic way, but don't really care how you do it.
        /// </summary>
        public static ValueGetter<StringBuilder> GetGetterAsStringBuilder(IRow row, int col)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.ColumnCount, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            var typeSrc = row.Schema.GetColumnType(col);
            Contracts.Check(typeSrc.IsPrimitive, "Source column type must be primitive");
            return Utils.MarshalInvoke(GetGetterAsStringBuilderCore<int>, typeSrc.RawType, typeSrc, row, col);
        }

        private static ValueGetter<StringBuilder> GetGetterAsStringBuilderCore<TSrc>(ColumnType typeSrc, IRow row, int col)
        {
            Contracts.Assert(typeof(TSrc) == typeSrc.RawType);

            var getter = row.GetGetter<TSrc>(col);
            var conv = Conversions.Instance.GetStringConversion<TSrc>(typeSrc);

            var src = default(TSrc);
            return
                (ref StringBuilder dst) =>
                {
                    getter(ref src);
                    conv(ref src, ref dst);
                };
        }

        /// <summary>
        /// Given the item type, typeDst, a row, and column index, return a ValueGetter for the vector-valued
        /// column with a conversion to a vector of typeDst, if needed. This is the weakly typed version of
        /// <see cref="GetVecGetterAs{TDst}(PrimitiveType, IRow, int)"/>.
        /// </summary>
        public static Delegate GetVecGetterAs(PrimitiveType typeDst, IRow row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.ColumnCount, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            var typeSrc = row.Schema.GetColumnType(col);
            Contracts.Check(typeSrc.IsVector, "Source column type must be vector");

            Func<VectorType, PrimitiveType, GetterFactory, ValueGetter<VBuffer<int>>> del = GetVecGetterAsCore<int, int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.ItemType.RawType, typeDst.RawType);
            return (Delegate)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, GetterFactory.Create(row, col) });
        }

        /// <summary>
        /// Given the item type, typeDst, a row, and column index, return a ValueGetter{VBuffer{TDst}} for the
        /// vector-valued column with a conversion to a vector of typeDst, if needed.
        /// </summary>
        public static ValueGetter<VBuffer<TDst>> GetVecGetterAs<TDst>(PrimitiveType typeDst, IRow row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst.RawType == typeof(TDst), nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.ColumnCount, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            var typeSrc = row.Schema.GetColumnType(col);
            Contracts.Check(typeSrc.IsVector, "Source column type must be vector");

            Func<VectorType, PrimitiveType, GetterFactory, ValueGetter<VBuffer<TDst>>> del = GetVecGetterAsCore<int, TDst>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.ItemType.RawType, typeof(TDst));
            return (ValueGetter<VBuffer<TDst>>)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, GetterFactory.Create(row, col) });
        }

        /// <summary>
        /// Given the item type, typeDst, and a slot cursor, return a ValueGetter{VBuffer{TDst}} for the
        /// vector-valued column with a conversion to a vector of typeDst, if needed.
        /// </summary>
        public static ValueGetter<VBuffer<TDst>> GetVecGetterAs<TDst>(PrimitiveType typeDst, ISlotCursor cursor)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst.RawType == typeof(TDst), nameof(typeDst));

            var typeSrc = cursor.GetSlotType();

            Func<VectorType, PrimitiveType, GetterFactory, ValueGetter<VBuffer<TDst>>> del = GetVecGetterAsCore<int, TDst>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.ItemType.RawType, typeof(TDst));
            return (ValueGetter<VBuffer<TDst>>)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, GetterFactory.Create(cursor) });
        }

        /// <summary>
        /// A convenience wrapper to generalize the operation of fetching a <see cref="ValueGetter{TValue}"/>.
        /// </summary>
        private abstract class GetterFactory
        {
            public static GetterFactory Create(IRow row, int col)
            {
                return new RowImpl(row, col);
            }

            public static GetterFactory Create(ISlotCursor cursor)
            {
                return new SlotImpl(cursor);
            }

            public abstract ValueGetter<TValue> GetGetter<TValue>();

            private sealed class RowImpl : GetterFactory
            {
                private readonly IRow _row;
                private readonly int _col;

                public RowImpl(IRow row, int col)
                {
                    _row = row;
                    _col = col;
                }

                public override ValueGetter<TValue> GetGetter<TValue>()
                {
                    return _row.GetGetter<TValue>(_col);
                }
            }

            private sealed class SlotImpl : GetterFactory
            {
                private readonly ISlotCursor _cursor;

                public SlotImpl(ISlotCursor cursor)
                {
                    _cursor = cursor;
                }

                public override ValueGetter<TValue> GetGetter<TValue>()
                {
                    return _cursor.GetGetterWithVectorType<TValue>();
                }
            }
        }

        private static ValueGetter<VBuffer<TDst>> GetVecGetterAsCore<TSrc, TDst>(VectorType typeSrc, PrimitiveType typeDst, GetterFactory getterFact)
        {
            Contracts.Assert(typeof(TSrc) == typeSrc.ItemType.RawType);
            Contracts.Assert(typeof(TDst) == typeDst.RawType);
            Contracts.AssertValue(getterFact);

            var getter = getterFact.GetGetter<VBuffer<TSrc>>();
            bool identity;
            var conv = Conversions.Instance.GetStandardConversion<TSrc, TDst>(typeSrc.ItemType, typeDst, out identity);
            if (identity)
            {
                Contracts.Assert(typeof(TSrc) == typeof(TDst));
                return (ValueGetter<VBuffer<TDst>>)(Delegate)getter;
            }

            int size = typeSrc.VectorSize;
            var src = default(VBuffer<TSrc>);
            return (ref VBuffer<TDst> dst) =>
            {
                getter(ref src);
                if (size > 0)
                    Contracts.Check(src.Length == size);

                var values = dst.Values;
                var indices = dst.Indices;
                int count = src.Count;
                if (count > 0)
                {
                    if (Utils.Size(values) < count)
                        values = new TDst[count];

                    // REVIEW: This would be faster if there were loops for each std conversion.
                    // Consider adding those to the Conversions class.
                    for (int i = 0; i < count; i++)
                        conv(ref src.Values[i], ref values[i]);

                    if (!src.IsDense)
                    {
                        if (Utils.Size(indices) < count)
                            indices = new int[count];
                        Array.Copy(src.Indices, indices, count);
                    }
                }
                dst = new VBuffer<TDst>(src.Length, count, values, indices);
            };
        }

        /// <summary>
        /// This method returns a small helper delegate that returns whether we are at the start
        /// of a new group, that is, we have just started, or the key-value at indicated column
        /// is different than it was, in the last call. This is practically useful for determining
        /// group boundaries. Note that the delegate will return true on the first row.
        /// </summary>
        public static Func<bool> GetIsNewGroupDelegate(IRow cursor, int col)
        {
            Contracts.CheckValue(cursor, nameof(cursor));
            Contracts.Check(0 <= col && col < cursor.Schema.ColumnCount);
            ColumnType type = cursor.Schema.GetColumnType(col);
            Contracts.Check(type.IsKey);
            return Utils.MarshalInvoke(GetIsNewGroupDelegateCore<int>, type.RawType, cursor, col);
        }

        private static Func<bool> GetIsNewGroupDelegateCore<T>(IRow cursor, int col)
        {
            var getter = cursor.GetGetter<T>(col);
            bool first = true;
            T old = default(T);
            T val = default(T);
            var compare = EqualityComparer<T>.Default;
            return () =>
            {
                getter(ref val);
                if (first)
                {
                    first = false;
                    old = val;
                    return true;
                }
                if (compare.Equals(val, old))
                    return false;
                old = val;
                return true;
            };
        }

        public static Func<bool> GetIsNewBatchDelegate(IRow cursor, int batchSize)
        {
            Contracts.CheckParam(batchSize > 0, nameof(batchSize), "Batch size must be > 0");
            long lastNewBatchPosition = -1;
            return () =>
            {
                if (cursor.Position % batchSize != 0)
                    return false;

                // If the cursor just moved to a new batch, we need to return true.
                if (lastNewBatchPosition != cursor.Position)
                {
                    lastNewBatchPosition = cursor.Position;
                    return true;
                }

                // The cursor is already in the new batch, if the condition is tested again, we need to return false.
                return false;
            };
        }

        public static string TestGetLabelGetter(ColumnType type)
        {
            return TestGetLabelGetter(type, true);
        }

        public static string TestGetLabelGetter(ColumnType type, bool allowKeys)
        {
            if (type == NumberType.R4 || type == NumberType.R8 || type.IsBool)
                return null;

            if (allowKeys && type.IsKey)
                return null;

            return allowKeys ? "Expected R4, R8, Bool or Key type" : "Expected R4, R8 or Bool type";
        }

        public static ValueGetter<Single> GetLabelGetter(IRow cursor, int labelIndex)
        {
            var type = cursor.Schema.GetColumnType(labelIndex);

            if (type == NumberType.R4)
                return cursor.GetGetter<Single>(labelIndex);

            if (type == NumberType.R8)
            {
                var getSingleSrc = cursor.GetGetter<Double>(labelIndex);
                return
                    (ref Single dst) =>
                    {
                        Double src = Double.NaN;
                        getSingleSrc(ref src);
                        dst = Convert.ToSingle(src);
                    };
            }

            return GetLabelGetterNotFloat(cursor, labelIndex);
        }

        private static ValueGetter<Single> GetLabelGetterNotFloat(IRow cursor, int labelIndex)
        {
            var type = cursor.Schema.GetColumnType(labelIndex);

            Contracts.Assert(type != NumberType.R4 && type != NumberType.R8);

            // boolean type label mapping: True -> 1, False -> 0.
            if (type.IsBool)
            {
                var getBoolSrc = cursor.GetGetter<bool>(labelIndex);
                return
                    (ref Single dst) =>
                    {
                        bool src = default;
                        getBoolSrc(ref src);
                        dst = Convert.ToSingle(src);
                    };
            }

            Contracts.Check(type.IsKey, "Only floating point number, boolean, and key type values can be used as label.");
            Contracts.Assert(TestGetLabelGetter(type) == null);
            ulong keyMax = (ulong)type.KeyCount;
            if (keyMax == 0)
                keyMax = ulong.MaxValue;
            var getSrc = RowCursorUtils.GetGetterAs<ulong>(NumberType.U8, cursor, labelIndex);
            return
                (ref Single dst) =>
                {
                    ulong src = 0;
                    getSrc(ref src);
                    if (0 < src && src <= keyMax)
                        dst = src - 1;
                    else
                        dst = Single.NaN;
                };
        }

        public static ValueGetter<VBuffer<Single>> GetLabelGetter(ISlotCursor cursor)
        {
            var type = cursor.GetSlotType().ItemType;
            if (type == NumberType.R4)
                return cursor.GetGetter<Single>();
            if (type == NumberType.R8 || type.IsBool)
                return GetVecGetterAs<Single>(NumberType.R4, cursor);
            Contracts.Check(type.IsKey, "Only floating point number, boolean, and key type values can be used as label.");
            Contracts.Assert(TestGetLabelGetter(type) == null);
            ulong keyMax = (ulong)type.KeyCount;
            if (keyMax == 0)
                keyMax = ulong.MaxValue;
            var getSrc = RowCursorUtils.GetVecGetterAs<ulong>(NumberType.U8, cursor);
            VBuffer<ulong> src = default(VBuffer<ulong>);
            return
                (ref VBuffer<Single> dst) =>
                {
                    getSrc(ref src);
                    // Unfortunately defaults in one to not translate to defaults of the other,
                    // so this will not be sparsity preserving. Assume a dense output.
                    Single[] vals = dst.Values;
                    Utils.EnsureSize(ref vals, src.Length);
                    foreach (var kv in src.Items(all: true))
                    {
                        if (0 < kv.Value && kv.Value <= keyMax)
                            vals[kv.Key] = kv.Value - 1;
                        else
                            vals[kv.Key] = Single.NaN;
                    }
                    dst = new VBuffer<Single>(src.Length, vals, dst.Indices);
                };
        }

        /// <summary>
        /// Returns a row that is a deep in-memory copy of an input row. Note that inactive
        /// columns are allowed in this row, and their activity or inactivity will be reflected
        /// in the output row. Note that the deep copy includes a copy of the metadata as well.
        /// </summary>
        /// <param name="row">The input row</param>
        /// <returns>A deep in-memory copy of the input row</returns>
        public static IRow CloneRow(IRow row)
        {
            Contracts.CheckValue(row, nameof(row));
            return RowColumnUtils.GetRow(null,
                Utils.BuildArray(row.Schema.ColumnCount, c => RowColumnUtils.GetColumn(row, c)));
        }

        /// <summary>
        /// Fetches the value of the column by name, in the given row.
        /// Used by the evaluators to retrieve the metrics from the results IDataView.
        /// </summary>
        public static T Fetch<T>(IExceptionContext ectx, IRow row, string name)
        {
            if (!row.Schema.TryGetColumnIndex(name, out int col))
                throw ectx.Except($"Could not find column '{name}'");
            T val = default;
            row.GetGetter<T>(col)(ref val);
            return val;
        }

        /// <summary>
        /// Given a row, returns a one-row data view. This is useful for cases where you have a row, and you
        /// wish to use some facility normally only exposed to dataviews. (E.g., you have an <see cref="IRow"/>
        /// but want to save it somewhere using a <see cref="Microsoft.ML.Runtime.Data.IO.BinarySaver"/>.)
        /// Note that it is not possible for this method to ensure that the input <paramref name="row"/> does not
        /// change, so users of this convenience must take care of what they do with the input row or the data
        /// source it came from, while the returned dataview is potentially being used; if this is somehow
        /// difficult it may be wise to use <see cref="CloneRow"/> to first have a deep copy of the resulting row.
        /// </summary>
        /// <param name="env">An environment used to create the host for the resulting data view</param>
        /// <param name="row">A row, whose columns must all be active</param>
        /// <returns>A single-row data view incorporating that row</returns>
        public static IDataView RowAsDataView(IHostEnvironment env, IRow row)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(row, nameof(row));
            env.CheckParam(Enumerable.Range(0, row.Schema.ColumnCount).All(c => row.IsColumnActive(c)), nameof(row), "Some columns were inactive");
            return new OneRowDataView(env, row);
        }

        private sealed class OneRowDataView : IDataView
        {
            private readonly IRow _row;
            private readonly IHost _host; // A channel provider is required for creating the cursor.

            public ISchema Schema { get { return _row.Schema; } }
            public bool CanShuffle { get { return true; } } // The shuffling is even uniformly IID!! :)

            public OneRowDataView(IHostEnvironment env, IRow row)
            {
                Contracts.AssertValue(env);
                _host = env.Register("OneRowDataView");
                _host.AssertValue(row);
                _host.Assert(Enumerable.Range(0, row.Schema.ColumnCount).All(c => row.IsColumnActive(c)));

                _row = row;
            }

            public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
            {
                _host.CheckValue(needCol, nameof(needCol));
                _host.CheckValueOrNull(rand);
                bool[] active = Utils.BuildArray(Schema.ColumnCount, needCol);
                return new Cursor(_host, this, active);
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
            {
                _host.CheckValue(needCol, nameof(needCol));
                _host.CheckValueOrNull(rand);
                consolidator = null;
                return new IRowCursor[] { GetRowCursor(needCol, rand) };
            }

            public long? GetRowCount(bool lazy = true)
            {
                return 1;
            }

            private sealed class Cursor : RootCursorBase, IRowCursor
            {
                private readonly OneRowDataView _parent;
                private readonly bool[] _active;

                public ISchema Schema { get { return _parent.Schema; } }
                public override long Batch { get { return 0; } }

                public Cursor(IHost host, OneRowDataView parent, bool[] active)
                    : base(host)
                {
                    Ch.AssertValue(parent);
                    Ch.AssertValue(active);
                    Ch.Assert(active.Length == parent.Schema.ColumnCount);
                    _parent = parent;
                    _active = active;
                }

                protected override bool MoveNextCore()
                {
                    return State == CursorState.NotStarted;
                }

                public ValueGetter<TValue> GetGetter<TValue>(int col)
                {
                    Ch.CheckParam(0 <= col && col < Schema.ColumnCount, nameof(col));
                    Ch.CheckParam(IsColumnActive(col), nameof(col), "Requested column is not active");
                    var getter = _parent._row.GetGetter<TValue>(col);
                    return
                        (ref TValue val) =>
                        {
                            Ch.Check(IsGood, "Cannot call value getter in current state");
                            getter(ref val);
                        };
                }

                public bool IsColumnActive(int col)
                {
                    Ch.CheckParam(0 <= col && col < Schema.ColumnCount, nameof(col));
                    // We present the "illusion" that this column is not active, even though it must be
                    // in the input row.
                    Ch.Assert(_parent._row.IsColumnActive(col));
                    return _active[col];
                }

                public override ValueGetter<UInt128> GetIdGetter()
                {
                    return
                        (ref UInt128 val) =>
                        {
                            Ch.Check(IsGood, "Cannot call ID getter in current state");
                            val = new UInt128((ulong)Position, 0);
                        };
                }
            }
        }
    }
}
