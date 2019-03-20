// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    [BestFriend]
    internal static class RowCursorUtils
    {
        /// <summary>
        /// Returns an appropriate <see cref="ValueGetter{T}"/> for a row given an active column
        /// index, but as a delegate. The type parameter for the delegate will correspond to the
        /// raw type of the column.
        /// </summary>
        /// <param name="row">The row to get the getter for</param>
        /// <param name="col">The column index, which must be active on that row</param>
        /// <returns>The getter as a delegate</returns>
        public static Delegate GetGetterAsDelegate(DataViewRow row, int col)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(row.Schema[col]), nameof(col), "column was not active");

            Func<DataViewRow, int, Delegate> getGetter = GetGetterAsDelegateCore<int>;
            return Utils.MarshalInvoke(getGetter, row.Schema[col].Type.RawType, row, col);
        }

        private static Delegate GetGetterAsDelegateCore<TValue>(DataViewRow row, int col)
        {
            return row.GetGetter<TValue>(row.Schema[col]);
        }

        /// <summary>
        /// Given a destination type, IRow, and column index, return a ValueGetter for the column
        /// with a conversion to typeDst, if needed. This is a weakly typed version of
        /// <see cref="GetGetterAs{TDst}"/>.
        /// </summary>
        /// <seealso cref="GetGetterAs{TDst}"/>
        public static Delegate GetGetterAs(DataViewType typeDst, DataViewRow row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst is PrimitiveDataViewType, nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(row.Schema[col]), nameof(col), "column was not active");

            var typeSrc = row.Schema[col].Type;
            Contracts.Check(typeSrc is PrimitiveDataViewType, "Source column type must be primitive");

            Func<DataViewType, DataViewType, DataViewRow, int, ValueGetter<int>> del = GetGetterAsCore<int, int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType, typeDst.RawType);
            return (Delegate)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, row, col });
        }

        /// <summary>
        /// Given a destination type, IRow, and column index, return a ValueGetter{TDst} for the column
        /// with a conversion to typeDst, if needed.
        /// </summary>
        public static ValueGetter<TDst> GetGetterAs<TDst>(DataViewType typeDst, DataViewRow row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst is PrimitiveDataViewType, nameof(typeDst));
            Contracts.CheckParam(typeDst.RawType == typeof(TDst), nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(row.Schema[col]), nameof(col), "column was not active");

            var typeSrc = row.Schema[col].Type;
            Contracts.Check(typeSrc is PrimitiveDataViewType, "Source column type must be primitive");

            Func<DataViewType, DataViewType, DataViewRow, int, ValueGetter<TDst>> del = GetGetterAsCore<int, TDst>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType, typeof(TDst));
            return (ValueGetter<TDst>)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, row, col });
        }

        private static ValueGetter<TDst> GetGetterAsCore<TSrc, TDst>(DataViewType typeSrc, DataViewType typeDst, DataViewRow row, int col)
        {
            Contracts.Assert(typeof(TSrc) == typeSrc.RawType);
            Contracts.Assert(typeof(TDst) == typeDst.RawType);

            var getter = row.GetGetter<TSrc>(row.Schema[col]);
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
                    conv(in src, ref dst);
                };
        }

        /// <summary>
        /// Given an IRow, and column index, return a function that utilizes the
        /// <see cref="Conversions.GetStringConversion{TSrc}(DataViewType)"/> on the input
        /// rows to map the values in the column, whatever type they may be, into a string
        /// builder. This method will obviously succeed only if there is a string conversion
        /// into the required type. This method can be useful if you want to output a value
        /// as a string in a generic way, but don't really care how you do it.
        /// </summary>
        public static ValueGetter<StringBuilder> GetGetterAsStringBuilder(DataViewRow row, int col)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(row.Schema[col]), nameof(col), "column was not active");

            var typeSrc = row.Schema[col].Type;
            Contracts.Check(typeSrc is PrimitiveDataViewType, "Source column type must be primitive");
            return Utils.MarshalInvoke(GetGetterAsStringBuilderCore<int>, typeSrc.RawType, typeSrc, row, col);
        }

        private static ValueGetter<StringBuilder> GetGetterAsStringBuilderCore<TSrc>(DataViewType typeSrc, DataViewRow row, int col)
        {
            Contracts.Assert(typeof(TSrc) == typeSrc.RawType);

            var getter = row.GetGetter<TSrc>(row.Schema[col]);
            var conv = Conversions.Instance.GetStringConversion<TSrc>(typeSrc);

            var src = default(TSrc);
            return
                (ref StringBuilder dst) =>
                {
                    getter(ref src);
                    conv(in src, ref dst);
                };
        }

        /// <summary>
        /// Given the item type, typeDst, a row, and column index, return a ValueGetter for the vector-valued
        /// column with a conversion to a vector of typeDst, if needed. This is the weakly typed version of
        /// <see cref="GetVecGetterAs{TDst}(PrimitiveDataViewType, DataViewRow, int)"/>.
        /// </summary>
        public static Delegate GetVecGetterAs(PrimitiveDataViewType typeDst, DataViewRow row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(row.Schema[col]), nameof(col), "column was not active");

            var typeSrc = row.Schema[col].Type as VectorType;
            Contracts.Check(typeSrc != null, "Source column type must be vector");

            Func<VectorType, PrimitiveDataViewType, GetterFactory, ValueGetter<VBuffer<int>>> del = GetVecGetterAsCore<int, int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.ItemType.RawType, typeDst.RawType);
            return (Delegate)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, GetterFactory.Create(row, col) });
        }

        /// <summary>
        /// Given the item type, typeDst, a row, and column index, return a ValueGetter{VBuffer{TDst}} for the
        /// vector-valued column with a conversion to a vector of typeDst, if needed.
        /// </summary>
        public static ValueGetter<VBuffer<TDst>> GetVecGetterAs<TDst>(PrimitiveDataViewType typeDst, DataViewRow row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst.RawType == typeof(TDst), nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(row.Schema[col]), nameof(col), "column was not active");

            var typeSrc = row.Schema[col].Type as VectorType;
            Contracts.Check(typeSrc != null, "Source column type must be vector");

            Func<VectorType, PrimitiveDataViewType, GetterFactory, ValueGetter<VBuffer<TDst>>> del = GetVecGetterAsCore<int, TDst>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.ItemType.RawType, typeof(TDst));
            return (ValueGetter<VBuffer<TDst>>)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, GetterFactory.Create(row, col) });
        }

        /// <summary>
        /// Given the item type, typeDst, and a slot cursor, return a ValueGetter{VBuffer{TDst}} for the
        /// vector-valued column with a conversion to a vector of typeDst, if needed.
        /// </summary>
        [BestFriend]
        internal static ValueGetter<VBuffer<TDst>> GetVecGetterAs<TDst>(PrimitiveDataViewType typeDst, SlotCursor cursor)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst.RawType == typeof(TDst), nameof(typeDst));

            var typeSrc = cursor.GetSlotType();

            Func<VectorType, PrimitiveDataViewType, GetterFactory, ValueGetter<VBuffer<TDst>>> del = GetVecGetterAsCore<int, TDst>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.ItemType.RawType, typeof(TDst));
            return (ValueGetter<VBuffer<TDst>>)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, GetterFactory.Create(cursor) });
        }

        /// <summary>
        /// A convenience wrapper to generalize the operation of fetching a <see cref="ValueGetter{TValue}"/>.
        /// </summary>
        private abstract class GetterFactory
        {
            public static GetterFactory Create(DataViewRow row, int col)
            {
                return new RowImpl(row, col);
            }

            public static GetterFactory Create(SlotCursor cursor)
            {
                return new SlotImpl(cursor);
            }

            public abstract ValueGetter<TValue> GetGetter<TValue>();

            private sealed class RowImpl : GetterFactory
            {
                private readonly DataViewRow _row;
                private readonly int _col;

                public RowImpl(DataViewRow row, int col)
                {
                    _row = row;
                    _col = col;
                }

                public override ValueGetter<TValue> GetGetter<TValue>()
                {
                    return _row.GetGetter<TValue>(_row.Schema[_col]);
                }
            }

            private sealed class SlotImpl : GetterFactory
            {
                private readonly SlotCursor _cursor;

                public SlotImpl(SlotCursor cursor)
                {
                    _cursor = cursor;
                }

                public override ValueGetter<TValue> GetGetter<TValue>()
                {
                    return _cursor.GetGetterWithVectorType<TValue>();
                }
            }
        }

        private static ValueGetter<VBuffer<TDst>> GetVecGetterAsCore<TSrc, TDst>(VectorType typeSrc, PrimitiveDataViewType typeDst, GetterFactory getterFact)
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

            int size = typeSrc.Size;
            var src = default(VBuffer<TSrc>);
            return (ref VBuffer<TDst> dst) =>
            {
                getter(ref src);
                if (size > 0)
                    Contracts.Check(src.Length == size);

                var srcValues = src.GetValues();
                int count = srcValues.Length;
                var editor = VBufferEditor.Create(ref dst, src.Length, count);
                if (count > 0)
                {
                    // REVIEW: This would be faster if there were loops for each std conversion.
                    // Consider adding those to the Conversions class.
                    for (int i = 0; i < count; i++)
                        conv(in srcValues[i], ref editor.Values[i]);

                    if (!src.IsDense)
                    {
                        var srcIndices = src.GetIndices();
                        srcIndices.CopyTo(editor.Indices);
                    }
                }
                dst = editor.Commit();
            };
        }

        /// <summary>
        /// This method returns a small helper delegate that returns whether we are at the start
        /// of a new group, that is, we have just started, or the key-value at indicated column
        /// is different than it was, in the last call. This is practically useful for determining
        /// group boundaries. Note that the delegate will return true on the first row.
        /// </summary>
        public static Func<bool> GetIsNewGroupDelegate(DataViewRow cursor, int col)
        {
            Contracts.CheckValue(cursor, nameof(cursor));
            Contracts.Check(0 <= col && col < cursor.Schema.Count);
            DataViewType type = cursor.Schema[col].Type;
            Contracts.Check(type is KeyType);
            return Utils.MarshalInvoke(GetIsNewGroupDelegateCore<int>, type.RawType, cursor, col);
        }

        private static Func<bool> GetIsNewGroupDelegateCore<T>(DataViewRow cursor, int col)
        {
            var getter = cursor.GetGetter<T>(cursor.Schema[col]);
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

        public static string TestGetLabelGetter(DataViewType type)
        {
            return TestGetLabelGetter(type, true);
        }

        public static string TestGetLabelGetter(DataViewType type, bool allowKeys)
        {
            if (type == NumberDataViewType.Single || type == NumberDataViewType.Double || type is BooleanDataViewType)
                return null;

            if (allowKeys && type is KeyType)
                return null;

            return allowKeys ? "Expected R4, R8, Bool or Key type" : "Expected R4, R8 or Bool type";
        }

        public static ValueGetter<Single> GetLabelGetter(DataViewRow cursor, int labelIndex)
        {
            var type = cursor.Schema[labelIndex].Type;

            if (type == NumberDataViewType.Single)
                return cursor.GetGetter<Single>(cursor.Schema[labelIndex]);

            if (type == NumberDataViewType.Double)
            {
                var getSingleSrc = cursor.GetGetter<Double>(cursor.Schema[labelIndex]);
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

        private static ValueGetter<Single> GetLabelGetterNotFloat(DataViewRow cursor, int labelIndex)
        {
            var type = cursor.Schema[labelIndex].Type;

            Contracts.Assert(type != NumberDataViewType.Single && type != NumberDataViewType.Double);

            // boolean type label mapping: True -> 1, False -> 0.
            if (type is BooleanDataViewType)
            {
                var getBoolSrc = cursor.GetGetter<bool>(cursor.Schema[labelIndex]);
                return
                    (ref Single dst) =>
                    {
                        bool src = default;
                        getBoolSrc(ref src);
                        dst = Convert.ToSingle(src);
                    };
            }

            if (!(type is KeyType keyType))
                throw Contracts.Except("Only floating point number, boolean, and key type values can be used as label.");

            Contracts.Assert(TestGetLabelGetter(type) == null);
            ulong keyMax = (ulong)keyType.Count;
            if (keyMax == 0)
                keyMax = ulong.MaxValue;
            var getSrc = RowCursorUtils.GetGetterAs<ulong>(NumberDataViewType.UInt64, cursor, labelIndex);
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

        [BestFriend]
        internal static ValueGetter<VBuffer<Single>> GetLabelGetter(SlotCursor cursor)
        {
            var type = cursor.GetSlotType().ItemType;
            if (type == NumberDataViewType.Single)
                return cursor.GetGetter<Single>();
            if (type == NumberDataViewType.Double || type is BooleanDataViewType)
                return GetVecGetterAs<Single>(NumberDataViewType.Single, cursor);
            if (!(type is KeyType keyType))
            {
                throw Contracts.Except("Only floating point number, boolean, and key type values can be used as label.");
            }
            Contracts.Assert(TestGetLabelGetter(type) == null);
            ulong keyMax = (ulong)keyType.Count;
            if (keyMax == 0)
                keyMax = ulong.MaxValue;
            var getSrc = RowCursorUtils.GetVecGetterAs<ulong>(NumberDataViewType.UInt64, cursor);
            VBuffer<ulong> src = default(VBuffer<ulong>);
            return
                (ref VBuffer<Single> dst) =>
                {
                    getSrc(ref src);
                    // Unfortunately defaults in one to not translate to defaults of the other,
                    // so this will not be sparsity preserving. Assume a dense output.
                    var editor = VBufferEditor.Create(ref dst, src.Length);
                    foreach (var kv in src.Items(all: true))
                    {
                        if (0 < kv.Value && kv.Value <= keyMax)
                            editor.Values[kv.Key] = kv.Value - 1;
                        else
                            editor.Values[kv.Key] = Single.NaN;
                    }
                    dst = editor.Commit();
                };
        }

        /// <summary>
        /// Fetches the value of the column by name, in the given row.
        /// Used by the evaluators to retrieve the metrics from the results IDataView.
        /// </summary>
        public static T Fetch<T>(IExceptionContext ectx, DataViewRow row, string name)
        {
            if (!row.Schema.TryGetColumnIndex(name, out int col))
                throw ectx.Except($"Could not find column '{name}'");
            T val = default;
            row.GetGetter<T>(row.Schema[col])(ref val);
            return val;
        }

        /// <summary>
        /// Given a row, returns a one-row data view. This is useful for cases where you have a row, and you
        /// wish to use some facility normally only exposed to dataviews. (For example, you have an <see cref="DataViewRow"/>
        /// but want to save it somewhere using a <see cref="Microsoft.ML.Data.IO.BinarySaver"/>.)
        /// Note that it is not possible for this method to ensure that the input <paramref name="row"/> does not
        /// change, so users of this convenience must take care of what they do with the input row or the data
        /// source it came from, while the returned dataview is potentially being used.
        /// </summary>
        /// <param name="env">An environment used to create the host for the resulting data view</param>
        /// <param name="row">A row, whose columns must all be active</param>
        /// <returns>A single-row data view incorporating that row</returns>
        public static IDataView RowAsDataView(IHostEnvironment env, DataViewRow row)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(row, nameof(row));
            env.CheckParam(Enumerable.Range(0, row.Schema.Count).All(c => row.IsColumnActive(row.Schema[c])), nameof(row), "Some columns were inactive");
            return new OneRowDataView(env, row);
        }

        /// <summary>
        /// Given a collection of <see cref="DataViewSchema.Column"/>, that is a subset of the Schema of the data, create a predicate,
        /// that when passed a column index, will return <langword>true</langword> or <langword>false</langword>, based on whether
        /// the column with the given <see cref="DataViewSchema.Column.Index"/> is part of the <paramref name="columnsNeeded"/>.
        /// </summary>
        /// <param name="columnsNeeded">The subset of columns from the <see cref="DataViewSchema"/> that are needed from this <see cref="DataViewRowCursor"/>.</param>
        /// <param name="sourceSchema">The <see cref="DataViewSchema"/> from where the columnsNeeded originate.</param>
        [BestFriend]
        internal static Func<int, bool> FromColumnsToPredicate(IEnumerable<DataViewSchema.Column> columnsNeeded, DataViewSchema sourceSchema)
        {
            Contracts.CheckValue(columnsNeeded, nameof(columnsNeeded));
            Contracts.CheckValue(sourceSchema, nameof(sourceSchema));

            bool[] indicesRequested = new bool[sourceSchema.Count];

            foreach (var col in columnsNeeded)
            {
                if (col.Index >= indicesRequested.Length)
                    throw Contracts.Except($"The requested column: {col} is not part of the {nameof(sourceSchema)}");

                indicesRequested[col.Index] = true;
            }

            return c => indicesRequested[c];
        }

        private sealed class OneRowDataView : IDataView
        {
            private readonly DataViewRow _row;
            private readonly IHost _host; // A channel provider is required for creating the cursor.

            public DataViewSchema Schema => _row.Schema;
            public bool CanShuffle => true; // The shuffling is even uniformly IID!! :)

            public OneRowDataView(IHostEnvironment env, DataViewRow row)
            {
                Contracts.AssertValue(env);
                _host = env.Register("OneRowDataView");
                _host.AssertValue(row);
                _host.Assert(Enumerable.Range(0, row.Schema.Count).All(c => row.IsColumnActive(row.Schema[c])));

                _row = row;
            }

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnNeeded, Random rand = null)
            {
                _host.CheckValueOrNull(rand);
                bool[] active = Utils.BuildArray(Schema.Count, columnNeeded);
                return new Cursor(_host, this, active);
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnNeeded, int n, Random rand = null)
            {
                _host.CheckValueOrNull(rand);
                return new DataViewRowCursor[] { GetRowCursor(columnNeeded, rand) };
            }

            public long? GetRowCount()
            {
                return 1;
            }

            private sealed class Cursor : RootCursorBase
            {
                private readonly OneRowDataView _parent;
                private readonly bool[] _active;

                public override DataViewSchema Schema => _parent.Schema;
                public override long Batch => 0;

                public Cursor(IHost host, OneRowDataView parent, bool[] active)
                    : base(host)
                {
                    Ch.AssertValue(parent);
                    Ch.AssertValue(active);
                    Ch.Assert(active.Length == parent.Schema.Count);
                    _parent = parent;
                    _active = active;
                }

                protected override bool MoveNextCore() => Position < 0;

                /// <summary>
                /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
                /// This throws if the column is not active in this row, or if the type
                /// <typeparamref name="TValue"/> differs from this column's type.
                /// </summary>
                /// <typeparam name="TValue"> is the column's content type.</typeparam>
                /// <param name="column"> is the output column whose getter should be returned.</param>
                public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
                {
                    Ch.CheckParam(column.Index < Schema.Count, nameof(column));
                    Ch.CheckParam(IsColumnActive(column), nameof(column.Index), "Requested column is not active.");

                    var getter = _parent._row.GetGetter<TValue>(column);
                    return
                        (ref TValue val) =>
                        {
                            Ch.Check(IsGood, "Cannot call value getter in current state");
                            getter(ref val);
                        };
                }

                /// <summary>
                /// Returns whether the given column is active in this row.
                /// </summary>
                public override bool IsColumnActive(DataViewSchema.Column column)
                {
                    Ch.CheckParam(column.Index < Schema.Count, nameof(column));
                    // We present the "illusion" that this column is not active, even though it must be
                    // in the input row.
                    Ch.Assert(_parent._row.IsColumnActive(column));
                    return _active[column.Index];
                }

                public override ValueGetter<DataViewRowId> GetIdGetter()
                {
                    return
                        (ref DataViewRowId val) =>
                        {
                            Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                            val = new DataViewRowId((ulong)Position, 0);
                        };
                }
            }
        }

        /// <summary>
        /// This is an error message meant to be used in the situation where a user calls a delegate as returned from
        /// <see cref="DataViewRow.GetIdGetter"/> or <see cref="DataViewRow.GetGetter{TValue}(DataViewSchema.Column)"/>.
        /// </summary>
        [BestFriend]
        internal const string FetchValueStateError = "Values cannot be fetched at this time. This method was called either before the first call to "
            + nameof(DataViewRowCursor.MoveNext) + ", or at any point after that method returned false.";
    }
}
