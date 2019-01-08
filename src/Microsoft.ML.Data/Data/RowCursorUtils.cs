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

namespace Microsoft.ML.Data
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
        public static Delegate GetGetterAsDelegate(Row row, int col)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            Func<Row, int, Delegate> getGetter = GetGetterAsDelegateCore<int>;
            return Utils.MarshalInvoke(getGetter, row.Schema[col].Type.RawType, row, col);
        }

        private static Delegate GetGetterAsDelegateCore<TValue>(Row row, int col)
        {
            return row.GetGetter<TValue>(col);
        }

        /// <summary>
        /// Given a destination type, IRow, and column index, return a ValueGetter for the column
        /// with a conversion to typeDst, if needed. This is a weakly typed version of
        /// <see cref="GetGetterAs{TDst}"/>.
        /// </summary>
        /// <seealso cref="GetGetterAs{TDst}"/>
        public static Delegate GetGetterAs(ColumnType typeDst, Row row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst is PrimitiveType, nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            var typeSrc = row.Schema[col].Type;
            Contracts.Check(typeSrc is PrimitiveType, "Source column type must be primitive");

            Func<ColumnType, ColumnType, Row, int, ValueGetter<int>> del = GetGetterAsCore<int, int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType, typeDst.RawType);
            return (Delegate)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, row, col });
        }

        /// <summary>
        /// Given a destination type, IRow, and column index, return a ValueGetter{TDst} for the column
        /// with a conversion to typeDst, if needed.
        /// </summary>
        public static ValueGetter<TDst> GetGetterAs<TDst>(ColumnType typeDst, Row row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst is PrimitiveType, nameof(typeDst));
            Contracts.CheckParam(typeDst.RawType == typeof(TDst), nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            var typeSrc = row.Schema[col].Type;
            Contracts.Check(typeSrc is PrimitiveType, "Source column type must be primitive");

            Func<ColumnType, ColumnType, Row, int, ValueGetter<TDst>> del = GetGetterAsCore<int, TDst>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType, typeof(TDst));
            return (ValueGetter<TDst>)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, row, col });
        }

        private static ValueGetter<TDst> GetGetterAsCore<TSrc, TDst>(ColumnType typeSrc, ColumnType typeDst, Row row, int col)
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
                    conv(in src, ref dst);
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
        public static ValueGetter<StringBuilder> GetGetterAsStringBuilder(Row row, int col)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            var typeSrc = row.Schema[col].Type;
            Contracts.Check(typeSrc is PrimitiveType, "Source column type must be primitive");
            return Utils.MarshalInvoke(GetGetterAsStringBuilderCore<int>, typeSrc.RawType, typeSrc, row, col);
        }

        private static ValueGetter<StringBuilder> GetGetterAsStringBuilderCore<TSrc>(ColumnType typeSrc, Row row, int col)
        {
            Contracts.Assert(typeof(TSrc) == typeSrc.RawType);

            var getter = row.GetGetter<TSrc>(col);
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
        /// <see cref="GetVecGetterAs{TDst}(PrimitiveType, Row, int)"/>.
        /// </summary>
        public static Delegate GetVecGetterAs(PrimitiveType typeDst, Row row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            var typeSrc = row.Schema[col].Type;
            Contracts.Check(typeSrc.IsVector, "Source column type must be vector");

            Func<VectorType, PrimitiveType, GetterFactory, ValueGetter<VBuffer<int>>> del = GetVecGetterAsCore<int, int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.ItemType.RawType, typeDst.RawType);
            return (Delegate)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, GetterFactory.Create(row, col) });
        }

        /// <summary>
        /// Given the item type, typeDst, a row, and column index, return a ValueGetter{VBuffer{TDst}} for the
        /// vector-valued column with a conversion to a vector of typeDst, if needed.
        /// </summary>
        public static ValueGetter<VBuffer<TDst>> GetVecGetterAs<TDst>(PrimitiveType typeDst, Row row, int col)
        {
            Contracts.CheckValue(typeDst, nameof(typeDst));
            Contracts.CheckParam(typeDst.RawType == typeof(TDst), nameof(typeDst));
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.Count, nameof(col));
            Contracts.CheckParam(row.IsColumnActive(col), nameof(col), "column was not active");

            var typeSrc = row.Schema[col].Type;
            Contracts.Check(typeSrc.IsVector, "Source column type must be vector");

            Func<VectorType, PrimitiveType, GetterFactory, ValueGetter<VBuffer<TDst>>> del = GetVecGetterAsCore<int, TDst>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.ItemType.RawType, typeof(TDst));
            return (ValueGetter<VBuffer<TDst>>)methodInfo.Invoke(null, new object[] { typeSrc, typeDst, GetterFactory.Create(row, col) });
        }

        /// <summary>
        /// Given the item type, typeDst, and a slot cursor, return a ValueGetter{VBuffer{TDst}} for the
        /// vector-valued column with a conversion to a vector of typeDst, if needed.
        /// </summary>
        public static ValueGetter<VBuffer<TDst>> GetVecGetterAs<TDst>(PrimitiveType typeDst, SlotCursor cursor)
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
            public static GetterFactory Create(Row row, int col)
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
                private readonly Row _row;
                private readonly int _col;

                public RowImpl(Row row, int col)
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
        public static Func<bool> GetIsNewGroupDelegate(Row cursor, int col)
        {
            Contracts.CheckValue(cursor, nameof(cursor));
            Contracts.Check(0 <= col && col < cursor.Schema.Count);
            ColumnType type = cursor.Schema[col].Type;
            Contracts.Check(type.IsKey);
            return Utils.MarshalInvoke(GetIsNewGroupDelegateCore<int>, type.RawType, cursor, col);
        }

        private static Func<bool> GetIsNewGroupDelegateCore<T>(Row cursor, int col)
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

        public static string TestGetLabelGetter(ColumnType type)
        {
            return TestGetLabelGetter(type, true);
        }

        public static string TestGetLabelGetter(ColumnType type, bool allowKeys)
        {
            if (type == NumberType.R4 || type == NumberType.R8 || type is BoolType)
                return null;

            if (allowKeys && type.IsKey)
                return null;

            return allowKeys ? "Expected R4, R8, Bool or Key type" : "Expected R4, R8 or Bool type";
        }

        public static ValueGetter<Single> GetLabelGetter(Row cursor, int labelIndex)
        {
            var type = cursor.Schema[labelIndex].Type;

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

        private static ValueGetter<Single> GetLabelGetterNotFloat(Row cursor, int labelIndex)
        {
            var type = cursor.Schema[labelIndex].Type;

            Contracts.Assert(type != NumberType.R4 && type != NumberType.R8);

            // boolean type label mapping: True -> 1, False -> 0.
            if (type is BoolType)
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

        public static ValueGetter<VBuffer<Single>> GetLabelGetter(SlotCursor cursor)
        {
            var type = cursor.GetSlotType().ItemType;
            if (type == NumberType.R4)
                return cursor.GetGetter<Single>();
            if (type == NumberType.R8 || type is BoolType)
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
        public static T Fetch<T>(IExceptionContext ectx, Row row, string name)
        {
            if (!row.Schema.TryGetColumnIndex(name, out int col))
                throw ectx.Except($"Could not find column '{name}'");
            T val = default;
            row.GetGetter<T>(col)(ref val);
            return val;
        }

        /// <summary>
        /// Given a row, returns a one-row data view. This is useful for cases where you have a row, and you
        /// wish to use some facility normally only exposed to dataviews. (For example, you have an <see cref="Row"/>
        /// but want to save it somewhere using a <see cref="Microsoft.ML.Data.IO.BinarySaver"/>.)
        /// Note that it is not possible for this method to ensure that the input <paramref name="row"/> does not
        /// change, so users of this convenience must take care of what they do with the input row or the data
        /// source it came from, while the returned dataview is potentially being used.
        /// </summary>
        /// <param name="env">An environment used to create the host for the resulting data view</param>
        /// <param name="row">A row, whose columns must all be active</param>
        /// <returns>A single-row data view incorporating that row</returns>
        public static IDataView RowAsDataView(IHostEnvironment env, Row row)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(row, nameof(row));
            env.CheckParam(Enumerable.Range(0, row.Schema.Count).All(c => row.IsColumnActive(c)), nameof(row), "Some columns were inactive");
            return new OneRowDataView(env, row);
        }

        private sealed class OneRowDataView : IDataView
        {
            private readonly Row _row;
            private readonly IHost _host; // A channel provider is required for creating the cursor.

            public Schema Schema => _row.Schema;
            public bool CanShuffle => true; // The shuffling is even uniformly IID!! :)

            public OneRowDataView(IHostEnvironment env, Row row)
            {
                Contracts.AssertValue(env);
                _host = env.Register("OneRowDataView");
                _host.AssertValue(row);
                _host.Assert(Enumerable.Range(0, row.Schema.Count).All(c => row.IsColumnActive(c)));

                _row = row;
            }

            public RowCursor GetRowCursor(Func<int, bool> needCol, Random rand = null)
            {
                _host.CheckValue(needCol, nameof(needCol));
                _host.CheckValueOrNull(rand);
                bool[] active = Utils.BuildArray(Schema.Count, needCol);
                return new Cursor(_host, this, active);
            }

            public RowCursor[] GetRowCursorSet(Func<int, bool> needCol, int n, Random rand = null)
            {
                _host.CheckValue(needCol, nameof(needCol));
                _host.CheckValueOrNull(rand);
                return new RowCursor[] { GetRowCursor(needCol, rand) };
            }

            public long? GetRowCount()
            {
                return 1;
            }

            private sealed class Cursor : RootCursorBase
            {
                private readonly OneRowDataView _parent;
                private readonly bool[] _active;

                public override Schema Schema => _parent.Schema;
                public override long Batch { get { return 0; } }

                public Cursor(IHost host, OneRowDataView parent, bool[] active)
                    : base(host)
                {
                    Ch.AssertValue(parent);
                    Ch.AssertValue(active);
                    Ch.Assert(active.Length == parent.Schema.Count);
                    _parent = parent;
                    _active = active;
                }

                protected override bool MoveNextCore()
                {
                    return State == CursorState.NotStarted;
                }

                public override ValueGetter<TValue> GetGetter<TValue>(int col)
                {
                    Ch.CheckParam(0 <= col && col < Schema.Count, nameof(col));
                    Ch.CheckParam(IsColumnActive(col), nameof(col), "Requested column is not active");
                    var getter = _parent._row.GetGetter<TValue>(col);
                    return
                        (ref TValue val) =>
                        {
                            Ch.Check(IsGood, "Cannot call value getter in current state");
                            getter(ref val);
                        };
                }

                public override bool IsColumnActive(int col)
                {
                    Ch.CheckParam(0 <= col && col < Schema.Count, nameof(col));
                    // We present the "illusion" that this column is not active, even though it must be
                    // in the input row.
                    Ch.Assert(_parent._row.IsColumnActive(col));
                    return _active[col];
                }

                public override ValueGetter<RowId> GetIdGetter()
                {
                    return
                        (ref RowId val) =>
                        {
                            Ch.Check(IsGood, "Cannot call ID getter in current state");
                            val = new RowId((ulong)Position, 0);
                        };
                }
            }
        }
    }
}
