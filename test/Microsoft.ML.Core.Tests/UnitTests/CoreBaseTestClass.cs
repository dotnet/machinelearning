﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFrameworkCommon;
using Xunit.Abstractions;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public class CoreBaseTestClass : BaseTestBaseline
    {
        public CoreBaseTestClass(ITestOutputHelper output)
            : base(output)
        {
        }

        protected bool Failed()
        {
            return false;
        }

        protected Func<bool> GetIdComparer(DataViewRow r1, DataViewRow r2, out ValueGetter<DataViewRowId> idGetter)
        {
            var g1 = r1.GetIdGetter();
            idGetter = g1;
            var g2 = r2.GetIdGetter();
            DataViewRowId v1 = default(DataViewRowId);
            DataViewRowId v2 = default(DataViewRowId);
            return
                () =>
                {
                    g1(ref v1);
                    g2(ref v2);
                    return v1.Equals(v2);
                };
        }

        protected Func<bool> GetComparerOne<T>(DataViewRow r1, DataViewRow r2, int col, Func<T, T, bool> fn)
        {
            var g1 = r1.GetGetter<T>(r1.Schema[col]);
            var g2 = r2.GetGetter<T>(r2.Schema[col]);
            T v1 = default(T);
            T v2 = default(T);
            return
                () =>
                {
                    g1(ref v1);
                    g2(ref v2);
                    if (!fn(v1, v2))
                        return false;
                    return true;
                };
        }

        private const Double DoubleEps = 1e-9;
        private static bool EqualWithEps(Double x, Double y)
        {
            // bitwise comparison is needed because Abs(Inf-Inf) and Abs(NaN-NaN) are not 0s.
            return FloatUtils.GetBits(x) == FloatUtils.GetBits(y) || Math.Abs(x - y) < DoubleEps;
        }
        protected Func<bool> GetComparerVec<T>(DataViewRow r1, DataViewRow r2, int col, int size, Func<T, T, bool> fn)
        {
            var g1 = r1.GetGetter<VBuffer<T>>(r1.Schema[col]);
            var g2 = r2.GetGetter<VBuffer<T>>(r2.Schema[col]);
            var v1 = default(VBuffer<T>);
            var v2 = default(VBuffer<T>);
            return
                () =>
                {
                    g1(ref v1);
                    g2(ref v2);
                    return TestCommon.CompareVec<T>(in v1, in v2, size, fn);
                };
        }

        protected Func<bool> GetColumnComparer(DataViewRow r1, DataViewRow r2, int col, DataViewType type, bool exactDoubles)
        {
            if (type is VectorDataViewType vecType)
            {
                int size = vecType.Size;
                Contracts.Assert(size >= 0);
                var result = vecType.ItemType.RawType.TryGetDataKind(out var kind);
                Contracts.Assert(result);

                switch (kind)
                {
                    case InternalDataKind.I1:
                        return GetComparerVec<sbyte>(r1, r2, col, size, (x, y) => x == y);
                    case InternalDataKind.U1:
                        return GetComparerVec<byte>(r1, r2, col, size, (x, y) => x == y);
                    case InternalDataKind.I2:
                        return GetComparerVec<short>(r1, r2, col, size, (x, y) => x == y);
                    case InternalDataKind.U2:
                        return GetComparerVec<ushort>(r1, r2, col, size, (x, y) => x == y);
                    case InternalDataKind.I4:
                        return GetComparerVec<int>(r1, r2, col, size, (x, y) => x == y);
                    case InternalDataKind.U4:
                        return GetComparerVec<uint>(r1, r2, col, size, (x, y) => x == y);
                    case InternalDataKind.I8:
                        return GetComparerVec<long>(r1, r2, col, size, (x, y) => x == y);
                    case InternalDataKind.U8:
                        return GetComparerVec<ulong>(r1, r2, col, size, (x, y) => x == y);
                    case InternalDataKind.R4:
                        return GetComparerVec<Single>(r1, r2, col, size, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                    case InternalDataKind.R8:
                        if (exactDoubles)
                            return GetComparerVec<Double>(r1, r2, col, size, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                        else
                            return GetComparerVec<Double>(r1, r2, col, size, EqualWithEps);
                    case InternalDataKind.Text:
                        return GetComparerVec<ReadOnlyMemory<char>>(r1, r2, col, size, (a, b) => a.Span.SequenceEqual(b.Span));
                    case InternalDataKind.Bool:
                        return GetComparerVec<bool>(r1, r2, col, size, (x, y) => x == y);
                    case InternalDataKind.TimeSpan:
                        return GetComparerVec<TimeSpan>(r1, r2, col, size, (x, y) => x.Ticks == y.Ticks);
                    case InternalDataKind.DT:
                        return GetComparerVec<DateTime>(r1, r2, col, size, (x, y) => x.Ticks == y.Ticks);
                    case InternalDataKind.DZ:
                        return GetComparerVec<DateTimeOffset>(r1, r2, col, size, (x, y) => x.Equals(y));
                    case InternalDataKind.UG:
                        return GetComparerVec<DataViewRowId>(r1, r2, col, size, (x, y) => x.Equals(y));
                }
            }
            else
            {
                var result = type.RawType.TryGetDataKind(out var kind);
                Contracts.Assert(result);
                switch (kind)
                {
                    case InternalDataKind.I1:
                        return GetComparerOne<sbyte>(r1, r2, col, (x, y) => x == y);
                    case InternalDataKind.U1:
                        return GetComparerOne<byte>(r1, r2, col, (x, y) => x == y);
                    case InternalDataKind.I2:
                        return GetComparerOne<short>(r1, r2, col, (x, y) => x == y);
                    case InternalDataKind.U2:
                        return GetComparerOne<ushort>(r1, r2, col, (x, y) => x == y);
                    case InternalDataKind.I4:
                        return GetComparerOne<int>(r1, r2, col, (x, y) => x == y);
                    case InternalDataKind.U4:
                        return GetComparerOne<uint>(r1, r2, col, (x, y) => x == y);
                    case InternalDataKind.I8:
                        return GetComparerOne<long>(r1, r2, col, (x, y) => x == y);
                    case InternalDataKind.U8:
                        return GetComparerOne<ulong>(r1, r2, col, (x, y) => x == y);
                    case InternalDataKind.R4:
                        return GetComparerOne<Single>(r1, r2, col, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                    case InternalDataKind.R8:
                        if (exactDoubles)
                            return GetComparerOne<Double>(r1, r2, col, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                        else
                            return GetComparerOne<Double>(r1, r2, col, EqualWithEps);
                    case InternalDataKind.Text:
                        return GetComparerOne<ReadOnlyMemory<char>>(r1, r2, col, (a, b) => a.Span.SequenceEqual(b.Span));
                    case InternalDataKind.Bool:
                        return GetComparerOne<bool>(r1, r2, col, (x, y) => x == y);
                    case InternalDataKind.TimeSpan:
                        return GetComparerOne<TimeSpan>(r1, r2, col, (x, y) => x.Ticks == y.Ticks);
                    case InternalDataKind.DT:
                        return GetComparerOne<DateTime>(r1, r2, col, (x, y) => x.Ticks == y.Ticks);
                    case InternalDataKind.DZ:
                        return GetComparerOne<DateTimeOffset>(r1, r2, col, (x, y) => x.Equals(y));
                    case InternalDataKind.UG:
                        return GetComparerOne<DataViewRowId>(r1, r2, col, (x, y) => x.Equals(y));
                }
            }

#if !CORECLR // REVIEW: Port Picture type to CoreTLC.
            if (type is PictureType)
            {
                var g1 = r1.GetGetter<Picture>(col);
                var g2 = r2.GetGetter<Picture>(col);
                Picture v1 = null;
                Picture v2 = null;
                return
                    () =>
                    {
                        g1(ref v1);
                        g2(ref v2);
                        return ComparePicture(v1, v2);
                    };
            }
#endif

            throw Contracts.Except("Unknown type in GetColumnComparer: '{0}'", type);
        }

        protected bool CheckSameValues(IDataView view1, IDataView view2, bool exactTypes = true, bool exactDoubles = true, bool checkId = true)
        {
            Contracts.Assert(view1.Schema.Count == view2.Schema.Count);

            bool all = true;
            bool tmp;

            using (var curs1 = view1.GetRowCursorForAllColumns())
            using (var curs2 = view2.GetRowCursorForAllColumns())
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, true);
            }
            Check(tmp, "All same failed");
            all &= tmp;

            var view2EvenCols = view2.Schema.Where(col => (col.Index & 1) == 0);
            using (var curs1 = view1.GetRowCursorForAllColumns())
            using (var curs2 = view2.GetRowCursor(view2EvenCols))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, false);
            }
            Check(tmp, "Even same failed");
            all &= tmp;

            var view2OddCols = view2.Schema.Where(col => (col.Index & 1) == 0);
            using (var curs1 = view1.GetRowCursorForAllColumns())
            using (var curs2 = view2.GetRowCursor(view2OddCols))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, false);
            }
            Check(tmp, "Odd same failed");

            using (var curs1 = view1.GetRowCursor(view1.Schema))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                tmp = CheckSameValues(curs1, view2, exactTypes, exactDoubles, checkId);
            }
            Check(tmp, "Single value same failed");

            all &= tmp;
            return all;
        }

        protected bool CheckSameValues(DataViewRowCursor curs1, DataViewRowCursor curs2, bool exactTypes, bool exactDoubles, bool checkId, bool checkIdCollisions = true)
        {
            Contracts.Assert(curs1.Schema.Count == curs2.Schema.Count);

            // Get the comparison delegates for each column.
            int colLim = curs1.Schema.Count;
            Func<bool>[] comps = new Func<bool>[colLim];
            for (int col = 0; col < colLim; col++)
            {
                var f1 = curs1.IsColumnActive(curs1.Schema[col]);
                var f2 = curs2.IsColumnActive(curs2.Schema[col]);

                if (f1 && f2)
                {
                    var type1 = curs1.Schema[col].Type;
                    var type2 = curs2.Schema[col].Type;
                    if (!TestCommon.EqualTypes(type1, type2, exactTypes))
                    {
                        Fail($"Different types {type1} and {type2}");
                        return Failed();
                    }
                    comps[col] = GetColumnComparer(curs1, curs2, col, type1, exactDoubles);
                }
            }
            ValueGetter<DataViewRowId> idGetter = null;
            Func<bool> idComp = checkId ? GetIdComparer(curs1, curs2, out idGetter) : null;
            HashSet<DataViewRowId> idsSeen = null;
            if (checkIdCollisions && idGetter == null)
                idGetter = curs1.GetIdGetter();
            long idCollisions = 0;
            DataViewRowId id = default(DataViewRowId);

            for (; ; )
            {
                bool f1 = curs1.MoveNext();
                bool f2 = curs2.MoveNext();
                if (f1 != f2)
                {
                    if (f1)
                        Fail("Left has more rows at position: {0}", curs1.Position);
                    else
                        Fail("Right has more rows at position: {0}", curs2.Position);
                    return Failed();
                }

                if (!f1)
                {
                    if (idCollisions > 0)
                        Fail("{0} id collisions among {1} items", idCollisions, Utils.Size(idsSeen) + idCollisions);
                    return idCollisions == 0;
                }
                else if (checkIdCollisions)
                {
                    idGetter(ref id);
                    if (!Utils.Add(ref idsSeen, id))
                    {
                        if (idCollisions == 0)
                            idCollisions++;
                    }
                }

                Contracts.Assert(curs1.Position == curs2.Position);

                for (int col = 0; col < colLim; col++)
                {
                    var comp = comps[col];
                    if (comp != null && !comp())
                    {
                        Fail("Different values in column {0} of row {1}", col, curs1.Position);
                        return Failed();
                    }
                    if (idComp != null && !idComp())
                    {
                        Fail("Different values in ID of row {0}", curs1.Position);
                        return Failed();
                    }
                }
            }
        }

        protected bool CheckSameValues(DataViewRowCursor curs1, IDataView view2, bool exactTypes = true, bool exactDoubles = true, bool checkId = true)
        {
            Contracts.Assert(curs1.Schema.Count == view2.Schema.Count);

            // Get a cursor for each column.
            int colLim = curs1.Schema.Count;
            var cursors = new DataViewRowCursor[colLim];
            try
            {
                for (int col = 0; col < colLim; col++)
                {
                    // curs1 should have all columns active (for simplicity of the code here).
                    Contracts.Assert(curs1.IsColumnActive(curs1.Schema[col]));
                    cursors[col] = view2.GetRowCursor(view2.Schema[col]);
                }

                // Get the comparison delegates for each column.
                Func<bool>[] comps = new Func<bool>[colLim];
                // We have also one ID comparison delegate for each cursor.
                Func<bool>[] idComps = new Func<bool>[cursors.Length];
                for (int col = 0; col < colLim; col++)
                {
                    Contracts.Assert(cursors[col] != null);
                    var type1 = curs1.Schema[col].Type;
                    var type2 = cursors[col].Schema[col].Type;
                    if (!TestCommon.EqualTypes(type1, type2, exactTypes))
                    {
                        Fail("Different types");
                        return Failed();
                    }
                    comps[col] = GetColumnComparer(curs1, cursors[col], col, type1, exactDoubles);
                    ValueGetter<DataViewRowId> idGetter;
                    idComps[col] = checkId ? GetIdComparer(curs1, cursors[col], out idGetter) : null;
                }

                for (; ; )
                {
                    bool f1 = curs1.MoveNext();
                    for (int col = 0; col < colLim; col++)
                    {
                        bool f2 = cursors[col].MoveNext();
                        if (f1 != f2)
                        {
                            if (f1)
                                Fail("Left has more rows at position: {0}", curs1.Position);
                            else
                                Fail("Right {0} has more rows at position: {1}", col, cursors[2].Position);
                            return Failed();
                        }
                    }

                    if (!f1)
                        return true;

                    for (int col = 0; col < colLim; col++)
                    {
                        Contracts.Assert(curs1.Position == cursors[col].Position);
                        var comp = comps[col];
                        if (comp != null && !comp())
                        {
                            Fail("Different values in column {0} of row {1}", col, curs1.Position);
                            return Failed();
                        }
                        comp = idComps[col];
                        if (comp != null && !comp())
                        {
                            Fail("Different values in ID values for column {0} cursor of row {1}", col, curs1.Position);
                            return Failed();
                        }
                    }
                }
            }
            finally
            {
                for (int col = 0; col < colLim; col++)
                {
                    var c = cursors[col];
                    if (c != null)
                        c.Dispose();
                }
            }
        }
    }
}
