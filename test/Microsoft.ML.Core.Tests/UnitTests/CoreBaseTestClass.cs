// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.RunTests;
using System;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.Core.Tests.UnitTests
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

        protected bool EqualTypes(ColumnType type1, ColumnType type2, bool exactTypes)
        {
            Contracts.AssertValue(type1);
            Contracts.AssertValue(type2);

            return exactTypes ? type1.Equals(type2) : type1.SameSizeAndItemType(type2);
        }

        protected Func<bool> GetIdComparer(IRow r1, IRow r2, out ValueGetter<UInt128> idGetter)
        {
            var g1 = r1.GetIdGetter();
            idGetter = g1;
            var g2 = r2.GetIdGetter();
            UInt128 v1 = default(UInt128);
            UInt128 v2 = default(UInt128);
            return
                () =>
                {
                    g1(ref v1);
                    g2(ref v2);
                    return v1.Equals(v2);
                };
        }

        protected Func<bool> GetComparerOne<T>(IRow r1, IRow r2, int col, Func<T, T, bool> fn)
        {
            var g1 = r1.GetGetter<T>(col);
            var g2 = r2.GetGetter<T>(col);
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
        protected Func<bool> GetComparerVec<T>(IRow r1, IRow r2, int col, int size, Func<T, T, bool> fn)
        {
            var g1 = r1.GetGetter<VBuffer<T>>(col);
            var g2 = r2.GetGetter<VBuffer<T>>(col);
            var v1 = default(VBuffer<T>);
            var v2 = default(VBuffer<T>);
            return
                () =>
                {
                    g1(ref v1);
                    g2(ref v2);
                    return CompareVec<T>(in v1, in v2, size, fn);
                };
        }

        protected bool CompareVec<T>(in VBuffer<T> v1, in VBuffer<T> v2, int size, Func<T, T, bool> fn)
        {
            return CompareVec(in v1, in v2, size, (i, x, y) => fn(x, y));
        }

        protected bool CompareVec<T>(in VBuffer<T> v1, in VBuffer<T> v2, int size, Func<int, T, T, bool> fn)
        {
            Contracts.Assert(size == 0 || v1.Length == size);
            Contracts.Assert(size == 0 || v2.Length == size);
            Contracts.Assert(v1.Length == v2.Length);

            if (v1.IsDense && v2.IsDense)
            {
                for (int i = 0; i < v1.Length; i++)
                {
                    var x1 = v1.Values[i];
                    var x2 = v2.Values[i];
                    if (!fn(i, x1, x2))
                        return false;
                }
                return true;
            }

            Contracts.Assert(!v1.IsDense || !v2.IsDense);
            int iiv1 = 0;
            int iiv2 = 0;
            for (; ; )
            {
                int iv1 = v1.IsDense ? iiv1 : iiv1 < v1.Count ? v1.Indices[iiv1] : v1.Length;
                int iv2 = v2.IsDense ? iiv2 : iiv2 < v2.Count ? v2.Indices[iiv2] : v2.Length;
                T x1, x2;
                int iv;
                if (iv1 == iv2)
                {
                    if (iv1 == v1.Length)
                        return true;
                    x1 = v1.Values[iiv1];
                    x2 = v2.Values[iiv2];
                    iv = iv1;
                    iiv1++;
                    iiv2++;
                }
                else if (iv1 < iv2)
                {
                    x1 = v1.Values[iiv1];
                    x2 = default(T);
                    iv = iv1;
                    iiv1++;
                }
                else
                {
                    x1 = default(T);
                    x2 = v2.Values[iiv2];
                    iv = iv2;
                    iiv2++;
                }
                if (!fn(iv, x1, x2))
                    return false;
            }
        }
        protected Func<bool> GetColumnComparer(IRow r1, IRow r2, int col, ColumnType type, bool exactDoubles)
        {
            if (!type.IsVector)
            {
                switch (type.RawKind)
                {
                    case DataKind.I1:
                        return GetComparerOne<sbyte>(r1, r2, col, (x, y) => x == y);
                    case DataKind.U1:
                        return GetComparerOne<byte>(r1, r2, col, (x, y) => x == y);
                    case DataKind.I2:
                        return GetComparerOne<short>(r1, r2, col, (x, y) => x == y);
                    case DataKind.U2:
                        return GetComparerOne<ushort>(r1, r2, col, (x, y) => x == y);
                    case DataKind.I4:
                        return GetComparerOne<int>(r1, r2, col, (x, y) => x == y);
                    case DataKind.U4:
                        return GetComparerOne<uint>(r1, r2, col, (x, y) => x == y);
                    case DataKind.I8:
                        return GetComparerOne<long>(r1, r2, col, (x, y) => x == y);
                    case DataKind.U8:
                        return GetComparerOne<ulong>(r1, r2, col, (x, y) => x == y);
                    case DataKind.R4:
                        return GetComparerOne<Single>(r1, r2, col, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                    case DataKind.R8:
                        if (exactDoubles)
                            return GetComparerOne<Double>(r1, r2, col, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                        else
                            return GetComparerOne<Double>(r1, r2, col, EqualWithEps);
                    case DataKind.Text:
                        return GetComparerOne<ReadOnlyMemory<char>>(r1, r2, col, (a, b) => a.Span.SequenceEqual(b.Span));
                    case DataKind.Bool:
                        return GetComparerOne<bool>(r1, r2, col, (x, y) => x == y);
                    case DataKind.TimeSpan:
                        return GetComparerOne<TimeSpan>(r1, r2, col, (x, y) => x.Ticks == y.Ticks);
                    case DataKind.DT:
                        return GetComparerOne<DateTime>(r1, r2, col, (x, y) => x.Ticks == y.Ticks);
                    case DataKind.DZ:
                        return GetComparerOne<DateTimeOffset>(r1, r2, col, (x, y) => x.Equals(y));
                    case DataKind.UG:
                        return GetComparerOne<UInt128>(r1, r2, col, (x, y) => x.Equals(y));
                }
            }
            else
            {
                int size = type.VectorSize;
                Contracts.Assert(size >= 0);
                switch (type.ItemType.RawKind)
                {
                    case DataKind.I1:
                        return GetComparerVec<sbyte>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.U1:
                        return GetComparerVec<byte>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.I2:
                        return GetComparerVec<short>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.U2:
                        return GetComparerVec<ushort>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.I4:
                        return GetComparerVec<int>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.U4:
                        return GetComparerVec<uint>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.I8:
                        return GetComparerVec<long>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.U8:
                        return GetComparerVec<ulong>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.R4:
                        return GetComparerVec<Single>(r1, r2, col, size, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                    case DataKind.R8:
                        if (exactDoubles)
                            return GetComparerVec<Double>(r1, r2, col, size, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                        else
                            return GetComparerVec<Double>(r1, r2, col, size, EqualWithEps);
                    case DataKind.Text:
                        return GetComparerVec<ReadOnlyMemory<char>>(r1, r2, col, size, (a,b) => a.Span.SequenceEqual(b.Span));
                    case DataKind.Bool:
                        return GetComparerVec<bool>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.TimeSpan:
                        return GetComparerVec<TimeSpan>(r1, r2, col, size, (x, y) => x.Ticks == y.Ticks);
                    case DataKind.DT:
                        return GetComparerVec<DateTime>(r1, r2, col, size, (x, y) => x.Ticks == y.Ticks);
                    case DataKind.DZ:
                        return GetComparerVec<DateTimeOffset>(r1, r2, col, size, (x, y) => x.Equals(y));
                    case DataKind.UG:
                        return GetComparerVec<UInt128>(r1, r2, col, size, (x, y) => x.Equals(y));
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
            Contracts.Assert(view1.Schema.ColumnCount == view2.Schema.ColumnCount);

            bool all = true;
            bool tmp;

            using (var curs1 = view1.GetRowCursor(col => true))
            using (var curs2 = view2.GetRowCursor(col => true))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, true);
            }
            Check(tmp, "All same failed");
            all &= tmp;

            using (var curs1 = view1.GetRowCursor(col => true))
            using (var curs2 = view2.GetRowCursor(col => (col & 1) == 0, null))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, false);
            }
            Check(tmp, "Even same failed");
            all &= tmp;

            using (var curs1 = view1.GetRowCursor(col => true))
            using (var curs2 = view2.GetRowCursor(col => (col & 1) != 0, null))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, false);
            }
            Check(tmp, "Odd same failed");

            using (var curs1 = view1.GetRowCursor(col => true))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                tmp = CheckSameValues(curs1, view2, exactTypes, exactDoubles, checkId);
            }
            Check(tmp, "Single value same failed");

            all &= tmp;
            return all;
        }

        protected bool CheckSameValues(IRowCursor curs1, IRowCursor curs2, bool exactTypes, bool exactDoubles, bool checkId, bool checkIdCollisions = true)
        {
            Contracts.Assert(curs1.Schema.ColumnCount == curs2.Schema.ColumnCount);

            // Get the comparison delegates for each column.
            int colLim = curs1.Schema.ColumnCount;
            Func<bool>[] comps = new Func<bool>[colLim];
            for (int col = 0; col < colLim; col++)
            {
                var f1 = curs1.IsColumnActive(col);
                var f2 = curs2.IsColumnActive(col);

                if (f1 && f2)
                {
                    var type1 = curs1.Schema.GetColumnType(col);
                    var type2 = curs2.Schema.GetColumnType(col);
                    if (!EqualTypes(type1, type2, exactTypes))
                    {
                        Fail("Different types");
                        return Failed();
                    }
                    comps[col] = GetColumnComparer(curs1, curs2, col, type1, exactDoubles);
                }
            }
            ValueGetter<UInt128> idGetter = null;
            Func<bool> idComp = checkId ? GetIdComparer(curs1, curs2, out idGetter) : null;
            HashSet<UInt128> idsSeen = null;
            if (checkIdCollisions && idGetter == null)
                idGetter = curs1.GetIdGetter();
            long idCollisions = 0;
            UInt128 id = default(UInt128);

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

        protected bool CheckSameValues(IRowCursor curs1, IDataView view2, bool exactTypes = true, bool exactDoubles = true, bool checkId = true)
        {
            Contracts.Assert(curs1.Schema.ColumnCount == view2.Schema.ColumnCount);

            // Get a cursor for each column.
            int colLim = curs1.Schema.ColumnCount;
            var cursors = new IRowCursor[colLim];
            try
            {
                for (int col = 0; col < colLim; col++)
                {
                    // curs1 should have all columns active (for simplicity of the code here).
                    Contracts.Assert(curs1.IsColumnActive(col));
                    cursors[col] = view2.GetRowCursor(c => c == col);
                }

                // Get the comparison delegates for each column.
                Func<bool>[] comps = new Func<bool>[colLim];
                // We have also one ID comparison delegate for each cursor.
                Func<bool>[] idComps = new Func<bool>[cursors.Length];
                for (int col = 0; col < colLim; col++)
                {
                    Contracts.Assert(cursors[col] != null);
                    var type1 = curs1.Schema.GetColumnType(col);
                    var type2 = cursors[col].Schema.GetColumnType(col);
                    if (!EqualTypes(type1, type2, exactTypes))
                    {
                        Fail("Different types");
                        return Failed();
                    }
                    comps[col] = GetColumnComparer(curs1, cursors[col], col, type1, exactDoubles);
                    ValueGetter<UInt128> idGetter;
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
