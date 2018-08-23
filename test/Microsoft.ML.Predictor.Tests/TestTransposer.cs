// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed class TestTransposer : TestDataPipeBase
    {
        public TestTransposer(ITestOutputHelper helper) : base(helper)
        {
        }

        private static T[] NaiveTranspose<T>(IDataView view, int col)
        {
            var type = view.Schema.GetColumnType(col);
            int rc = checked((int)DataViewUtils.ComputeRowCount(view));
            Assert.True(type.ItemType.RawType == typeof(T));
            Assert.True(type.ValueCount > 0);
            T[] retval = new T[rc * type.ValueCount];

            using (var cursor = view.GetRowCursor(c => c == col))
            {
                if (type.IsVector)
                {
                    var getter = cursor.GetGetter<VBuffer<T>>(col);
                    VBuffer<T> temp = default(VBuffer<T>);
                    int offset = 0;
                    while (cursor.MoveNext())
                    {
                        Assert.True(0 <= offset && offset < rc && offset == cursor.Position);
                        getter(ref temp);
                        for (int i = 0; i < temp.Count; ++i)
                            retval[(temp.IsDense ? i : temp.Indices[i]) * rc + offset] = temp.Values[i];
                        offset++;
                    }
                }
                else
                {
                    var getter = cursor.GetGetter<T>(col);
                    while (cursor.MoveNext())
                    {
                        Assert.True(0 <= cursor.Position && cursor.Position < rc);
                        getter(ref retval[(int)cursor.Position]);
                    }
                }
            }
            return retval;
        }

        private static void TransposeCheckHelper<T>(IDataView view, int viewCol, ITransposeDataView trans)
        {
            int col = viewCol;
            var type = trans.TransposeSchema.GetSlotType(col);
            var colType = trans.Schema.GetColumnType(col);
            Assert.Equal(view.Schema.GetColumnName(viewCol), trans.Schema.GetColumnName(col));
            var expectedType = view.Schema.GetColumnType(viewCol);
            // Unfortunately can't use equals because column type equality is a simple reference comparison. :P
            Assert.Equal(expectedType, colType);
            Assert.Equal(DataViewUtils.ComputeRowCount(view), (long)type.VectorSize);
            string desc = string.Format("Column {0} named '{1}'", col, trans.Schema.GetColumnName(col));
            Assert.True(typeof(T) == type.ItemType.RawType, $"{desc} had wrong type for slot cursor");
            Assert.True(type.IsVector, $"{desc} expected to be vector but is not");
            Assert.True(type.VectorSize > 0, $"{desc} expected to be known sized vector but is not");
            Assert.True(0 != colType.ValueCount, $"{desc} expected to have fixed size, but does not");
            int rc = type.VectorSize;
            T[] expectedVals = NaiveTranspose<T>(view, viewCol);
            T[] vals = new T[rc * colType.ValueCount];
            Contracts.Assert(vals.Length == expectedVals.Length);
            using (var cursor = trans.GetSlotCursor(col))
            {
                var getter = cursor.GetGetter<T>();
                VBuffer<T> temp = default(VBuffer<T>);
                int offset = 0;
                while (cursor.MoveNext())
                {
                    Assert.True(offset < vals.Length, $"{desc} slot cursor went further than it should have");
                    getter(ref temp);
                    Assert.True(rc == temp.Length, $"{desc} slot cursor yielded vector with unexpected length");
                    temp.CopyTo(vals, offset);
                    offset += rc;
                }
                Assert.True(colType.ValueCount == offset / rc, $"{desc} slot cursor yielded fewer than expected values");
            }
            for (int i = 0; i < vals.Length; ++i)
                Assert.Equal(expectedVals[i], vals[i]);
        }

        private static VBuffer<T>[] GenerateHelper<T>(
            int rowCount, Double density, Random rgen, Func<T> generator, int slotCount, params int[] forceDenseSlot)
        {
            HashSet<int> forceDenseSlotSet = new HashSet<int>(forceDenseSlot);
            VBuffer<T>[] vecs = new VBuffer<T>[rowCount];
            for (int r = 0; r < vecs.Length; ++r)
            {
                // Density controls both the prevelence of dense arrays, as well as the sparsity of the sparse arrays.
                if (rgen.NextDouble() < density)
                {
                    // Must be dense.
                    T[] vals = new T[slotCount];
                    for (int i = 0; i < vals.Length; ++i)
                        vals[i] = generator();
                    vecs[r] = new VBuffer<T>(slotCount, vals);
                }
                else
                {
                    // Must be sparse.
                    List<int> indices = new List<int>();
                    for (int i = 0; i < slotCount; ++i)
                    {
                        if (forceDenseSlotSet.Contains(i) || rgen.NextDouble() < density)
                            indices.Add(i);
                    }
                    T[] vals = new T[indices.Count];
                    for (int i = 0; i < vals.Length; ++i)
                        vals[i] = generator();
                    vecs[r] = new VBuffer<T>(slotCount, indices.Count, vals, indices.ToArray());
                }
            }
            return vecs;
        }

        private static T[] GenerateHelper<T>(int rowCount, Double density, Random rgen, Func<T> generator)
        {
            T[] values = new T[rowCount];
            for (int r = 0; r < values.Length; ++r)
            {
                if (rgen.NextDouble() < density)
                    values[r] = generator();
            }
            return values;
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Transposer")]
        public void TransposerTest()
        {
            const int rowCount = 1000;
            Random rgen = new Random(0);
            ArrayDataViewBuilder builder = new ArrayDataViewBuilder(Env);

            // A is to check the splitting of a sparse-ish column.
            var dataA = GenerateHelper(rowCount, 0.1, rgen, () => (int)rgen.Next(), 50, 5, 10, 15);
            dataA[rowCount / 2] = new VBuffer<int>(50, 0, null, null); // Coverage for the null vbuffer case.
            builder.AddColumn("A", NumberType.I4, dataA);
            // B is to check the splitting of a dense-ish column.
            builder.AddColumn("B", NumberType.R8, GenerateHelper(rowCount, 0.8, rgen, rgen.NextDouble, 50, 0, 25, 49));
            // C is to just have some column we do nothing with.
            builder.AddColumn("C", NumberType.I2, GenerateHelper(rowCount, 0.1, rgen, () => (short)1, 30, 3, 10, 24));
            // D is to check some column we don't have to split because it's sufficiently small.
            builder.AddColumn("D", NumberType.R8, GenerateHelper(rowCount, 0.1, rgen, rgen.NextDouble, 3, 1));
            // E is to check a sparse scalar column.
            builder.AddColumn("E", NumberType.U4, GenerateHelper(rowCount, 0.1, rgen, () => (uint)rgen.Next(int.MinValue, int.MaxValue)));
            // F is to check a dense-ish scalar column.
            builder.AddColumn("F", NumberType.I4, GenerateHelper(rowCount, 0.8, rgen, () => rgen.Next()));

            IDataView view = builder.GetDataView();

            // Do not force save. This will have a mix of passthrough and saved columns. Note that duplicate
            // specification of "D" to test that specifying a column twice has no ill effects.
            string[] names = { "B", "A", "E", "D", "F", "D" };
            using (Transposer trans = Transposer.Create(Env, view, false, names))
            {
                // Before checking the contents, check the names.
                for (int i = 0; i < names.Length; ++i)
                {
                    int index;
                    Assert.True(trans.Schema.TryGetColumnIndex(names[i], out index), $"Transpose schema couldn't find column '{names[i]}'");
                    int trueIndex;
                    bool result = view.Schema.TryGetColumnIndex(names[i], out trueIndex);
                    Contracts.Assert(result);
                    Assert.True(trueIndex == index, $"Transpose schema had column '{names[i]}' at unexpected index");
                }
                // Check the contents
                Assert.Null(trans.TransposeSchema.GetSlotType(2)); // C check to see that it's not transposable.
                TransposeCheckHelper<int>(view, 0, trans); // A check.
                TransposeCheckHelper<Double>(view, 1, trans); // B check.
                TransposeCheckHelper<Double>(view, 3, trans); // D check.
                TransposeCheckHelper<uint>(view, 4, trans);   // E check.
                TransposeCheckHelper<int>(view, 5, trans); // F check.
            }

            // Force save. Recheck columns that would have previously been passthrough columns.
            // The primary benefit of this check is that we check the binary saving / loading
            // functionality of scalars which are otherwise always must necessarily be
            // passthrough. Also exercise the select by index functionality while we're at it.
            using (Transposer trans = Transposer.Create(Env, view, true, 3, 5, 4))
            {
                // Check to see that A, B, and C were not transposed somehow.
                Assert.Null(trans.TransposeSchema.GetSlotType(0));
                Assert.Null(trans.TransposeSchema.GetSlotType(1));
                Assert.Null(trans.TransposeSchema.GetSlotType(2));
                TransposeCheckHelper<Double>(view, 3, trans); // D check.
                TransposeCheckHelper<uint>(view, 4, trans);   // E check.
                TransposeCheckHelper<int>(view, 5, trans); // F check.
            }
        }

        [Fact]
        [TestCategory("Transposer")]
        public void TransposerSaverLoaderTest()
        {
            const int rowCount = 1000;
            Random rgen = new Random(1);
            ArrayDataViewBuilder builder = new ArrayDataViewBuilder(Env);

            // A is to check the splitting of a sparse-ish column.
            var dataA = GenerateHelper(rowCount, 0.1, rgen, () => (int)rgen.Next(), 50, 5, 10, 15);
            dataA[rowCount / 2] = new VBuffer<int>(50, 0, null, null); // Coverage for the null vbuffer case.
            builder.AddColumn("A", NumberType.I4, dataA);
            // B is to check the splitting of a dense-ish column.
            builder.AddColumn("B", NumberType.R8, GenerateHelper(rowCount, 0.8, rgen, rgen.NextDouble, 50, 0, 25, 49));
            // C is to just have some column we do nothing with.
            builder.AddColumn("C", NumberType.I2, GenerateHelper(rowCount, 0.1, rgen, () => (short)1, 30, 3, 10, 24));
            // D is to check some column we don't have to split because it's sufficiently small.
            builder.AddColumn("D", NumberType.R8, GenerateHelper(rowCount, 0.1, rgen, rgen.NextDouble, 3, 1));
            // E is to check a sparse scalar column.
            builder.AddColumn("E", NumberType.U4, GenerateHelper(rowCount, 0.1, rgen, () => (uint)rgen.Next(int.MinValue, int.MaxValue)));
            // F is to check a dense-ish scalar column.
            builder.AddColumn("F", NumberType.I4, GenerateHelper(rowCount, 0.8, rgen, () => (int)rgen.Next()));

            IDataView view = builder.GetDataView();

            IMultiStreamSource src;
            using (MemoryStream mem = new MemoryStream())
            {
                TransposeSaver saver = new TransposeSaver(Env, new TransposeSaver.Arguments());
                saver.SaveData(mem, view, Utils.GetIdentityPermutation(view.Schema.ColumnCount));
                src = new BytesSource(mem.ToArray());
            }
            TransposeLoader loader = new TransposeLoader(Env, new TransposeLoader.Arguments(), src);
            // First check whether this as an IDataView yields the same values.
            CheckSameValues(view, loader);

            TransposeCheckHelper<int>(view, 0, loader); // A
            TransposeCheckHelper<Double>(view, 1, loader); // B
            TransposeCheckHelper<short>(view, 2, loader); // C
            TransposeCheckHelper<Double>(view, 3, loader); // D
            TransposeCheckHelper<uint>(view, 4, loader); // E
            TransposeCheckHelper<int>(view, 5, loader); // F

            Done();
        }
    }
}
