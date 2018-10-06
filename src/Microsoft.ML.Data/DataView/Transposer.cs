// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This provides a scalable method of getting a "transposed" view of a subset of columns from an
    /// <see cref="IDataView"/>. Instances of <see cref="Transposer"/> act like a wrapped version of
    /// the input dataview, except that an indicated set of columns will be transposable, even if they
    /// were not transposable before. Note that transposition is a somewhat slow and resource intensive
    /// operation.
    /// </summary>
    public sealed class Transposer : ITransposeDataView, IDisposable
    {
        private readonly IHost _host;
        // The input view.
        private readonly IDataView _view;
        // Note that the transposer will still present things as transposed, if the input was a transpose
        // dataview and that thing was listed as transposed.
        private readonly ITransposeDataView _tview;
        private readonly Dictionary<string, int> _nameToICol;
        // The following may be null, if no columns needed to be split.
        private readonly BinaryLoader _splitView;
        public readonly int RowCount;
        // -1 for input columns that were not transposed, a non-negative index into _cols for those that were.
        private readonly int[] _inputToTransposed;
        private readonly ColumnInfo[] _cols;
        private readonly int[] _splitLim;
        private readonly SchemaImpl _tschema;
        private bool _disposed;

        public ITransposeSchema TransposeSchema { get { return _tschema; } }

        /// <summary>
        /// Creates an instance given a list of column names.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="view">The view whose columns we want to transpose</param>
        /// <param name="forceSave">Whether the internal transposer should always unconditionally
        /// save the column we are transposing. Can be useful if the original dataview is possibly
        /// slow to iterate over that column.</param>
        /// <param name="columns">The non-empty list of columns to transpose</param>
        public static Transposer Create(IHostEnvironment env, IDataView view, bool forceSave,
            params string[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Transposer");
            h.CheckValue(view, nameof(view));

            var indices = CheckNamesAndGetIndices(h, view, columns);
            return new Transposer(h, view, forceSave, indices);
        }

        /// <summary>
        /// Creates an instance given a list of column indices.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="view">The view whose columns we want to transpose</param>
        /// <param name="forceSave">Whether the internal transposer should always unconditionally
        /// save the column we are transposing. Can be useful if the original dataview is possibly
        /// slow to iterate over that column.</param>
        /// <param name="columns">The non-empty list of columns to transpose</param>
        public static Transposer Create(IHostEnvironment env, IDataView view, bool forceSave,
            params int[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Transposer");
            host.CheckValue(view, nameof(view));

            var indices = CheckIndices(host, view, columns);
            return new Transposer(host, view, forceSave, indices);
        }

        private Transposer(IHost host, IDataView view, bool forceSave, int[] columns)
        {
            Contracts.AssertValue(host);
            _host = host;
            _host.AssertValue(view);
            _host.CheckParam(Utils.Size(columns) > 0, nameof(columns), "Cannot be empty");

            // REVIEW: Might be a good idea to not have the view as is, but to
            // instead apply choose columns to it first. This could simplify some of
            // the operations.
            _view = view;
            _tview = _view as ITransposeDataView;
            // Remove duplicates and ensure it is sorted.
            IEnumerable<int> columnSet = columns.Distinct().OrderBy(c => c);
            if (_tview != null)
            {
                var ttschema = _tview.TransposeSchema;
                // Keep only those columns for which we do not have a slot view already.
                columnSet = columnSet.Where(c => ttschema.GetSlotType(c) == null);
            }
            columns = columnSet.ToArray();
            _cols = new ColumnInfo[columns.Length];
            var schema = _view.Schema;
            _nameToICol = new Dictionary<string, int>();
            _inputToTransposed = Utils.CreateArray(schema.ColumnCount, -1);
            for (int c = 0; c < columns.Length; ++c)
            {
                _nameToICol[(_cols[c] = ColumnInfo.CreateFromIndex(schema, columns[c])).Name] = c;
                _inputToTransposed[columns[c]] = c;
            }

            using (var ch = _host.Start("Init"))
            {
                var args = new BinarySaver.Arguments();
                // Run deflate at a slightly degraded level, since we anticipate that this is
                // a read-once situation, as opposed to general IDVs which we expect to be run
                // multiple times.
                args.Compression = CompressionKind.Default;
                // Our access into the file will be more or less
                // unstructured and random consistently so keep
                // the block size pretty safe.
                args.MaxBytesPerBlock = 1 << 28;
                args.Silent = true;
                var saver = new BinarySaver(_host, args);

                for (int c = 0; c < _cols.Length; ++c)
                {
                    // REVIEW: Despite not *necessarily* relying on the serialization
                    // for the transposition, I'm still going to insist on serialization,
                    // since it would be strange if the same type failed or not in the
                    // transposer depending on the size. At least as a user, that would
                    // surprise me. Also I expect this to never happen...
                    var type = schema.GetColumnType(_cols[c].Index);
                    if (!saver.IsColumnSavable(type))
                        throw ch.ExceptParam(nameof(view), "Column named '{0}' is not serializable by the transposer", _cols[c].Name);
                    if (type.IsVector && !type.IsKnownSizeVector)
                        throw ch.ExceptParam(nameof(view), "Column named '{0}' is vector, but not of known size, and so cannot be transposed", _cols[c].Name);
                }

                var slicer = new DataViewSlicer(_host, view, columns);
                var slicerSchema = slicer.Schema;
                ch.Assert(Enumerable.Range(0, slicerSchema.ColumnCount).All(c => saver.IsColumnSavable(slicerSchema.GetColumnType(c))));
                _splitLim = new int[_cols.Length];
                List<int> toSave = new List<int>();
                int offset = 0;
                int slicedCount = 0;
                for (int c = 0; c < _cols.Length; ++c)
                {
                    int min;
                    int lim;
                    slicer.InColToOutRange(c, out min, out lim);
                    // It must be a passthrough. We're not going to write it, and will just rely
                    // on the original view to provide the column.
                    ch.Assert(min < lim);
                    int count = lim - min;
                    if (forceSave || count > 1)
                    {
                        toSave.AddRange(Enumerable.Range(min, count));
                        slicedCount++;
                        offset += count;
                    }
                    _splitLim[c] = offset;
                }

                long rowCount;
                ch.Trace("{0} of {1} input columns sliced into {2} columns", slicedCount, _cols.Length, toSave.Count);
                if (toSave.Count > 0)
                {
                    // Only bother to create _splitView if we have to.
                    var stream = new HybridMemoryStream();
                    saver.SaveData(stream, slicer, toSave.ToArray());
                    stream.Seek(0, SeekOrigin.Begin);
                    ch.Trace("Sliced data saved to {0} bytes", stream.Length);
                    var loaderArgs = new BinaryLoader.Arguments();
                    _splitView = new BinaryLoader(_host, loaderArgs, stream, leaveOpen: false);
                    rowCount = DataViewUtils.ComputeRowCount(_splitView);
                }
                else
                    rowCount = DataViewUtils.ComputeRowCount(_view);
                ch.Assert(rowCount >= 0);
                if (rowCount > Utils.ArrayMaxSize)
                    throw _host.ExceptParam(nameof(view), "View has {0} rows, we cannot transpose with more than {1}", rowCount, Utils.ArrayMaxSize);
                RowCount = (int)rowCount;
                _tschema = new SchemaImpl(this);
                ch.Done();
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                if (_splitView != null)
                    _splitView.Dispose();
            }
        }

        private static int[] CheckNamesAndGetIndices(IHost host, IDataView view, string[] columns)
        {
            Contracts.AssertValue(host, "host");
            host.AssertValue(view, "view");
            host.CheckParam(Utils.Size(columns) > 0, nameof(columns), "Cannot be empty");

            var schema = view.Schema;
            int[] indices = new int[columns.Length];
            for (int c = 0; c < columns.Length; ++c)
            {
                if (!schema.TryGetColumnIndex(columns[c], out indices[c]))
                    throw host.ExceptParam(nameof(columns), "Column named '{0}' not found", columns[c]);
            }
            return indices;
        }

        private static int[] CheckIndices(IHost host, IDataView view, int[] columns)
        {
            Contracts.AssertValue(host);
            host.AssertValue(view);

            var schema = view.Schema;
            for (int c = 0; c < columns.Length; ++c)
            {
                if (!(0 <= columns[c] && columns[c] < schema.ColumnCount))
                    throw host.ExceptParam(nameof(columns), "Column index {0} illegal for data with {1} column", columns[c], schema.ColumnCount);
            }
            return columns;
        }

        public ISlotCursor GetSlotCursor(int col)
        {
            _host.CheckParam(0 <= col && col < _tschema.ColumnCount, nameof(col));
            if (_inputToTransposed[col] == -1)
            {
                // Check if the parent view has this slot transposed. If it doesn't, fail.
                if (_tview != null && _tview.TransposeSchema.GetSlotType(col) != null)
                    return _tview.GetSlotCursor(col);
                throw _host.ExceptParam(nameof(col), "Bad call to GetSlotCursor on untransposable column '{0}'",
                    _tschema.GetColumnName(col));
            }
            var type = _tschema.GetSlotType(col).ItemType.RawType;

            var tcol = _inputToTransposed[col];
            _host.Assert(0 <= tcol && tcol < _cols.Length);
            _host.Assert(_cols[tcol].Index == col);

            return Utils.MarshalInvoke(GetSlotCursorCore<int>, type, col);
        }

        private ISlotCursor GetSlotCursorCore<T>(int col)
        {
            if (_tschema.GetColumnType(col).IsVector)
                return new SlotCursorVec<T>(this, col);
            return new SlotCursorOne<T>(this, col);
        }

        #region IDataView implementation stuff, passthrough on to view.
        // It is helpful to have transposed data views actually implement dataview, since
        // we are still and will likely forever remain in a state where only a few specialized
        // operations make use of the transpose dataview, with many operations instead being
        // handled in the standard row-wise fashion.
        public Schema Schema => _view.Schema;

        public bool CanShuffle { get { return _view.CanShuffle; } }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            return _view.GetRowCursor(predicate, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            return _view.GetRowCursorSet(out consolidator, predicate, n, rand);
        }

        public long? GetRowCount(bool lazy = true)
        {
            // Not a passthrough.
            return RowCount;
        }
        #endregion

        private sealed class SchemaImpl : ITransposeSchema
        {
            private readonly Transposer _parent;
            private readonly IExceptionContext _ectx;
            private readonly VectorType[] _slotTypes;

            private Schema InputSchema { get { return _parent._view.Schema; } }

            public Schema AsSchema { get; }

            public int ColumnCount { get { return InputSchema.ColumnCount; } }

            public SchemaImpl(Transposer parent)
            {
                Contracts.AssertValue(parent, "parent");
                Contracts.AssertValue(parent._host, "parent");
                _parent = parent;
                _ectx = _parent._host;

                _slotTypes = new VectorType[_parent._cols.Length];
                for (int c = 0; c < _slotTypes.Length; ++c)
                {
                    ColumnInfo srcInfo = _parent._cols[c];
                    var ctype = srcInfo.Type.ItemType;
                    _ectx.Assert(ctype.IsPrimitive);
                    _slotTypes[c] = new VectorType(ctype.AsPrimitive, _parent.RowCount);
                }

                AsSchema = Data.Schema.Create(this);
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                _ectx.CheckValueOrNull(name);
                return InputSchema.TryGetColumnIndex(name, out col);
            }

            public string GetColumnName(int col)
            {
                return InputSchema.GetColumnName(col);
            }

            public ColumnType GetColumnType(int col)
            {
                return InputSchema.GetColumnType(col);
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                return InputSchema.GetMetadataTypes(col);
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                return InputSchema.GetMetadataTypeOrNull(kind, col);
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                InputSchema.GetMetadata(kind, col, ref value);
            }

            public VectorType GetSlotType(int col)
            {
                _ectx.Check(0 <= col && col < ColumnCount, "col");
                if (_parent._inputToTransposed[col] == -1)
                {
                    if (_parent._tview != null)
                        return _parent._tview.TransposeSchema.GetSlotType(col);
                    return null;
                }
                return _slotTypes[_parent._inputToTransposed[col]];
            }
        }

        private abstract class SlotCursor<T> : RootCursorBase, ISlotCursor
        {
            private readonly Transposer _parent;
            private readonly int _col;
            private ValueGetter<VBuffer<T>> _getter;

            public override long Batch { get { return 0; } }

            protected SlotCursor(Transposer parent, int col)
                : base(parent._host)
            {
                Ch.Assert(0 <= col && col < parent.Schema.ColumnCount);
                _parent = parent;
                _col = col;
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

            public ValueGetter<VBuffer<TValue>> GetGetter<TValue>()
            {
                if (_getter == null)
                    _getter = GetGetterCore();
                ValueGetter<VBuffer<TValue>> getter = _getter as ValueGetter<VBuffer<TValue>>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }

            public VectorType GetSlotType()
            {
                return _parent.TransposeSchema.GetSlotType(_col);
            }

            protected abstract ValueGetter<VBuffer<T>> GetGetterCore();
        }

        private sealed class SlotCursorOne<T> : SlotCursor<T>
        {
            private readonly IDataView _view;
            private readonly int _col;
            private readonly int _len;

            public SlotCursorOne(Transposer parent, int col)
                : base(parent, col)
            {
                Ch.Assert(0 <= col && col < parent.Schema.ColumnCount);
                int iinfo = parent._inputToTransposed[col];
                Ch.Assert(iinfo >= 0);
                int smin = iinfo == 0 ? 0 : parent._splitLim[iinfo - 1];
                if (parent._splitLim[iinfo] == smin)
                {
                    // This is a passthrough column.
                    _view = parent._view;
                    _col = parent._cols[iinfo].Index;
                }
                else
                {
                    _view = parent._splitView;
                    _col = smin;
                    Ch.Assert(parent._splitLim[iinfo] - _col == 1);
                }
                Ch.AssertValue(_view);
                Ch.Assert(_view.Schema.GetColumnType(_col).IsPrimitive);
                Ch.Assert(_view.Schema.GetColumnType(_col).RawType == typeof(T));
                _len = parent.RowCount;
            }

            protected override bool MoveNextCore()
            {
                // We only can move next on one slot, since this is a scalar column.
                return State == CursorState.NotStarted;
            }

            protected override ValueGetter<VBuffer<T>> GetGetterCore()
            {
                var isDefault = Conversion.Conversions.Instance.GetIsDefaultPredicate<T>(_view.Schema.GetColumnType(_col));
                bool valid = false;
                VBuffer<T> cached = default(VBuffer<T>);
                return
                    (ref VBuffer<T> dst) =>
                    {
                        Ch.Check(IsGood, "Cannot get values in the cursor's current state");
                        if (!valid)
                        {
                            using (var cursor = _view.GetRowCursor(c => c == _col))
                            {
                                int[] indices = null;
                                T[] values = null;
                                int len = -1;
                                int count = 0;
                                T value = default(T);
                                ValueGetter<T> getter = cursor.GetGetter<T>(_col);
                                while (cursor.MoveNext())
                                {
                                    len++;
                                    Ch.Assert(len <= _len);
                                    getter(ref value);
                                    if (isDefault(ref value))
                                        continue;
                                    Utils.EnsureSize(ref indices, ++count);
                                    indices[count - 1] = len;
                                    Utils.EnsureSize(ref values, count);
                                    values[count - 1] = value;
                                }
                                len++;
                                Ch.Assert(len == _len);
                                if (count < len / 2 || count == len)
                                    cached = new VBuffer<T>(len, count, values, count == len ? null : indices);
                                else
                                    (new VBuffer<T>(len, count, values, indices)).CopyToDense(ref cached);
                            }
                            valid = true;
                        }
                        cached.CopyTo(ref dst);
                    };
            }
        }

        private sealed class SlotCursorVec<T> : SlotCursor<T>
        {
            // The source data view. Note that this might be either the original input dataview
            // if the column was not sufficiently large to justify "slicing," or the slicer dataview
            // if it was large enough to justify splitting. (So this source data view will not
            // necessarily be the same as the Transposer _view, but it might be.)
            private readonly IDataView _view;
            // In the case when we've sliced a dataview, the slot cursor will need to iterative over
            // multiple cursors to get all slots from the original dataview that transposer is transposing.
            // These fields define this range.
            private readonly int _colMin;
            private readonly int _colLim;
            // The length of the resulting vectors. This is the same as the row count from the original dataview.
            private readonly int _len;

            // Temporary working/storage buffers.
            private readonly VBuffer<T>[] _rbuff; // Working intermediate row-wise buffer.
            private readonly int[] _rbuffIndices; // Working intermediate row-wise indices.
            private int[][] _indices;    // Working intermediate index buffers.
            private T[][] _values;       // Working intermediate value buffers.
            private int[] _counts;       // Working intermediate count buffers.

            // The transposed contents of _colStored.
            private VBuffer<T>[] _cbuff; // Working intermediate column-wise buffer.

            // Variables to track current cursor position.
            private int _colStored;      // The current column of the source data view actually stored in the intermediate buffers.
            private int _colCurr;        // The current column of the split view that our cursor has on its position.
            private int _slotCurr;       // The current slot that our cursor has on its position.
            private int _slotLim;        // The limit of the slot index for the current column, so we know when to move to next columns.

            /// <summary>
            /// Constructs a slot cursor.
            /// </summary>
            /// <param name="parent">The transposer.</param>
            /// <param name="col">The index of the transposed column.</param>
            public SlotCursorVec(Transposer parent, int col)
                : base(parent, col)
            {
                int iinfo = parent._inputToTransposed[col];
                Ch.Assert(iinfo >= 0);
                int smin = iinfo == 0 ? 0 : parent._splitLim[iinfo - 1];
                if (parent._splitLim[iinfo] == smin)
                {
                    // This is a passthrough column.
                    _view = parent._view;
                    _colMin = parent._cols[iinfo].Index;
                    _colLim = _colMin + 1;
                }
                else
                {
                    _view = parent._splitView;
                    _colMin = smin;
                    _colLim = parent._splitLim[iinfo];
                }

                Ch.AssertValue(_view);
                // Make the current state just "before" the first column so
                // we can move cleanly onto the first slot of the first column.
                _colStored = _colCurr = _colMin - 1;
                _slotLim = 0;
                _slotCurr = -1;
                // The transposer will store this many rows from the source data view (either the
                // slicer, or the original dataview if the column was not sufficiently large) column
                // before copying into the _indices/_values/_counts working buffers, during a phase
                // of EnsureValid.
                _rbuff = new VBuffer<T>[16];
                _rbuffIndices = new int[_rbuff.Length];
                _len = parent.RowCount;
            }

            /// <summary>
            /// Ensures that the column from the source data view stored in our intermediate buffers is the
            /// current column requested.
            /// </summary>
            private void EnsureValid()
            {
                Ch.Check(State == CursorState.Good, "Cursor is not in good state, cannot get values");
                Ch.Assert(_slotCurr >= 0);
                if (_colStored == _colCurr)
                    return;

                var type = _view.Schema.GetColumnType(_colCurr);
                Ch.Assert(type.ItemType.RawType == typeof(T));
                Ch.Assert(type.ValueCount > 0);
                RefPredicate<T> isDefault = Conversion.Conversions.Instance.GetIsDefaultPredicate<T>(type.ItemType);
                int vecLen = type.ValueCount;
                int maxPossibleSize = _rbuff.Length * vecLen;
                const int sparseThresholdRatio = 5;
                int sparseThreshold = (maxPossibleSize + sparseThresholdRatio - 1) / sparseThresholdRatio;
                Array.Clear(_rbuffIndices, 0, _rbuffIndices.Length);
                int offset = 0;

                // REVIEW: An obvious enhancement to make to this system is to take everything in the
                // below "using" and make it part of some sort of external task, which this method waits on
                // instead of actually doing the computation itself. The benefit there is that the next column
                // is having its values loaded into _indices/_values/_counts while the current column is being
                // served up to the consumer through _cbuff.

                using (var cursor = _view.GetRowCursor(c => c == _colCurr))
                {
                    // Make sure that the buffers (and subbuffers) are all of appropriate size.
                    Utils.EnsureSize(ref _indices, vecLen);
                    for (int i = 0; i < type.ValueCount; ++i)
                        _indices[i] = _indices[i] ?? new int[_len];
                    Utils.EnsureSize(ref _values, vecLen);
                    for (int i = 0; i < type.ValueCount; ++i)
                        _values[i] = _values[i] ?? new T[_len];
                    Utils.EnsureSize(ref _counts, vecLen, keepOld: false);
                    if (vecLen > 0)
                        Array.Clear(_counts, 0, vecLen);

                    var getter = cursor.GetGetter<VBuffer<T>>(_colCurr);
                    int irbuff = 0; // Next index into _rbuff. During the copy phase this doubles as the lim.
                    int countSum = 0;
                    // In the key value pair, the first is the slot index, then second is the row index in _rbuff.
                    var heap = new Heap<KeyValuePair<int, int>>((p1, p2) => p1.Key > p2.Key || (p1.Key == p2.Key && p1.Value > p2.Value), _rbuff.Length);

                    Action copyPhase =
                        () =>
                        {
                            if (countSum >= sparseThreshold)
                            {
                                // Slot by slot insertion, involving an exhaustive check over the tile.
                                for (int s = 0; s < vecLen; ++s)
                                {
                                    int[] indices = _indices[s];
                                    T[] values = _values[s];

                                    for (int r = 0; r < irbuff; ++r)
                                    {
                                        int rowNum = offset + r;
                                        var rbuff = _rbuff[r];

                                        if (rbuff.IsDense)
                                        {
                                            // Store it as sparse. We will densify later, if we must.
                                            if (!isDefault(ref rbuff.Values[s]))
                                            {
                                                indices[_counts[s]] = rowNum;
                                                values[_counts[s]++] = rbuff.Values[s];
                                            }
                                        }
                                        else
                                        {
                                            int ii = _rbuffIndices[r];
                                            if (ii < rbuff.Count && rbuff.Indices[ii] == s)
                                            {
                                                if (!isDefault(ref rbuff.Values[ii]))
                                                {
                                                    indices[_counts[s]] = rowNum;
                                                    values[_counts[s]++] = rbuff.Values[ii];
                                                }
                                                _rbuffIndices[r]++;
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                // Slot by slot insertion, involving a structure to determine the row to insert next.
                                Ch.Assert(heap.Count == 0);
                                int s = -1;
                                int[] indices = null;
                                T[] values = null;
                                // Construct the initial heap.
                                for (int r = 0; r < irbuff; ++r)
                                {
                                    var rbuff = _rbuff[r];
                                    if (rbuff.Count > 0)
                                        heap.Add(new KeyValuePair<int, int>(rbuff.IsDense ? 0 : rbuff.Indices[0], r));
                                }
                                while (heap.Count > 0)
                                {
                                    var pair = heap.Pop(); // Key is the slot, pair is the row index.
                                    if (pair.Key != s)
                                    {
                                        Ch.Assert(pair.Key > s);
                                        s = pair.Key;
                                        indices = _indices[s];
                                        values = _values[s];
                                    }
                                    var rbuff = _rbuff[pair.Value];
                                    int ii = rbuff.IsDense ? s : _rbuffIndices[pair.Value]++;
                                    Ch.Assert(rbuff.IsDense || rbuff.Indices[ii] == s);
                                    indices[_counts[s]] = pair.Value + offset;
                                    values[_counts[s]++] = rbuff.Values[ii];
                                    if (++ii < rbuff.Count) // Still more stuff. Add another followup item to the heap.
                                        heap.Add(new KeyValuePair<int, int>(rbuff.IsDense ? s + 1 : rbuff.Indices[ii], pair.Value));
                                }
                            }
                            Array.Clear(_rbuffIndices, 0, irbuff);
                            offset += irbuff;
                            countSum = irbuff = 0;
                        };

                    while (cursor.MoveNext())
                    {
                        int idx = checked((int)cursor.Position);
                        Ch.Assert(0 <= idx && idx < _len);
                        getter(ref _rbuff[irbuff]);
                        countSum += _rbuff[irbuff].Count;
                        if (++irbuff == _rbuff.Length)
                            copyPhase();
                    }
                    if (irbuff > 0)
                        copyPhase();
                    Ch.Assert(offset == _len);
                }

                // REVIEW: Everything *above* could be factored into async code, but the below absolutely must
                // occur as an exclusive section.

                // Finalize the contents of _cbuff based on _counts/_values/_indices.
                Utils.EnsureSize(ref _cbuff, vecLen);
                for (int s = 0; s < vecLen; ++s)
                {
                    var temp = new VBuffer<T>(_len, _counts[s], _values[s], _indices[s]);
                    if (temp.Count < _len / 2)
                    {
                        // Already sparse enough, I guess. Swap out the arrays.
                        Utils.Swap(ref temp, ref _cbuff[s]);
                        _indices[s] = temp.Indices ?? new int[_len];
                        _values[s] = temp.Values ?? new T[_len];
                        Ch.Assert(_indices[s].Length == _len);
                        Ch.Assert(_values[s].Length == _len);
                    }
                    else
                    {
                        // Not dense enough. Densify temp into _cbuff[s]. Don't swap the arrays.
                        temp.CopyToDense(ref _cbuff[s]);
                    }
                }
                _colStored = _colCurr;
            }

            protected override bool MoveNextCore()
            {
                if (++_slotCurr < _slotLim)
                    return true;
                Ch.Assert(_slotCurr == _slotLim);
                _slotCurr = 0;
                if (++_colCurr == _colLim)
                    return false;
                _slotLim = _view.Schema.GetColumnType(_colCurr).ValueCount;
                Ch.Assert(_slotLim > 0);
                return true;
            }

            private void Getter(ref VBuffer<T> dst)
            {
                Ch.Check(IsGood, "Cannot get values in the cursor's current state");
                EnsureValid();
                Ch.Assert(0 <= _slotCurr && _slotCurr < Utils.Size(_cbuff) && _cbuff[_slotCurr].Length == _len);
                _cbuff[_slotCurr].CopyTo(ref dst);
            }

            protected override ValueGetter<VBuffer<T>> GetGetterCore()
            {
                return Getter;
            }
        }

        /// <summary>
        /// This takes an input data view, and presents a dataset with "sliced" up columns
        /// that are partitionings of the original columns. Scalar columns and sufficiently
        /// small vector columns are just served up as themselves. The idea is that each of
        /// those slices should be small enough that storing an entire column in memory.
        /// </summary>
        private sealed class DataViewSlicer : IDataView
        {
            // REVIEW: Could this be useful in its own right as a transform? We will
            // have to have something that selects out a subset of columns, somehow, someday.

            private readonly IDataView _input;
            // For each input column, the structure handling the mapping of that column
            // into multiple split output columns.
            private readonly Splitter[] _splitters;
            // For each input column, indicate what the limit of the output columns is.
            private readonly int[] _incolToLim;
            // Each of our output columns maps to a splitter. Multiple columns can
            // map to the same splitter, that being kind of the point of a splitter.
            private readonly int[] _colToSplitIndex;
            // For each output column, indicates what output column it's surfacing
            // from the splitter.
            private readonly int[] _colToSplitCol;
            private readonly SchemaImpl _schema;

            private readonly IHost _host;
            public Schema Schema => _schema.AsSchema;

            public bool CanShuffle { get { return _input.CanShuffle; } }

            public DataViewSlicer(IHost host, IDataView input, int[] toSlice)
            {
                Contracts.AssertValue(host, "host");
                _host = host;

                _host.AssertValue(input);
                _host.AssertValue(toSlice);

                _input = input;
                _splitters = new Splitter[toSlice.Length];
                _incolToLim = new int[toSlice.Length];
                int outputColumnCount = 0;
                // Also build our schema's name to index here. The slicers just surface the original
                // input name as the name to all of our input columns.
                var nameToCol = new Dictionary<string, int>();
                for (int c = 0; c < toSlice.Length; ++c)
                {
                    var splitter = _splitters[c] = Splitter.Create(_input, toSlice[c]);
                    _host.Assert(splitter.ColumnCount >= 1);
                    _incolToLim[c] = outputColumnCount += splitter.ColumnCount;
                    nameToCol[_input.Schema.GetColumnName(toSlice[c])] = outputColumnCount - 1;
                }
                _colToSplitIndex = new int[outputColumnCount];
                _colToSplitCol = new int[outputColumnCount];
                outputColumnCount = 0;
                for (int c = 0; c < _splitters.Length; ++c)
                {
                    int outCount = _splitters[c].ColumnCount;
                    for (int i = 0; i < outCount; ++i)
                    {
                        _colToSplitIndex[outputColumnCount] = c;
                        _colToSplitCol[outputColumnCount++] = i;
                    }
                }
                _host.Assert(outputColumnCount == _colToSplitIndex.Length);
                _schema = new SchemaImpl(this, nameToCol);
            }

            public long? GetRowCount(bool lazy = true)
            {
                return _input.GetRowCount(lazy);
            }

            /// <summary>
            /// Given the index of a column we were told to split, get the corresponding range out output
            /// ranges.
            /// </summary>
            /// <param name="incol">The index into the array of column indices.</param>
            /// <param name="outMin">The minimum output column index corresponding to that split column</param>
            /// <param name="outLim">The exclusive limit of the output column index corresponding to that
            /// split column</param>
            public void InColToOutRange(int incol, out int outMin, out int outLim)
            {
                _host.Assert(0 <= incol && incol < _incolToLim.Length);
                outMin = incol == 0 ? 0 : _incolToLim[incol - 1];
                outLim = _incolToLim[incol];
            }

            private void OutputColumnToSplitterIndices(int col, out int splitInd, out int splitCol)
            {
                _host.Assert(0 <= col && col < _colToSplitIndex.Length);
                splitInd = _colToSplitIndex[col];
                splitCol = _colToSplitCol[col];
            }

            public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                _host.CheckValue(predicate, nameof(predicate));
                bool[] activeSplitters;
                var srcPred = CreateInputPredicate(predicate, out activeSplitters);
                return new Cursor(_host, this, _input.GetRowCursor(srcPred, rand), predicate, activeSplitters);
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
            {
                _host.CheckValue(predicate, nameof(predicate));
                _host.CheckValueOrNull(rand);
                bool[] activeSplitters;
                var srcPred = CreateInputPredicate(predicate, out activeSplitters);
                var result = _input.GetRowCursorSet(out consolidator, srcPred, n, rand);
                for (int i = 0; i < result.Length; ++i)
                    result[i] = new Cursor(_host, this, result[i], predicate, activeSplitters);
                return result;
            }

            /// <summary>
            /// Given a possibly null predicate for this data view, produce the dependency predicate for the sources,
            /// as well as a list of all the splitters for which we should produce rowsets.
            /// </summary>
            /// <param name="pred">The predicate input into the <see cref="GetRowCursor(Func{int, bool}, IRandom)"/> method.</param>
            /// <param name="activeSplitters">A boolean indicator array of length equal to the number of splitters,
            /// indicating whether that splitter has any active columns in its outputs or not</param>
            /// <returns>The predicate to use when constructing the row cursor from the source</returns>
            private Func<int, bool> CreateInputPredicate(Func<int, bool> pred, out bool[] activeSplitters)
            {
                _host.AssertValueOrNull(pred);

                activeSplitters = new bool[_splitters.Length];
                var activeSrcSet = new HashSet<int>();
                int offset = 0;
                for (int i = 0; i < activeSplitters.Length; ++i)
                {
                    var splitter = _splitters[i];
                    // Don't activate input source columns if none of the resulting columns were selected.
                    bool isActive = pred == null || Enumerable.Range(offset, splitter.ColumnCount).Any(c => pred(c));
                    if (isActive)
                    {
                        activeSplitters[i] = isActive;
                        activeSrcSet.Add(splitter.SrcCol);
                    }
                    offset += splitter.ColumnCount;
                }
                return activeSrcSet.Contains;
            }

            /// <summary>
            /// This collates the schemas of all the columns from the <see cref="Splitter"/> instances.
            /// </summary>
            private sealed class SchemaImpl : NoMetadataSchema
            {
                private readonly DataViewSlicer _slicer;
                private readonly Dictionary<string, int> _nameToCol;

                public Schema AsSchema { get; }

                public override int ColumnCount { get { return _slicer._colToSplitIndex.Length; } }

                public SchemaImpl(DataViewSlicer slicer, Dictionary<string, int> nameToCol)
                {
                    Contracts.AssertValue(slicer);
                    Contracts.AssertValue(nameToCol);
                    _slicer = slicer;
                    _nameToCol = nameToCol;
                    AsSchema = Data.Schema.Create(this);
                }

                public override bool TryGetColumnIndex(string name, out int col)
                {
                    Contracts.CheckValueOrNull(name);
                    return Utils.TryGetValue(_nameToCol, name, out col);
                }

                public override string GetColumnName(int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    int splitInd;
                    int splitCol;
                    _slicer.OutputColumnToSplitterIndices(col, out splitInd, out splitCol);
                    return _slicer._splitters[splitInd].GetColumnName(splitCol);
                }

                public override ColumnType GetColumnType(int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    int splitInd;
                    int splitCol;
                    _slicer.OutputColumnToSplitterIndices(col, out splitInd, out splitCol);
                    return _slicer._splitters[splitInd].GetColumnType(splitCol);
                }
            }

            /// <summary>
            /// Very simple schema base that surfaces no metadata, since I have a couple schema
            /// implementations neither of which I care about surfacing metadata.
            /// </summary>
            private abstract class NoMetadataSchema : ISchema
            {
                public abstract int ColumnCount { get; }

                public abstract bool TryGetColumnIndex(string name, out int col);

                public abstract string GetColumnName(int col);

                public abstract ColumnType GetColumnType(int col);

                public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    return Enumerable.Empty<KeyValuePair<string, ColumnType>>();
                }

                public ColumnType GetMetadataTypeOrNull(string kind, int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    return null;
                }

                public void GetMetadata<TValue>(string kind, int col, ref TValue value)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    throw MetadataUtils.ExceptGetMetadata();
                }
            }

            /// <summary>
            /// There is one instance of these per column, implementing the possible splitting
            /// of one column from a <see cref="IDataView"/> into multiple columns. The instance
            /// describes the resulting split columns through its implementation of
            /// <see cref="ISchema"/>, and then can be bound to an <see cref="IRow"/> to provide
            /// that splitting functionality.
            /// </summary>
            private abstract class Splitter : NoMetadataSchema
            {
                private readonly IDataView _view;
                private readonly int _col;

                public Schema AsSchema { get; }

                public int SrcCol { get { return _col; } }

                protected Splitter(IDataView view, int col)
                {
                    Contracts.AssertValue(view);
                    Contracts.Assert(0 <= col && col < view.Schema.ColumnCount);
                    _view = view;
                    _col = col;
                    AsSchema = Data.Schema.Create(this);
                }

                /// <summary>
                /// Creates a splitter for a given row.
                /// </summary>
                public static Splitter Create(IDataView view, int col)
                {
                    var type = view.Schema.GetColumnType(col);
                    Contracts.Assert(type.IsPrimitive || type.VectorSize > 0);
                    const int defaultSplitThreshold = 16;
                    if (type.VectorSize <= defaultSplitThreshold)
                        return Utils.MarshalInvoke(CreateCore<int>, type.RawType, view, col);
                    else
                    {
                        // There are serious practical problems with trying to save many thousands of columns.
                        // We balance this by setting a hard limit on the number of splits per column we will
                        // generate.
                        const int maxSplitInto = 256;
                        int splitInto = (type.VectorSize - 1) / defaultSplitThreshold + 1;
                        int[] ends;
                        if (splitInto <= maxSplitInto)
                        {
                            ends = new int[splitInto];
                            for (int i = 0; i < ends.Length; ++i)
                                ends[i] = (i + 1) * defaultSplitThreshold;
                        }
                        else
                        {
                            ends = new int[maxSplitInto];
                            for (int i = 0; i < ends.Length; ++i)
                                ends[i] = (int)((long)(i + 1) * type.VectorSize / maxSplitInto);
                        }
                        ends[ends.Length - 1] = type.VectorSize;
                        // We have a min of 1 here, because if the first min was 0 then
                        // the first split would cover no slots.
                        Contracts.Assert(Utils.IsIncreasing(1, ends, type.VectorSize + 1));
                        return Utils.MarshalInvoke(CreateCore<int>, type.ItemType.RawType, view, col, ends);
                    }
                }

                /// <summary>
                /// Given an input <see cref="IRow"/>, create the <see cref="IRow"/> containing the split
                /// version of the columns.
                /// </summary>
                public abstract IRow Bind(IRow row, Func<int, bool> pred);

                private static Splitter CreateCore<T>(IDataView view, int col)
                {
                    return new NoSplitter<T>(view, col);
                }

                private static Splitter CreateCore<T>(IDataView view, int col, int[] ends)
                {
                    return new ColumnSplitter<T>(view, col, ends);
                }

                #region ISchema implementation
                // Subclasses should implement ColumnCount and GetColumnType.
                public override bool TryGetColumnIndex(string name, out int col)
                {
                    Contracts.CheckNonEmpty(name, nameof(name));
                    if (name != _view.Schema.GetColumnName(SrcCol))
                    {
                        col = default(int);
                        return false;
                    }
                    // We're just pretending all the columns have the same name, so we
                    // just return the last column's index if it happens to match.
                    col = ColumnCount - 1;
                    return true;
                }

                public override string GetColumnName(int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    return _view.Schema.GetColumnName(SrcCol);
                }
                #endregion

                private abstract class RowBase<TSplitter> : IRow
                    where TSplitter : Splitter
                {
                    protected readonly TSplitter Parent;
                    protected readonly IRow Input;

                    public Schema Schema => Parent.AsSchema;
                    public long Position => Input.Position;
                    public long Batch => Input.Batch;

                    public RowBase(TSplitter parent, IRow input)
                    {
                        Contracts.AssertValue(parent);
                        Contracts.AssertValue(input);
                        Contracts.Assert(input.IsColumnActive(parent.SrcCol));
                        Parent = parent;
                        Input = input;
                    }

                    public ValueGetter<UInt128> GetIdGetter()
                    {
                        return Input.GetIdGetter();
                    }

                    public abstract bool IsColumnActive(int col);

                    public abstract ValueGetter<TValue> GetGetter<TValue>(int col);
                }

                /// <summary>
                /// A splitter that doesn't split, just passes through the column contents.
                /// Useful for when we've been told to "split" a column that we don't need
                /// to split.
                /// </summary>
                private sealed class NoSplitter<T> : Splitter
                {
                    public override int ColumnCount => 1;

                    public NoSplitter(IDataView view, int col)
                        : base(view, col)
                    {
                        Contracts.Assert(_view.Schema.GetColumnType(col).RawType == typeof(T));
                    }

                    public override ColumnType GetColumnType(int col)
                    {
                        Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                        return _view.Schema.GetColumnType(SrcCol);
                    }

                    public override IRow Bind(IRow row, Func<int, bool> pred)
                    {
                        Contracts.AssertValue(row);
                        Contracts.Assert(row.Schema == _view.Schema);
                        Contracts.AssertValue(pred);
                        Contracts.Assert(row.IsColumnActive(SrcCol));
                        return new Row(this, row, pred(0));
                    }

                    private sealed class Row : RowBase<NoSplitter<T>>
                    {
                        private readonly bool _isActive;

                        public Row(NoSplitter<T> parent, IRow input, bool isActive)
                            : base(parent, input)
                        {
                            Contracts.Assert(Parent.ColumnCount == 1);
                            _isActive = isActive;
                        }

                        public override bool IsColumnActive(int col)
                        {
                            Contracts.CheckParam(0 <= col && col < Parent.ColumnCount, nameof(col));
                            return _isActive;
                        }

                        public override ValueGetter<TValue> GetGetter<TValue>(int col)
                        {
                            Contracts.Check(IsColumnActive(col));
                            return Input.GetGetter<TValue>(Parent.SrcCol);
                        }
                    }
                }

                /// <summary>
                /// This splitter enables the partition of a single column into two or more
                /// columns.
                /// </summary>
                private sealed class ColumnSplitter<T> : Splitter
                {
                    private readonly int[] _lims;
                    // Cache of the types of each slice.
                    private readonly VectorType[] _types;

                    public override int ColumnCount { get { return _lims.Length; } }

                    /// <summary>
                    /// Provide a column partitioner that partitions a vector column into multiple
                    /// vector columns.
                    /// </summary>
                    /// <param name="view">The view where we are slicing one column</param>
                    /// <param name="col">The column we are slicing</param>
                    /// <param name="lims">Equal in length to the number of slices, this is
                    /// the limit of the indices of each slice, where the successive slice
                    /// starts with that limit as its minimum index. So slice i comes from
                    /// source slot indices from <c><paramref name="lims"/>[i-1]</c> inclusive to
                    /// <c><paramref name="lims"/>[i]</c> exclusive, with slice 0 starting at 0.</param>
                    public ColumnSplitter(IDataView view, int col, int[] lims)
                        : base(view, col)
                    {
                        var type = _view.Schema.GetColumnType(SrcCol).AsVector;
                        // Only valid use is for two or more slices.
                        Contracts.Assert(Utils.Size(lims) >= 2);
                        Contracts.AssertValue(type);
                        Contracts.Assert(type.VectorSize > 0);
                        Contracts.Assert(type.ItemType.RawType == typeof(T));
                        Contracts.Assert(Utils.IsIncreasing(0, lims, type.VectorSize + 1));
                        Contracts.Assert(lims[lims.Length - 1] == type.VectorSize);

                        _lims = lims;
                        _types = new VectorType[_lims.Length];
                        _types[0] = new VectorType(type.ItemType, _lims[0]);
                        for (int c = 1; c < _lims.Length; ++c)
                            _types[c] = new VectorType(type.ItemType, _lims[c] - _lims[c - 1]);
                    }

                    public override ColumnType GetColumnType(int col)
                    {
                        Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                        return _types[col];
                    }

                    public override IRow Bind(IRow row, Func<int, bool> pred)
                    {
                        Contracts.AssertValue(row);
                        Contracts.Assert(row.Schema == _view.Schema);
                        Contracts.AssertValue(pred);
                        Contracts.Assert(row.IsColumnActive(SrcCol));
                        return new Row(this, row, pred);
                    }

                    private sealed class Row : RowBase<ColumnSplitter<T>>
                    {
                        // Counter of the last valid input, updated by EnsureValid.
                        private long _lastValid;
                        // The last valid input value.
                        private VBuffer<T> _inputValue;
                        // The delegate to get the input value.
                        private readonly ValueGetter<VBuffer<T>> _inputGetter;
                        // The limit of _inputValue.Indices
                        private readonly int[] _srcIndicesLims;
                        // Convenient accessor since we use this all over the place.
                        private int[] Lims { get { return Parent._lims; } }
                        // Getters.
                        private readonly ValueGetter<VBuffer<T>>[] _getters;

                        public Row(ColumnSplitter<T> parent, IRow input, Func<int, bool> pred)
                            : base(parent, input)
                        {
                            _inputGetter = input.GetGetter<VBuffer<T>>(Parent.SrcCol);
                            _srcIndicesLims = new int[Lims.Length];
                            _lastValid = -1;
                            _getters = new ValueGetter<VBuffer<T>>[Lims.Length];
                            for (int c = 0; c < _getters.Length; ++c)
                                _getters[c] = pred(c) ? CreateGetter(c) : null;
                        }

                        public override bool IsColumnActive(int col)
                        {
                            Contracts.CheckParam(0 <= col && col < Parent.ColumnCount, nameof(col));
                            return _getters[col] != null;
                        }

                        public override ValueGetter<TValue> GetGetter<TValue>(int col)
                        {
                            Contracts.Check(IsColumnActive(col));
                            Contracts.AssertValue(_getters[col]);
                            var fn = _getters[col] as ValueGetter<TValue>;
                            if (fn == null)
                                throw Contracts.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                            return fn;
                        }

                        private ValueGetter<VBuffer<T>> CreateGetter(int col)
                        {
                            int min = col == 0 ? 0 : Lims[col - 1];
                            int len = Lims[col] - min;
                            return
                                (ref VBuffer<T> value) =>
                                {
                                    EnsureValid();
                                    var values = value.Values;
                                    if (_inputValue.IsDense)
                                    {
                                        Utils.EnsureSize(ref values, len);
                                        Array.Copy(_inputValue.Values, min, values, 0, len);
                                        value = new VBuffer<T>(len, values, value.Indices);
                                        return;
                                    }
                                    // In the sparse case we have ranges on Indices/Values to consider.
                                    int smin = col == 0 ? 0 : _srcIndicesLims[col - 1];
                                    int slim = _srcIndicesLims[col];
                                    int scount = slim - smin;
                                    if (scount == 0)
                                    {
                                        value = new VBuffer<T>(len, 0, value.Values, value.Indices);
                                        return;
                                    }
                                    var indices = value.Indices;
                                    Utils.EnsureSize(ref indices, scount);
                                    Utils.EnsureSize(ref values, scount);
                                    Array.Copy(_inputValue.Indices, smin, indices, 0, scount);
                                    if (min != 0)
                                    {
                                        for (int i = 0; i < scount; ++i)
                                            indices[i] -= min;
                                    }
                                    Array.Copy(_inputValue.Values, smin, values, 0, scount);
                                    value = new VBuffer<T>(len, scount, values, indices);
                                };
                        }

                        private void EnsureValid()
                        {
                            if (_lastValid == Input.Position)
                                return;
                            _inputGetter(ref _inputValue);
                            Contracts.Assert(_inputValue.Length == Parent._lims[Parent._lims.Length - 1]);
                            // If it's dense, there's no need to determine the beginnings
                            // and end of each slice.
                            if (_inputValue.IsDense)
                                return;
                            if (_inputValue.Count == 0)
                            {
                                // Handle this separately, since _inputValue.Indices might be null
                                // in this case, and then we may as well short circuit it anyway.
                                Array.Clear(_srcIndicesLims, 0, _srcIndicesLims.Length);
                                return;
                            }
                            var indices = _inputValue.Indices;
                            Contracts.AssertValue(indices);

                            int ii = 0;
                            for (int i = 0; i < Lims.Length; ++i)
                            {
                                int lim = Lims[i];
                                // REVIEW: Would some form of bisection search be better
                                // than this scan? Possibly if the search were to happen across
                                // all lims at the same time, somehow.
                                while (ii < _inputValue.Count && indices[ii] < lim)
                                    ii++;
                                _srcIndicesLims[i] = ii;
                            }
                            _lastValid = Input.Position;
                        }
                    }
                }
            }

            /// <summary>
            /// The cursor implementation creates the <see cref="IRow"/>s using <see cref="Splitter.Bind"/>,
            /// then collates the results from those rows as effectively one big row.
            /// </summary>
            private sealed class Cursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
            {
                private readonly DataViewSlicer _slicer;
                private readonly IRow[] _sliceRows;

                public Schema Schema => _slicer.Schema;

                public Cursor(IChannelProvider provider, DataViewSlicer slicer, IRowCursor input, Func<int, bool> pred, bool[] activeSplitters)
                    : base(provider, input)
                {
                    Ch.AssertValue(slicer);
                    Ch.AssertValueOrNull(pred);
                    Ch.Assert(Utils.Size(activeSplitters) == slicer._splitters.Length);

                    _slicer = slicer;
                    _sliceRows = new IRow[_slicer._splitters.Length];
                    var activeSrc = new bool[slicer._splitters.Length];
                    var activeSrcSet = new HashSet<int>();
                    int offset = 0;
                    Func<int, bool> defaultPred = null;
                    if (pred == null)
                        defaultPred = col => true;

                    for (int i = 0; i < activeSplitters.Length; ++i)
                    {
                        var splitter = _slicer._splitters[i];
                        var localOffset = offset;
                        if (activeSplitters[i])
                            _sliceRows[i] = splitter.Bind(input, pred == null ? defaultPred : col => pred(col + localOffset));
                        offset += splitter.ColumnCount;
                    }
                }

                public bool IsColumnActive(int col)
                {
                    Ch.Check(0 <= col && col < Schema.ColumnCount, "col");
                    int splitInd;
                    int splitCol;
                    _slicer.OutputColumnToSplitterIndices(col, out splitInd, out splitCol);
                    return _sliceRows[splitInd] != null && _sliceRows[splitInd].IsColumnActive(splitCol);
                }

                public ValueGetter<TValue> GetGetter<TValue>(int col)
                {
                    Ch.Check(IsColumnActive(col));
                    int splitInd;
                    int splitCol;
                    _slicer.OutputColumnToSplitterIndices(col, out splitInd, out splitCol);
                    return _sliceRows[splitInd].GetGetter<TValue>(splitCol);
                }
            }
        }
    }

    public static class TransposerUtils
    {
        /// <summary>
        /// This is a convenience method that extracts a single slot value's vector,
        /// while simultaneously verifying that there is exactly one value.
        /// </summary>
        public static void GetSingleSlotValue<T>(this ITransposeDataView view, int col, ref VBuffer<T> dst)
        {
            Contracts.CheckValue(view, nameof(view));
            Contracts.CheckParam(0 <= col && col < view.Schema.ColumnCount, nameof(col));
            using (var cursor = view.GetSlotCursor(col))
            {
                var getter = cursor.GetGetter<T>();
                if (!cursor.MoveNext())
                    throw Contracts.Except("Could not get single value on column '{0}' because there are no slots", view.Schema.GetColumnName(col));
                getter(ref dst);
                if (cursor.MoveNext())
                    throw Contracts.Except("Could not get single value on column '{0}' because there is more than one slot", view.Schema.GetColumnName(col));
            }
        }

        /// <summary>
        /// The <see cref="ISlotCursor.GetGetter{TValue}"/> is parameterized by a type that becomes the
        /// type parameter for a <see cref="VBuffer{T}"/>, and this is generally preferable and more
        /// sensible but for various reasons it's often a lot simpler to have a get-getter be over
        /// the actual type returned by the getter, that is, parameterize this by the actual
        /// <see cref="VBuffer{T}"/> type.
        /// </summary>
        /// <typeparam name="TValue">The type, must be a <see cref="VBuffer{T}"/> generic type,
        /// though enforcement of this has to be done only at runtime for practical reasons</typeparam>
        /// <param name="cursor">The cursor to get the getter for</param>
        /// <param name="ctx">The exception contxt</param>
        /// <returns>The value getter</returns>
        public static ValueGetter<TValue> GetGetterWithVectorType<TValue>(this ISlotCursor cursor, IExceptionContext ctx = null)
        {
            Contracts.CheckValueOrNull(ctx);
            ctx.CheckValue(cursor, nameof(cursor));
            var type = typeof(TValue);
            if (!type.IsGenericEx(typeof(VBuffer<>)))
                throw ctx.Except("Invalid TValue: '{0}'", typeof(TValue));
            var genTypeArgs = type.GetGenericArguments();
            ctx.Assert(genTypeArgs.Length == 1);

            Func<ValueGetter<VBuffer<int>>> del = cursor.GetGetter<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(genTypeArgs[0]);
            var getter = methodInfo.Invoke(cursor, null) as ValueGetter<TValue>;
            if (getter == null)
                throw ctx.Except("Invalid TValue: '{0}'", typeof(TValue));
            return getter;
        }

        /// <summary>
        /// Given a slot cursor, construct a single-column equivalent row cursor, with the single column
        /// active and having the same type. This is useful to exploit the many utility methods that exist
        /// to handle <see cref="IRowCursor"/> and <see cref="IRow"/> but that know nothing about
        /// <see cref="ISlotCursor"/>, without having to rewrite all of them. This is, however, rather
        /// something of a hack; whenever possible or reasonable the slot cursor should be used directly.
        /// The name of this column is always "Waffles".
        /// </summary>
        /// <param name="provider">The channel provider used in creating the wrapping row cursor</param>
        /// <param name="cursor">The slot cursor to wrap</param>
        /// <returns>A row cursor with a single active column with the same type as the slot type</returns>
        public static IRowCursor GetRowCursorShim(IChannelProvider provider, ISlotCursor cursor)
        {
            Contracts.CheckValue(provider, nameof(provider));
            provider.CheckValue(cursor, nameof(cursor));

            return Utils.MarshalInvoke(GetRowCursorShimCore<int>, cursor.GetSlotType().ItemType.RawType, provider, cursor);
        }

        private static IRowCursor GetRowCursorShimCore<T>(IChannelProvider provider, ISlotCursor cursor)
        {
            return new SlotRowCursorShim<T>(provider, cursor);
        }

        /// <summary>
        /// Presents a single transposed column as a single-column dataview.
        /// </summary>
        public sealed class SlotDataView : IDataView
        {
            private readonly IHost _host;
            private readonly ITransposeDataView _data;
            private readonly int _col;
            private readonly ColumnType _type;
            private readonly SchemaImpl _schemaImpl;

            public Schema Schema => _schemaImpl.AsSchema;

            public bool CanShuffle { get { return false; } }

            public SlotDataView(IHostEnvironment env, ITransposeDataView data, int col)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register("SlotDataView");
                _host.CheckValue(data, nameof(data));
                _host.CheckParam(0 <= col && col < data.Schema.ColumnCount, nameof(col));
                _type = data.TransposeSchema.GetSlotType(col);
                _host.AssertValue(_type);

                _data = data;
                _col = col;
                _schemaImpl = new SchemaImpl(this);
            }

            public long? GetRowCount(bool lazy = true)
            {
                var type = _data.Schema.GetColumnType(_col);
                int valueCount = type.ValueCount;
                _host.Assert(valueCount > 0);
                return valueCount;
            }

            public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                _host.CheckValue(predicate, nameof(predicate));
                return Utils.MarshalInvoke(GetRowCursor<int>, _type.ItemType.RawType, predicate(0));
            }

            private IRowCursor GetRowCursor<T>(bool active)
            {
                return new Cursor<T>(this, active);
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
            {
                _host.CheckValue(predicate, nameof(predicate));
                consolidator = null;
                return new IRowCursor[] { GetRowCursor(predicate, rand) };
            }

            private sealed class SchemaImpl : ISchema
            {
                private readonly SlotDataView _parent;

                private IHost Host { get { return _parent._host; } }

                public Schema AsSchema { get; }

                public int ColumnCount { get { return 1; } }

                public SchemaImpl(SlotDataView parent)
                {
                    Contracts.AssertValue(parent);
                    _parent = parent;
                    AsSchema = Data.Schema.Create(this);
                }

                public ColumnType GetColumnType(int col)
                {
                    Host.CheckParam(col == 0, nameof(col));
                    return _parent._type;
                }

                public string GetColumnName(int col)
                {
                    Host.CheckParam(col == 0, nameof(col));
                    // There is no real need for this to have the real name as the internal IDV
                    // substream does not have its name accessed, but we'll save it just the same.
                    // I am tempted though to just have this thing always claim its name is 'Pancakes'.
                    return _parent._data.Schema.GetColumnName(_parent._col);
                }

                public bool TryGetColumnIndex(string name, out int col)
                {
                    if (name == GetColumnName(0))
                    {
                        col = 0;
                        return true;
                    }
                    col = -1;
                    return false;
                }

                // No metadata. The top level IDV will hold the schema information, including metadata.
                // This per-column dataview schema information is just minimally functional.

                public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
                {
                    Host.CheckParam(col == 0, nameof(col));
                    return Enumerable.Empty<KeyValuePair<string, ColumnType>>();
                }

                public ColumnType GetMetadataTypeOrNull(string kind, int col)
                {
                    Host.CheckNonEmpty(kind, nameof(kind));
                    Host.CheckParam(col == 0, nameof(col));
                    return null;
                }

                public void GetMetadata<TValue>(string kind, int col, ref TValue value)
                {
                    Host.CheckNonEmpty(kind, nameof(kind));
                    Host.CheckParam(col == 0, nameof(col));
                    throw MetadataUtils.ExceptGetMetadata();
                }
            }

            private sealed class Cursor<T> : SynchronizedCursorBase<ISlotCursor>, IRowCursor
            {
                private readonly SlotDataView _parent;
                private readonly Delegate _getter;

                public Schema Schema => _parent.Schema;

                public Cursor(SlotDataView parent, bool active)
                    : base(parent._host, parent._data.GetSlotCursor(parent._col))
                {
                    _parent = parent;
                    if (active)
                        _getter = Input.GetGetter<T>();
                }

                public bool IsColumnActive(int col)
                {
                    Ch.CheckParam(col == 0, nameof(col));
                    return _getter != null;
                }

                public ValueGetter<TValue> GetGetter<TValue>(int col)
                {
                    Ch.CheckParam(col == 0, nameof(col));
                    Ch.CheckParam(_getter != null, nameof(col), "requested column not active");

                    var getter = _getter as ValueGetter<TValue>;
                    if (getter == null)
                        throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                    return getter;
                }
            }
        }

        // REVIEW: This shim class is very similar to the above shim class, except at the
        // cursor level, not the cursorable level. Is there some non-horrifying way to unify both, somehow?
        private sealed class SlotRowCursorShim<T> : SynchronizedCursorBase<ISlotCursor>, IRowCursor
        {
            private readonly SchemaImpl _schema;

            public Schema Schema => _schema.AsSchema;

            private sealed class SchemaImpl : ISchema
            {
                private readonly SlotRowCursorShim<T> _parent;
                private readonly VectorType _type;

                private IChannel Ch { get { return _parent.Ch; } }

                public Schema AsSchema { get; }

                public int ColumnCount { get { return 1; } }

                public SchemaImpl(SlotRowCursorShim<T> parent, VectorType slotType)
                {
                    Contracts.AssertValue(parent);
                    _parent = parent;
                    Ch.AssertValue(slotType);
                    _type = slotType;
                    AsSchema = Schema.Create(this);
                }

                public ColumnType GetColumnType(int col)
                {
                    Ch.CheckParam(col == 0, nameof(col));
                    return _type;
                }

                public string GetColumnName(int col)
                {
                    Ch.CheckParam(col == 0, nameof(col));
                    return "Waffles";
                }

                public bool TryGetColumnIndex(string name, out int col)
                {
                    if (name == GetColumnName(0))
                    {
                        col = 0;
                        return true;
                    }
                    col = -1;
                    return false;
                }

                public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
                {
                    Ch.CheckParam(col == 0, nameof(col));
                    return Enumerable.Empty<KeyValuePair<string, ColumnType>>();
                }

                public ColumnType GetMetadataTypeOrNull(string kind, int col)
                {
                    Ch.CheckNonEmpty(kind, nameof(kind));
                    Ch.CheckParam(col == 0, nameof(col));
                    return null;
                }

                public void GetMetadata<TValue>(string kind, int col, ref TValue value)
                {
                    Ch.CheckNonEmpty(kind, nameof(kind));
                    Ch.CheckParam(col == 0, nameof(col));
                    throw MetadataUtils.ExceptGetMetadata();
                }
            }

            public SlotRowCursorShim(IChannelProvider provider, ISlotCursor cursor)
                : base(provider, cursor)
            {
                _schema = new SchemaImpl(this, Input.GetSlotType());
            }

            public bool IsColumnActive(int col)
            {
                Ch.CheckParam(col == 0, nameof(col));
                return true;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.CheckParam(col == 0, nameof(col));
                return Input.GetGetterWithVectorType<TValue>(Ch);
            }
        }

        /// <summary>
        /// This <see cref="ITransposeSchema"/> implementation wraps an <see cref="ISchema"/>,
        /// while indicating that no columns are actually transposed. This is useful for
        /// <see cref="ITransposeDataView"/> implementations that are wrapping a data view
        /// that might not implement that interface.
        /// </summary>
        internal sealed class SimpleTransposeSchema : ITransposeSchema
        {
            private readonly ISchema _schema;

            public int ColumnCount { get { return _schema.ColumnCount; } }

            public SimpleTransposeSchema(ISchema schema)
            {
                Contracts.CheckValue(schema, nameof(schema));
                _schema = schema;
            }

            public string GetColumnName(int col)
            {
                return _schema.GetColumnName(col);
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                return _schema.TryGetColumnIndex(name, out col);
            }

            public ColumnType GetColumnType(int col)
            {
                return _schema.GetColumnType(col);
            }

            public VectorType GetSlotType(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return null;
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                return _schema.GetMetadataTypes(col);
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                return _schema.GetMetadataTypeOrNull(kind, col);
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                _schema.GetMetadata(kind, col, ref value);
            }
        }
    }
}