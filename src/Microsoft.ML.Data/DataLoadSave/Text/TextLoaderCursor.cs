// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    public sealed partial class TextLoader : IDataLoader
    {
        private sealed class Cursor : RootCursorBase, IRowCursor
        {
            // Lines are divided into batches and processed a batch at a time. This enables
            // parallel parsing.
            private const int BatchSize = 64;

            private readonly Bindings _bindings;
            private readonly Parser _parser;
            private readonly bool[] _active; // Which columns are active.
            private readonly int _srcNeeded; // Largest source index that is needed by this cursor.
            private readonly LineReader _reader;
            private readonly IEnumerator<int> _ator;
            private readonly Delegate[] _getters;

            private readonly ParseStats _stats;
            private readonly RowSet _rows;

            // This holds the overall line from the line reader currently served up in the cursor.
            private long _total;
            private long _batch;
            private bool _disposed;

            public override long Batch
            {
                get { return _batch; }
            }

            private static void SetupCursor(TextLoader parent, bool[] active, int n,
                out int srcNeeded, out int cthd)
            {
                // Note that files is allowed to be empty.
                Contracts.AssertValue(parent);
                Contracts.AssertValue(parent._files);
                Contracts.Assert(active == null || active.Length == parent._bindings.Infos.Length);

                var bindings = parent._bindings;

                // This ensures _srcNeeded is >= 0.
                int srcLim = 1;
                for (int i = 0; i < bindings.Infos.Length; i++)
                {
                    if (active != null && !active[i])
                        continue;
                    var info = bindings.Infos[i];
                    foreach (var seg in info.Segments)
                    {
                        if (srcLim < seg.Lim)
                            srcLim = seg.Lim;
                    }
                }

                if (srcLim > parent._inputSize && parent._inputSize > 0)
                    srcLim = parent._inputSize;
                srcNeeded = srcLim - 1;
                Contracts.Assert(srcNeeded >= 0);

                // Determine the number of threads to use.
                cthd = DataViewUtils.GetThreadCount(parent._host, n, !parent._useThreads);

                long cblkMax = parent._maxRows / BatchSize;
                if (cthd > cblkMax)
                    cthd = Math.Max(1, (int)cblkMax);
            }

            // Note that we don't filter out rows with parsing issues since it's not acceptable to
            // produce a different set of rows when subsetting columns. Any parsing errors need to be
            // translated to NaN, not result in skipping the row. We should produce some diagnostics
            // to alert the user to the issues.
            private Cursor(TextLoader parent, ParseStats stats, bool[] active, LineReader reader, int srcNeeded, int cthd)
                : base(parent._host)
            {
                Ch.AssertValue(parent._files);
                Ch.Assert(active == null || active.Length == parent._bindings.Infos.Length);
                Ch.AssertValue(reader);
                Ch.AssertValue(stats);
                Ch.Assert(srcNeeded >= 0);
                Ch.Assert(cthd > 0);

                _total = -1;
                _batch = -1;
                _bindings = parent._bindings;
                _parser = parent._parser;
                _active = active;
                _reader = reader;
                _stats = stats;
                _srcNeeded = srcNeeded;

                ParallelState state = null;
                if (cthd > 1)
                    state = new ParallelState(this, out _rows, cthd);
                else
                    _rows = _parser.CreateRowSet(_stats, 1, _active);

                try
                {
                    _getters = new Delegate[_bindings.Infos.Length];
                    for (int i = 0; i < _getters.Length; i++)
                    {
                        if (_active != null && !_active[i])
                            continue;
                        ColumnPipe v = _rows.Pipes[i];
                        Ch.Assert(v != null);
                        _getters[i] = v.GetGetter();
                        Ch.Assert(_getters[i] != null);
                    }

                    if (state != null)
                    {
                        _ator = ParseParallel(state).GetEnumerator();
                        state = null;
                    }
                    else
                        _ator = ParseSequential().GetEnumerator();
                }
                finally
                {
                    if (state != null)
                        state.Dispose();
                }
            }

            public static IRowCursor Create(TextLoader parent, bool[] active)
            {
                // Note that files is allowed to be empty.
                Contracts.AssertValue(parent);
                Contracts.AssertValue(parent._files);
                Contracts.Assert(active == null || active.Length == parent._bindings.Infos.Length);

                int srcNeeded;
                int cthd;
                SetupCursor(parent, active, 0, out srcNeeded, out cthd);
                Contracts.Assert(cthd > 0);

                var reader = new LineReader(parent._files, BatchSize, 100, parent.HasHeader, parent._maxRows, 1);
                var stats = new ParseStats(parent._host, 1);
                return new Cursor(parent, stats, active, reader, srcNeeded, cthd);
            }

            public static IRowCursor[] CreateSet(out IRowCursorConsolidator consolidator,
                TextLoader parent, bool[] active, int n)
            {
                // Note that files is allowed to be empty.
                Contracts.AssertValue(parent);
                Contracts.AssertValue(parent._files);
                Contracts.Assert(active == null || active.Length == parent._bindings.Infos.Length);

                int srcNeeded;
                int cthd;
                SetupCursor(parent, active, n, out srcNeeded, out cthd);
                Contracts.Assert(cthd > 0);

                var reader = new LineReader(parent._files, BatchSize, 100, parent.HasHeader, parent._maxRows, cthd);
                var stats = new ParseStats(parent._host, cthd);
                if (cthd <= 1)
                {
                    consolidator = null;
                    return new IRowCursor[1] { new Cursor(parent, stats, active, reader, srcNeeded, 1) };
                }

                consolidator = new Consolidator(cthd);
                var cursors = new IRowCursor[cthd];
                try
                {
                    for (int i = 0; i < cursors.Length; i++)
                        cursors[i] = new Cursor(parent, stats, active, reader, srcNeeded, 1);
                    var result = cursors;
                    cursors = null;
                    return result;
                }
                finally
                {
                    if (cursors != null)
                    {
                        foreach (var curs in cursors)
                        {
                            if (curs != null)
                                curs.Dispose();
                            else
                            {
                                reader.Release();
                                stats.Release();
                            }
                        }
                    }
                }
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                return
                    (ref UInt128 val) =>
                    {
                        Ch.Check(IsGood, "Cannot call ID getter in current state");
                        val = new UInt128((ulong)_total, 0);
                    };
            }

            public static void GetSomeLines(IMultiStreamSource source, int count, ref List<DvText> lines)
            {
                Contracts.AssertValue(source);
                Contracts.Assert(count > 0);
                Contracts.AssertValueOrNull(lines);

                if (count < 2)
                    count = 2;

                LineBatch batch;
                var reader = new LineReader(source, count, 1, false, count, 1);
                try
                {
                    batch = reader.GetBatch();
                    Contracts.Assert(batch.Exception == null);
                    if (Utils.Size(batch.Infos) == 0)
                        return;
                }
                finally
                {
                    reader.Release();
                }

                for (int i = 0; i < batch.Infos.Length; i++)
                    Utils.Add(ref lines, new DvText(batch.Infos[i].Text));
            }

            /// <summary>
            /// Look in the first file for args embedded as comments. This gathers comments
            /// that come before any data line that start with #@.
            /// </summary>
            public static string GetEmbeddedArgs(IMultiStreamSource files)
            {
                Contracts.AssertValue(files);

                if (files.Count == 0)
                    return null;

                StringBuilder sb = new StringBuilder();
                using (var rdr = files.OpenTextReader(0))
                {
                    string pre = "";
                    for (; ; )
                    {
                        string text = rdr.ReadLine();
                        if (text == null)
                            break;

                        if (text.Length == 0)
                            continue;
                        if (text.StartsWith("//"))
                            continue;
                        if (text[0] != '#')
                            break;
                        if (text.Length <= 2 || text[1] != '@')
                            continue;

                        sb.Append(pre).Append(text.Substring(2).Trim());
                        pre = " ";
                    }
                }
                return sb.ToString();
            }

            public ISchema Schema { get { return _bindings; } }

            public override void Dispose()
            {
                if (_disposed)
                    return;

                _disposed = true;
                _ator.Dispose();
                _reader.Release();
                _stats.Release();
                base.Dispose();
            }

            protected override bool MoveNextCore()
            {
                Contracts.Assert(State != CursorState.Done);

                if (_ator.MoveNext())
                {
                    _rows.Index = _ator.Current;
                    return true;
                }

                _rows.Index = -1;
                return false;
            }

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.Infos.Length);
                return _active == null || _active[col];
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));
                var fn = _getters[col] as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            private IEnumerable<int> ParseSequential()
            {
                Ch.AssertValue(_rows);
                Ch.Assert(_rows.Count == 1);

                LineBatch batch;
                var helper = _parser.CreateHelper(_rows.Stats, _srcNeeded);
                while ((batch = _reader.GetBatch()).Infos != null)
                {
                    Ch.Assert(batch.Exception == null);
                    _total = batch.Total;
                    foreach (var info in batch.Infos)
                    {
                        Ch.Assert(info.Line > 0);
                        Ch.AssertNonEmpty(info.Text);
                        _parser.ParseRow(_rows, 0, helper, _active, batch.Path, info.Line, info.Text);
                        _batch = batch.Batch;
                        yield return 0;
                        ++_total;
                    }
                    _batch = long.MaxValue;
                }
            }

            // Time-out in milliseconds for waiting on an event. This allows checking for
            // abort situations.
            private const int TimeOut = 100;

            private struct LineBatch
            {
                public readonly string Path;
                // Total lines, up to the first line of this batch.
                public readonly long Total;
                public readonly long Batch;
                public readonly LineInfo[] Infos;
                public readonly Exception Exception;

                public LineBatch(string path, long total, long batch, LineInfo[] infos)
                {
                    Contracts.AssertValueOrNull(path);
                    Contracts.AssertValueOrNull(infos);
                    Path = path;
                    Total = total;
                    Batch = batch;
                    Infos = infos;
                    Exception = null;
                }

                public LineBatch(Exception ex)
                {
                    Contracts.AssertValue(ex);
                    Path = null;
                    Total = 0;
                    Batch = 0;
                    Infos = null;
                    Exception = ex;
                }
            }

            private struct LineInfo
            {
                public readonly long Line;
                public readonly string Text;

                public LineInfo(long line, string text)
                {
                    Contracts.Assert(line > 0);
                    Contracts.AssertNonEmpty(text);
                    Line = line;
                    Text = text;
                }
            }

            // This reads batches of lines on a separate thread. It filters out a header, comments, and blank lines.
            private sealed class LineReader
            {
                private readonly long _limit;
                private readonly bool _hasHeader;
                private readonly int _batchSize;
                private readonly IMultiStreamSource _files;

                // The line reader can be referenced by multiple workers. This is the reference count.
                private int _cref;
                private BlockingCollection<LineBatch> _queue;
                private Thread _thdRead;
                private volatile bool _abort;

                public LineReader(IMultiStreamSource files, int batchSize, int bufSize, bool hasHeader, long limit, int cref)
                {
                    // Note that files is allowed to be empty.
                    Contracts.AssertValue(files);
                    Contracts.Assert(files.Count >= 0);
                    Contracts.Assert(batchSize >= 2);
                    Contracts.Assert(bufSize > 0);
                    Contracts.Assert(limit >= 0);
                    Contracts.Assert(cref > 0);

                    _limit = limit;
                    _hasHeader = hasHeader;
                    _batchSize = batchSize;
                    _files = files;
                    _cref = cref;

                    _queue = new BlockingCollection<LineBatch>(bufSize);
                    _thdRead = Utils.CreateBackgroundThread(ThreadProc);
                    _thdRead.Start();
                }

                public void Release()
                {
                    int n = Interlocked.Decrement(ref _cref);
                    Contracts.Assert(n >= 0);

                    if (n != 0)
                        return;

                    if (_thdRead != null)
                    {
                        _abort = true;
                        _thdRead.Join();
                        _thdRead = null;
                    }

                    if (_queue != null)
                    {
                        _queue.Dispose();
                        _queue = null;
                    }
                }

                public LineBatch GetBatch()
                {
                    Exception inner = null;
                    try
                    {
                        var batch = _queue.Take();
                        if (batch.Exception == null)
                            return batch;
                        inner = batch.Exception;
                    }
                    catch (InvalidOperationException)
                    {
                        // This code detects when there is no more content by catching the exception thrown by _queue.Take() at the end.
                        // If _queue.IsAddingCompleted is true, we know that this exception was the result of that specifically.
                        // This should probably be re-engineered to not rely on exception blocks.
                        // REVIEW: Come up with a less strange scheme for the interthread communication here, that does not
                        // rely on exceptions and timeouts throughout the pipeline.
                        if (_queue.IsAddingCompleted)
                            return default(LineBatch);
                        throw;
                    }
                    Contracts.AssertValue(inner);
                    throw Contracts.ExceptDecode(inner, "Stream reading encountered exception");
                }

                private void ThreadProc()
                {
                    Contracts.Assert(_batchSize >= 2);

                    try
                    {
                        if (_limit <= 0)
                            return;

                        long total = 0;
                        long batch = -1;
                        for (int ifile = 0; ifile < _files.Count; ifile++)
                        {
                            string path = _files.GetPathOrNull(ifile);
                            using (var rdr = _files.OpenTextReader(ifile))
                            {
                                string text;
                                long line = 0;
                                for (; ; )
                                {
                                    // REVIEW: Avoid allocating a string for every line. This would probably require
                                    // introducing a CharSpan type (similar to DvText but based on char[] or StringBuilder)
                                    // and implementing all the necessary conversion functionality on it. See task 3871.
                                    text = rdr.ReadLine();
                                    if (text == null)
                                        goto LNext;
                                    line++;
                                    if (text.Length > 0 && text[0] != '#' && !text.StartsWith("//"))
                                        break;
                                }

                                // REVIEW: Use a pool of batches?
                                int index = 0;
                                var infos = new LineInfo[_batchSize];
                                if (!_hasHeader)
                                {
                                    // Not a header or comment, so first line is a real line.
                                    infos[index++] = new LineInfo(line, text);
                                    if (++total >= _limit)
                                    {
                                        PostPartial(path, total - index, ref batch, index, infos);
                                        return;
                                    }
                                }

                                for (; ; )
                                {
                                    if (_abort)
                                        return;

                                    text = rdr.ReadLine();
                                    if (text == null)
                                    {
                                        // We're done with this file. Queue the last partial batch.
                                        PostPartial(path, total - index, ref batch, index, infos);
                                        goto LNext;
                                    }
                                    line++;

                                    // Filter out comments and empty strings.
                                    if (text.Length >= 2)
                                    {
                                        // Don't use string.StartsWith("//") - it is too slow.
                                        if (text[0] == '/' && text[1] == '/')
                                            continue;
                                    }
                                    else if (text.Length == 0)
                                        continue;

                                    infos[index] = new LineInfo(line, text);
                                    if (++index >= infos.Length)
                                    {
                                        batch++;
                                        var lines = new LineBatch(path, total - index + 1, batch, infos);
                                        while (!_queue.TryAdd(lines, TimeOut))
                                        {
                                            if (_abort)
                                                return;
                                        }
                                        infos = new LineInfo[_batchSize];
                                        index = 0;
                                    }
                                    if (++total >= _limit)
                                    {
                                        PostPartial(path, total - index, ref batch, index, infos);
                                        return;
                                    }
                                }

                            LNext:
                                ;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        while (!_queue.TryAdd(new LineBatch(ex), TimeOut))
                        {
                            if (_abort)
                                return;
                        }
                    }
                    finally
                    {
                        _queue.CompleteAdding();
                    }
                }

                private void PostPartial(string path, long total, ref long batch, int index, LineInfo[] infos)
                {
                    Contracts.AssertValueOrNull(path);
                    Contracts.Assert(0 <= total);
                    Contracts.Assert(0 <= index && index < Utils.Size(infos));

                    // Queue the last partial batch.
                    if (index <= 0)
                        return;

                    Array.Resize(ref infos, index);
                    batch++;
                    while (!_queue.TryAdd(new LineBatch(path, total, batch, infos), TimeOut))
                    {
                        if (_abort)
                            return;
                    }
                }
            }

            private IEnumerable<int> ParseParallel(ParallelState state)
            {
                using (state)
                {
                    foreach (var batch in state.GetBatches())
                    {
                        // If the collation of rows happened correctly, this should have a precise value.
                        Contracts.Assert(batch.Total == _total + 1);
                        _total = batch.Total - 1;
                        for (int irow = batch.IrowMin; irow < batch.IrowLim; irow++)
                        {
                            ++_total;
                            yield return irow;
                        }
                    }
                    if (state.ParsingException != null)
                    {
                        throw Ch.ExceptDecode(state.ParsingException,
                            "Parsing failed with an exception: {0}", state.ParsingException.Message);
                    }
                }
            }

            private struct RowBatch
            {
                public int IrowMin;
                public int IrowLim;
                public long Total;

                public RowBatch(int irowMin, int irowLim, long total)
                {
                    Contracts.Assert(0 <= irowMin && irowMin < irowLim);
                    Contracts.Assert(total >= 0);
                    IrowMin = irowMin;
                    IrowLim = irowLim;
                    Total = total;
                }
            }

            private sealed class ParallelState : IDisposable
            {
                private readonly Cursor _curs;
                private readonly LineReader _reader;
                private readonly RowSet _rows;

                // Number of blocks in this RowSet.
                private readonly int _blockCount;
                // Size of blocks in this RowSet.
                private const int BlockSize = BatchSize;

                // Ordered waiters on block numbers.
                // _waiterWorking orders reading from _reader.
                // _waiterPublish orders publishing to _queue.
                private readonly OrderedWaiter _waiterReading;
                private readonly OrderedWaiter _waiterPublish;

                // A small capacity blocking collection that the main cursor thread consumes.
                private readonly BlockingCollection<RowBatch> _queue;

                private readonly Thread[] _threads;

                // Number of threads still running.
                private int _threadsRunning;

                // Signals threads to shut down.
                private volatile bool _done;

                // Exception during parsing.
                public volatile Exception ParsingException;

                public ParallelState(Cursor curs, out RowSet rows, int cthd)
                {
                    Contracts.AssertValue(curs);
                    Contracts.Assert(cthd > 0);

                    _curs = curs;
                    _reader = _curs._reader;

                    // Why cthd + 3? We need two blocks for the blocking collection, and one
                    // more for the block currently being dished out by the cursor.
                    _blockCount = cthd + 3;

                    // Why cthd + 3? We need two blocks for the blocking collection, and one
                    // more for the block currently being dished out by the cursor.
                    _rows = rows = _curs._parser.CreateRowSet(_curs._stats,
                        checked(_blockCount * BlockSize), _curs._active);

                    _waiterReading = new OrderedWaiter(firstCleared: false);
                    _waiterPublish = new OrderedWaiter(firstCleared: false);

                    // The size limit here ensures that worker threads are never writing to
                    // a range that is being served up by the cursor.
                    _queue = new BlockingCollection<RowBatch>(2);

                    _threads = new Thread[cthd];
                    _threadsRunning = cthd;

                    for (int tid = 0; tid < _threads.Length; tid++)
                    {
                        var thd = _threads[tid] = Utils.CreateBackgroundThread(ThreadProc);
                        thd.Start(tid);
                    }
                }

                public void Dispose()
                {
                    // Signal all the threads to shut down and wait for them.
                    Quit();
                    for (int i = 0; i < _threads.Length; i++)
                        _threads[i].Join();
                }

                private void Quit()
                {
                    // Signal that we're done and wake up all the threads.
                    _done = true;
                    _waiterReading.IncrementAll();
                    _waiterPublish.IncrementAll();
                }

                public IEnumerable<RowBatch> GetBatches()
                {
                    _waiterReading.Increment();
                    _waiterPublish.Increment();
                    return _queue.GetConsumingEnumerable();
                }

                private void ThreadProc(object obj)
                {
                    // The object is the thread index, or "id".
                    int tid = (int)obj;
                    Contracts.Assert(0 <= tid && tid < _threads.Length);

                    try
                    {
                        Parse(tid);
                    }
                    catch (Exception ex)
                    {
                        // Record the exception and tell everyone to shut down.
                        ParsingException = ex;
                        Quit();
                    }
                    finally
                    {
                        // If this is the last thread to shut down, close the queue.
                        if (Interlocked.Decrement(ref _threadsRunning) <= 0)
                            _queue.CompleteAdding();
                    }
                }

                private void Parse(int tid)
                {
                    long blk = tid;
                    int iblk = tid;
                    Contracts.Assert(iblk < _blockCount - 3);

                    var helper = _curs._parser.CreateHelper(_rows.Stats, _curs._srcNeeded);
                    while (!_done)
                    {
                        // Algorithm:
                        // * When it is our turn, grab a block of lines.
                        // * Parse rows.
                        // * When it is our turn, enqueue the batch.

                        // When it is our turn, read the lines and signal the next worker that it is ok to read.
                        LineBatch lines;
                        _waiterReading.Wait(blk);
                        if (_done)
                            return;
                        try
                        {
                            lines = _reader.GetBatch();
                        }
                        finally
                        {
                            _waiterReading.Increment();
                        }
                        Contracts.Assert(lines.Exception == null);
                        if (lines.Infos == null || _done)
                            return;

                        // Parse the lines into rows.
                        Contracts.Assert(lines.Infos.Length <= BlockSize);

                        var batch = new RowBatch(iblk * BlockSize, iblk * BlockSize + lines.Infos.Length, lines.Total);
                        int irow = batch.IrowMin;
                        foreach (var info in lines.Infos)
                        {
                            Contracts.Assert(info.Line > 0);
                            Contracts.AssertNonEmpty(info.Text);
                            if (_done)
                                return;
                            _curs._parser.ParseRow(_rows, irow, helper, _curs._active, lines.Path, info.Line, info.Text);
                            irow++;
                        }
                        Contracts.Assert(irow == batch.IrowLim);

                        if (_done)
                            return;

                        // When it is our turn, publish the rows and signal the next worker that it is ok to publish.
                        _waiterPublish.Wait(blk);
                        if (_done)
                            return;
                        while (!_queue.TryAdd(batch, TimeOut))
                        {
                            if (_done)
                                return;
                        }
                        _waiterPublish.Increment();

                        blk += _threads.Length;
                        iblk += _threads.Length;
                        if (iblk >= _blockCount)
                            iblk -= _blockCount;
                        Contracts.Assert(0 <= iblk && iblk < _blockCount);
                    }
                }
            }

            /// <summary>
            /// The consolidator object. This simply records the number of threads and checks
            /// that they match at the end.
            /// </summary>
            private sealed class Consolidator : IRowCursorConsolidator
            {
                private int _cthd;

                public Consolidator(int cthd)
                {
                    Contracts.Assert(cthd > 1);
                    _cthd = cthd;
                }

                public IRowCursor CreateCursor(IChannelProvider provider, IRowCursor[] inputs)
                {
                    Contracts.AssertValue(provider);
                    int cthd = Interlocked.Exchange(ref _cthd, 0);
                    provider.Check(cthd > 1, "Consolidator can only be used once");
                    provider.Check(Utils.Size(inputs) == cthd, "Unexpected number of cursors");

                    // ConsolidateGeneric does all the standard validity checks: all cursors non-null,
                    // all have the same schema, all have the same active columns, and all active
                    // column types are cachable.
                    using (var ch = provider.Start("Consolidator"))
                    {
                        var result = DataViewUtils.ConsolidateGeneric(provider, inputs, BatchSize);
                        ch.Done();
                        return result;
                    }
                }
            }
        }
    }
}
