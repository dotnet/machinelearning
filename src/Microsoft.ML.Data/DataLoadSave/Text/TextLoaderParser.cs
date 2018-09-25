// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    public sealed partial class TextLoader
    {
        /// <summary>
        /// This type exists to provide efficient delegates for creating a ColumnValue specific to a DataKind.
        /// </summary>
        private sealed class ValueCreatorCache
        {
            private static volatile ValueCreatorCache _instance;
            public static ValueCreatorCache Instance
            {
                get
                {
                    if (_instance == null)
                        Interlocked.CompareExchange(ref _instance, new ValueCreatorCache(), null);
                    return _instance;
                }
            }

            private readonly Conversions _conv;
            private readonly MethodInfo _methOne;
            private readonly MethodInfo _methVec;

            // Indexed by DataKind.ToIndex()
            private readonly Func<RowSet, ColumnPipe>[] _creatorsOne;
            private readonly Func<RowSet, ColumnPipe>[] _creatorsVec;

            private ValueCreatorCache()
            {
                _conv = Conversions.Instance;
                _methOne = new Func<PrimitiveType, Func<RowSet, ColumnPipe>>(GetCreatorOneCore<int>)
                    .GetMethodInfo().GetGenericMethodDefinition();
                _methVec = new Func<PrimitiveType, Func<RowSet, ColumnPipe>>(GetCreatorVecCore<int>)
                    .GetMethodInfo().GetGenericMethodDefinition();

                _creatorsOne = new Func<RowSet, ColumnPipe>[DataKindExtensions.KindCount];
                _creatorsVec = new Func<RowSet, ColumnPipe>[DataKindExtensions.KindCount];
                for (var kind = DataKindExtensions.KindMin; kind < DataKindExtensions.KindLim; kind++)
                {
                    var type = PrimitiveType.FromKind(kind);
                    _creatorsOne[kind.ToIndex()] = GetCreatorOneCore(type);
                    _creatorsVec[kind.ToIndex()] = GetCreatorVecCore(type);
                }
            }

            private Func<RowSet, ColumnPipe> GetCreatorOneCore(PrimitiveType type)
            {
                MethodInfo meth = _methOne.MakeGenericMethod(type.RawType);
                return (Func<RowSet, ColumnPipe>)meth.Invoke(this, new object[] { type });
            }

            private Func<RowSet, ColumnPipe> GetCreatorOneCore<T>(PrimitiveType type)
            {
                Contracts.Assert(type.IsStandardScalar || type.IsKey);
                Contracts.Assert(typeof(T) == type.RawType);
                var fn = _conv.GetParseConversion<T>(type);
                return rows => new PrimitivePipe<T>(rows, fn);
            }

            private Func<RowSet, ColumnPipe> GetCreatorVecCore(PrimitiveType type)
            {
                MethodInfo meth = _methVec.MakeGenericMethod(type.RawType);
                return (Func<RowSet, ColumnPipe>)meth.Invoke(this, new object[] { type });
            }

            private Func<RowSet, ColumnPipe> GetCreatorVecCore<T>(PrimitiveType type)
            {
                Contracts.Assert(type.IsStandardScalar || type.IsKey);
                Contracts.Assert(typeof(T) == type.RawType);
                var fn = _conv.GetParseConversion<T>(type);
                return rows => new VectorPipe<T>(rows, fn);
            }

            public Func<RowSet, ColumnPipe> GetCreatorOne(KeyType key)
            {
                // Have to produce a specific one - can't use a cached one.
                MethodInfo meth = _methOne.MakeGenericMethod(key.RawType);
                return (Func<RowSet, ColumnPipe>)meth.Invoke(this, new object[] { key });
            }

            public Func<RowSet, ColumnPipe> GetCreatorVec(KeyType key)
            {
                // Have to produce a specific one - can't use a cached one.
                MethodInfo meth = _methVec.MakeGenericMethod(key.RawType);
                return (Func<RowSet, ColumnPipe>)meth.Invoke(this, new object[] { key });
            }

            public Func<RowSet, ColumnPipe> GetCreatorOne(DataKind kind)
            {
                int index = kind.ToIndex();
                Contracts.Assert(0 <= index & index < _creatorsOne.Length);
                return _creatorsOne[index];
            }

            public Func<RowSet, ColumnPipe> GetCreatorVec(DataKind kind)
            {
                int index = kind.ToIndex();
                Contracts.Assert(0 <= index & index < _creatorsOne.Length);
                return _creatorsVec[index];
            }
        }

        /// <summary>
        /// Basic statistics and reporting of unparsable stuff.
        /// </summary>
        private sealed class ParseStats
        {
            // Maximum number of messages to show.
            private const long MaxShow = 10;
            private readonly long _maxShow;

            // The channel to report messages on.
            private readonly IChannel _ch;

            // Reference count.
            private volatile int _cref;

            // Total number of rows, number of unparsable values, number of format errors.
            private long _rowCount;
            private long _badCount;
            private long _fmtCount;

            public ParseStats(IChannelProvider provider, int cref, long maxShow = MaxShow)
            {
                Contracts.CheckValue(provider, nameof(provider));
                _ch = provider.Start("ParseStats");

                _ch.Assert(cref > 0);
                _cref = cref;
                _maxShow = maxShow;
            }

            public void Release()
            {
                int n = Interlocked.Decrement(ref _cref);
                _ch.Assert(n >= 0);

                if (n == 0)
                {
                    if (_badCount > 0 || _fmtCount > 0)
                    {
                        _ch.Info("Processed {0} rows with {1} bad values and {2} format errors",
                            _rowCount, _badCount, _fmtCount);
                    }
                    _ch.Done();
                    _ch.Dispose();
                }
            }

            public void LogRow()
            {
                Interlocked.Increment(ref _rowCount);
            }

            public void LogBadValue(long line, string colName, int slot)
            {
                long n = Interlocked.Increment(ref _badCount);
                if (n <= _maxShow)
                {
                    _ch.Info(MessageSensitivity.Schema, "  Bad value at line {0} in column {1} at slot {2}", line, colName, slot);
                    if (n == _maxShow)
                        _ch.Info("  Suppressing further bad value messages");
                }
            }

            public void LogBadValue(long line, string colName)
            {
                long n = Interlocked.Increment(ref _badCount);
                if (n <= _maxShow)
                {
                    _ch.Info(MessageSensitivity.Schema, "  Bad value at line {0} in column {1}", line, colName);
                    if (n == _maxShow)
                        _ch.Info("  Suppressing further bad value messages");
                }
            }

            public void LogBadFmt(ref ScanInfo scan, string msg)
            {
                long n = Interlocked.Increment(ref _fmtCount);
                if (n <= _maxShow)
                {
                    if (scan.Line > 0)
                    {
                        // The -1 is so the indices are 1-based instead of 0-based.
                        int ichBase = scan.IchMinBuf - 1;
                        _ch.Warning("Format error at {0}({1},{2})-({1},{3}): {4}",
                            scan.Path, scan.Line, scan.IchMin - ichBase, scan.IchLim - ichBase, msg);
                    }
                    else
                        _ch.Warning("Format error: {0}", msg);
                    if (n == _maxShow)
                        _ch.Warning("Suppressing further format error messages");
                }
            }
        }

        private abstract class ColumnPipe
        {
            public readonly RowSet Rows;

            protected ColumnPipe(RowSet rows)
            {
                Contracts.AssertValue(rows);
                Rows = rows;
            }

            public abstract void Reset(int irow, int size);

            // Passed by-ref for effeciency, not so it can be modified.
            public abstract bool Consume(int irow, int index, ref ReadOnlyMemory<char> text);

            public abstract Delegate GetGetter();
        }

        private sealed class PrimitivePipe<TResult> : ColumnPipe
        {
            private readonly TryParseMapper<TResult> _conv;

            // Has length Rows.Count, so indexed by irow.
            private TResult[] _values;

            public PrimitivePipe(RowSet rows, TryParseMapper<TResult> conv)
                : base(rows)
            {
                Contracts.AssertValue(conv);
                _conv = conv;
                _values = new TResult[Rows.Count];
            }

            public override void Reset(int irow, int size)
            {
                Contracts.Assert(0 <= irow && irow < _values.Length);
                Contracts.Assert(size == 0);
                _values[irow] = default(TResult);
            }

            public override bool Consume(int irow, int index, ref ReadOnlyMemory<char> text)
            {
                Contracts.Assert(0 <= irow && irow < _values.Length);
                Contracts.Assert(index == 0);
                return _conv(ref text, out _values[irow]);
            }

            public void Get(ref TResult value)
            {
                int index = Rows.Index;
                Contracts.Assert(-1 <= index && index < Rows.Count);
                Contracts.Check(index >= 0);
                value = _values[index];
            }

            public override Delegate GetGetter()
            {
                return (ValueGetter<TResult>)Get;
            }
        }

        private sealed class VectorPipe<TItem> : ColumnPipe
        {
            private readonly TryParseMapper<TItem> _conv;

            private class VectorValue
            {
                private readonly VectorPipe<TItem> _pipe;
                private readonly TryParseMapper<TItem> _conv;

                // We don't need the full power of the BufferBuilder stuff. We always record things
                // in index order, and never have to combine values.
                private int _size;
                private int _count;
                private int _indexPrev;
                private TItem[] _values;
                private int[] _indices;

                public VectorValue(VectorPipe<TItem> pipe)
                {
                    _pipe = pipe;
                    _conv = pipe._conv;
                    _values = new TItem[4];
                    _indices = new int[4];
                }

                [Conditional("DEBUG")]
                public void AssertValid()
                {
                    if (_size == 0)
                        return;

                    Contracts.Assert(_size > 0);
                    Contracts.Assert(-1 <= _indexPrev);
                    Contracts.Assert(_indexPrev < _size);
                    Contracts.Assert(0 <= _count);
                    Contracts.Assert(_count <= _size);
                    Contracts.Assert(_count <= _values.Length);

                    if (_count < _size)
                    {
                        // We're sparse, so there should not be more than _size/2 items and indices should
                        // be big enough.
                        Contracts.Assert(_count <= _size / 2);
                        Contracts.Assert(_count <= _indices.Length);
                    }
                }

                public void Reset(int size)
                {
                    Contracts.Assert(size >= 0);
                    _size = size;
                    _count = 0;
                    _indexPrev = -1;
                    AssertValid();
                }

                public bool Consume(int index, ref ReadOnlyMemory<char> text)
                {
                    AssertValid();
                    Contracts.Assert(_indexPrev < index & index < _size);

                    TItem tmp = default(TItem);
                    bool f = _conv(ref text, out tmp);
                    if (_count < _size)
                    {
                        if (_count < _size / 2)
                        {
                            // Stay sparse.
                            if (_values.Length <= _count)
                                Array.Resize(ref _values, 2 * _count);
                            if (_indices.Length <= _count)
                                Array.Resize(ref _indices, 2 * _count);
                            _values[_count] = tmp;
                            _indices[_count] = index;
                            _count++;

                            AssertValid();
                            return f;
                        }

                        // Convert to dense.
                        if (_values.Length >= _size)
                            Array.Clear(_values, _count, _size - _count);
                        else
                        {
                            if (_values.Length > _count)
                                Array.Clear(_values, _count, _values.Length - _count);
                            Array.Resize(ref _values, _size);
                        }
                        for (int ii = _count; --ii >= 0; )
                        {
                            int i = _indices[ii];
                            Contracts.Assert(ii <= i);

                            // If ii == i then we have every slot covered below this.
                            if (ii >= i)
                                break;

                            // Must fill vacated slots with default(TItem).
                            _values[i] = _values[ii];
                            _values[ii] = default(TItem);
                        }
                        _count = _size;

                        AssertValid();
                    }

                    Contracts.Assert(_count == _size);
                    _values[index] = tmp;

                    AssertValid();
                    return f;
                }

                public void Get(ref VBuffer<TItem> dst)
                {
                    AssertValid();

                    var values = dst.Values;
                    var indices = dst.Indices;

                    if (_count == 0)
                    {
                        dst = new VBuffer<TItem>(_size, 0, values, indices);
                        return;
                    }

                    if (Utils.Size(values) < _count)
                        values = new TItem[_count];
                    Array.Copy(_values, values, _count);
                    if (_count == _size)
                    {
                        dst = new VBuffer<TItem>(_size, values, indices);
                        return;
                    }

                    if (Utils.Size(indices) < _count)
                        indices = new int[_count];
                    Array.Copy(_indices, indices, _count);
                    dst = new VBuffer<TItem>(_size, _count, values, indices);
                }
            }

            // Has length Rows.Count, so indexed by irow.
            private VectorValue[] _values;

            public VectorPipe(RowSet rows, TryParseMapper<TItem> conv)
                : base(rows)
            {
                Contracts.AssertValue(conv);
                _conv = conv;
                _values = new VectorValue[Rows.Count];
                for (int i = 0; i < _values.Length; i++)
                    _values[i] = new VectorValue(this);
            }

            public override void Reset(int irow, int size)
            {
                Contracts.Assert(0 <= irow && irow < _values.Length);
                Contracts.Assert(size >= 0);
                _values[irow].Reset(size);
            }

            public override bool Consume(int irow, int index, ref ReadOnlyMemory<char> text)
            {
                Contracts.Assert(0 <= irow && irow < _values.Length);
                return _values[irow].Consume(index, ref text);
            }

            public void Get(ref VBuffer<TItem> dst)
            {
                int index = Rows.Index;
                Contracts.Assert(-1 <= index && index < Rows.Count);
                Contracts.Check(index >= 0);
                _values[index].Get(ref dst);
            }

            public override Delegate GetGetter()
            {
                return (ValueGetter<VBuffer<TItem>>)Get;
            }
        }

        private sealed class RowSet
        {
            // The associated parse statistics object. Note that multiple RowSets can share the
            // same stats object.
            public readonly ParseStats Stats;

            // The total number of rows in this row set.
            public readonly int Count;

            // The pipes - one per column. Inactive columns have a null entry.
            public readonly ColumnPipe[] Pipes;

            // Current row index being yielded. Only assigned or read on the main
            // cursor thread (assuming clients don't call the getters from other threads).
            public int Index;

            /// <summary>
            /// Takes the number of blocks, number of rows per block, and number of columns.
            /// </summary>
            public RowSet(ParseStats stats, int count, int ccol)
            {
                Contracts.AssertValue(stats);
                Contracts.Assert(count > 0);

                Stats = stats;
                Count = count;
                Pipes = new ColumnPipe[ccol];
                Index = -1;
            }
        }

        /// <summary>
        /// This is info tracked while scanning a line to find "fields". For each line, the first
        /// several values, Path, Line, LineText, IchMinText, and IchLimText, are unchanging, but the
        /// remaining values are updated for each field processed.
        /// </summary>
        private struct ScanInfo
        {
            /// <summary>
            /// Path for the input file containing the given line (may be null).
            /// </summary>
            public readonly string Path;

            /// <summary>
            /// Line number.
            /// </summary>
            public readonly long Line;

            /// <summary>
            /// The current text for the entire line (all fields), and possibly more.
            /// </summary>
            public ReadOnlyMemory<char> TextBuf;

            /// <summary>
            /// The min position in <see cref="TextBuf"/> to consider (all fields).
            /// </summary>
            public readonly int IchMinBuf;

            /// <summary>
            /// The lim position in <see cref="TextBuf"/> to consider (all fields).
            /// </summary>
            public readonly int IchLimBuf;

            /// <summary>
            /// Where to start for the next field. This is both an input and
            /// output to the code that fetches the next field.
            /// </summary>
            public int IchMinNext;

            /// <summary>
            /// The (unquoted) text of the field.
            /// </summary>
            public ReadOnlyMemory<char> Span;

            /// <summary>
            /// Whether there was a quoting error in the field.
            /// </summary>
            public bool QuotingError;

            /// <summary>
            /// For sparse encoding, this is the index of the field. Otherwise, -1.
            /// </summary>
            public int Index;

            /// <summary>
            /// The start character location in <see cref="TextBuf"/>, including the sparse index
            /// and quoting, if present. Used for logging.
            /// </summary>
            public int IchMin;

            /// <summary>
            /// The end character location in <see cref="TextBuf"/>, including the sparse index
            /// and quoting, if present. Used for logging.
            /// </summary>
            public int IchLim;

            /// <summary>
            /// Initializes the ScanInfo.
            /// </summary>
            public ScanInfo(ref ReadOnlyMemory<char> text, string path, long line)
                : this()
            {
                Contracts.AssertValueOrNull(path);
                Contracts.Assert(line >= 0);

                Path = path;
                Line = line;
                TextBuf = text;
                IchMinBuf = 0;
                IchLimBuf = text.Length;
                IchMinNext = IchMinBuf;
            }
        }

        private sealed class Parser
        {
            /// <summary>
            /// This holds a set of raw text fields. This is the input into the parsing
            /// of the individual typed values.
            /// </summary>
            private sealed class FieldSet
            {
                public int Count;

                // Source indices and associated text (parallel arrays).
                public int[] Indices;
                public ReadOnlyMemory<char>[] Spans;

                public FieldSet()
                {
                    // Always allocate/size Columns after Spans so even if exceptions are thrown we
                    // are guaranteed that Spans.Length >= Columns.Length.
                    Spans = new ReadOnlyMemory<char>[8];
                    Indices = new int[8];
                }

                [Conditional("DEBUG")]
                public void AssertValid()
                {
                    Contracts.AssertValue(Spans);
                    Contracts.AssertValue(Indices);
                    Contracts.Assert(0 <= Count & Count <= Indices.Length & Indices.Length <= Spans.Length);
                }

                [Conditional("DEBUG")]
                public void AssertEmpty()
                {
                    Contracts.AssertValue(Spans);
                    Contracts.AssertValue(Indices);
                    Contracts.Assert(Count == 0);
                }

                /// <summary>
                /// Make sure there is enough space to add one more item.
                /// </summary>
                public void EnsureSpace()
                {
                    AssertValid();
                    if (Count >= Indices.Length)
                    {
                        int size = 2 * Count;
                        if (Spans.Length < size)
                            Array.Resize(ref Spans, size);
                        Array.Resize(ref Indices, size);
                    }
                    AssertValid();
                }

                public void Clear()
                {
                    AssertValid();
                    Array.Clear(Spans, 0, Count);
                    Count = 0;
                    AssertEmpty();
                }
            }

            private readonly char[] _separators;
            private readonly Options _flags;
            private readonly int _inputSize;
            private readonly ColInfo[] _infos;

            // These delegates are used to construct new row objects.
            private readonly Func<RowSet, ColumnPipe>[] _creator;

            private volatile int _csrc;
            private volatile int _mismatchCount;

            public Parser(TextLoader parent)
            {
                Contracts.AssertValue(parent);

                _infos = parent._bindings.Infos;
                _creator = new Func<RowSet, ColumnPipe>[_infos.Length];
                var cache = ValueCreatorCache.Instance;
                var mapOne = new Dictionary<DataKind, Func<RowSet, ColumnPipe>>();
                var mapVec = new Dictionary<DataKind, Func<RowSet, ColumnPipe>>();
                for (int i = 0; i < _creator.Length; i++)
                {
                    var info = _infos[i];

                    if (info.ColType.ItemType.IsKey)
                    {
                        if (!info.ColType.IsVector)
                            _creator[i] = cache.GetCreatorOne(info.ColType.AsKey);
                        else
                            _creator[i] = cache.GetCreatorVec(info.ColType.ItemType.AsKey);
                        continue;
                    }

                    DataKind kind = info.ColType.ItemType.RawKind;
                    Contracts.Assert(kind != 0);
                    var map = info.ColType.IsVector ? mapVec : mapOne;
                    if (!map.TryGetValue(info.Kind, out _creator[i]))
                    {
                        var fn = info.ColType.IsVector ?
                            cache.GetCreatorVec(info.Kind) :
                            cache.GetCreatorOne(info.Kind);
                        map.Add(info.Kind, fn);
                        _creator[i] = fn;
                    }
                }

                _separators = parent._separators;
                _flags = parent._flags;
                _inputSize = parent._inputSize;
                Contracts.Assert(_inputSize >= 0);
            }

            public static void GetInputSize(TextLoader parent, List<ReadOnlyMemory<char>> lines, out int minSize, out int maxSize)
            {
                Contracts.AssertNonEmpty(lines);
                Contracts.Assert(parent._inputSize == 0, "Why is this being called when inputSize is known?");

                minSize = int.MaxValue;
                maxSize = 0;
                var stats = new ParseStats(parent._host, cref: 1, maxShow: 0);
                var impl = new HelperImpl(stats, parent._flags, parent._separators, 0, int.MaxValue);
                try
                {
                    foreach (var line in lines)
                    {
                        var text = (parent._flags & Options.TrimWhitespace) != 0 ? ReadOnlyMemoryUtils.TrimEndWhiteSpace(line) : line;
                        if (text.IsEmpty)
                            continue;

                        // REVIEW: This is doing more work than we need, but makes sure we're consistent....
                        int srcLim = impl.GatherFields(text, text.Span);
                        // Don't need the fields, just srcLim.
                        impl.Fields.Clear();

                        if (srcLim == 0)
                            continue;

                        if (minSize > srcLim)
                            minSize = srcLim;
                        if (maxSize < srcLim)
                            maxSize = srcLim;
                    }
                }
                finally
                {
                    stats.Release();
                }
            }

            public static void ParseSlotNames(TextLoader parent, ReadOnlyMemory<char> textHeader, ColInfo[] infos, VBuffer<ReadOnlyMemory<char>>[] slotNames)
            {
                Contracts.Assert(!textHeader.IsEmpty);
                Contracts.Assert(infos.Length == slotNames.Length);

                var sb = new StringBuilder();
                var stats = new ParseStats(parent._host, cref: 1, maxShow: 0);
                var impl = new HelperImpl(stats, parent._flags, parent._separators, parent._inputSize, int.MaxValue);
                try
                {
                    impl.GatherFields(textHeader, textHeader.Span);
                }
                finally
                {
                    stats.Release();
                }

                var header = impl.Fields;
                var bldr = BufferBuilder<ReadOnlyMemory<char>>.CreateDefault();
                for (int iinfo = 0; iinfo < infos.Length; iinfo++)
                {
                    var info = infos[iinfo];
                    if (!info.ColType.IsKnownSizeVector)
                        continue;
                    bldr.Reset(info.SizeBase, false);
                    int ivDst = 0;
                    // The following code is similar to the code in ProcessVec.
                    for (int i = 0; i < info.Segments.Length; i++)
                    {
                        var seg = info.Segments[i];
                        Contracts.Assert(!seg.IsVariable);

                        int min = seg.Min;
                        int lim = seg.Lim;
                        int sizeSeg = lim - min;
                        Contracts.Assert(ivDst <= info.SizeBase - sizeSeg);

                        int isrc = header.Indices.FindIndexSorted(0, header.Count, min);
                        if (isrc < header.Count && header.Indices[isrc] < lim)
                        {
                            int indexBase = ivDst - min;
                            int isrcLim = header.Indices.FindIndexSorted(isrc, header.Count, lim);
                            Contracts.Assert(isrc < isrcLim);
                            for (; isrc < isrcLim; isrc++)
                            {
                                var srcCur = header.Indices[isrc];
                                Contracts.Assert(min <= srcCur & srcCur < lim);
                                bldr.AddFeature(indexBase + srcCur, ReadOnlyMemoryUtils.TrimWhiteSpace(header.Spans[isrc]));
                            }
                        }
                        ivDst += sizeSeg;
                    }
                    Contracts.Assert(ivDst == info.SizeBase);
                    if (!bldr.IsEmpty)
                        bldr.GetResult(ref slotNames[iinfo]);
                }
            }

            public RowSet CreateRowSet(ParseStats stats, int count, bool[] active)
            {
                Contracts.Assert(active == null || active.Length == _creator.Length);

                RowSet rows = new RowSet(stats, count, _creator.Length);
                for (int i = 0; i < rows.Pipes.Length; i++)
                {
                    if (active == null || active[i])
                        rows.Pipes[i] = _creator[i](rows);
                }
                return rows;
            }

            /// <summary>
            /// Returns a <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> with trailing whitespace trimmed.
            /// </summary>
            private ReadOnlyMemory<char> TrimEndWhiteSpace(ReadOnlyMemory<char> memory, ReadOnlySpan<char> span)
            {
                if (memory.IsEmpty)
                    return memory;

                int ichLim = memory.Length;
                if (!char.IsWhiteSpace(span[ichLim - 1]))
                    return memory;

                while (0 < ichLim && char.IsWhiteSpace(span[ichLim - 1]))
                    ichLim--;

                return memory.Slice(0, ichLim);
            }

            public void ParseRow(RowSet rows, int irow, Helper helper, bool[] active, string path, long line, string text)
            {
                Contracts.AssertValue(rows);
                Contracts.Assert(irow >= 0);
                Contracts.Assert(helper is HelperImpl);
                Contracts.Assert(active == null | Utils.Size(active) == _infos.Length);

                var impl = (HelperImpl)helper;
                var lineSpan = text.AsMemory();
                var span = lineSpan.Span;
                if ((_flags & Options.TrimWhitespace) != 0)
                    lineSpan = TrimEndWhiteSpace(lineSpan, span);
                try
                {
                    // Parse the spans into items, ensuring that sparse don't precede non-sparse.
                    int srcLim = impl.GatherFields(lineSpan, span, path, line);
                    impl.Fields.AssertValid();

                    // REVIEW: When should we report inconsistency?
                    // VerifyColumnCount(srcLim);

                    ProcessItems(rows, irow, active, impl.Fields, srcLim, line);
                    rows.Stats.LogRow();
                }
                finally
                {
                    impl.Fields.Clear();
                }
            }

            public Helper CreateHelper(ParseStats stats, int srcNeeded)
            {
                Contracts.AssertValue(stats);
                Contracts.Assert(srcNeeded >= 0);
                return new HelperImpl(stats, _flags, _separators, _inputSize, srcNeeded);
            }

            /// <summary>
            /// This is an abstraction containing all the useful stuff for splitting a raw line of text
            /// into a FieldSet. A cursor has one of these that it passes in whenever it wants a line
            /// parsed.
            /// </summary>
            public abstract class Helper
            {
            }

            private sealed class HelperImpl : Helper
            {
                private readonly ParseStats _stats;
                private readonly char[] _seps;
                private readonly char _sep0;
                private readonly char _sep1;
                private readonly bool _sepContainsSpace;
                private readonly int _inputSize;
                private readonly int _srcNeeded;
                private readonly bool _quoting;
                private readonly bool _sparse;
                // This is a working buffer.
                private readonly StringBuilder _sb;

                // Result of a blank field - either Missing or Empty, depending on _quoting.
                private readonly ReadOnlyMemory<char> _blank;

                public readonly FieldSet Fields;

                public HelperImpl(ParseStats stats, Options flags, char[] seps, int inputSize, int srcNeeded)
                {
                    Contracts.AssertValue(stats);
                    // inputSize == 0 means unknown.
                    Contracts.Assert(inputSize >= 0);
                    Contracts.Assert(srcNeeded >= 0);
                    Contracts.Assert(inputSize == 0 || srcNeeded < inputSize);
                    Contracts.AssertNonEmpty(seps);

                    _stats = stats;
                    _seps = seps;
                    _sep0 = _seps[0];
                    _sep1 = _seps.Length > 1 ? _seps[1] : '\0';
                    _sepContainsSpace = IsSep(' ');
                    _inputSize = inputSize;
                    _srcNeeded = srcNeeded;
                    _quoting = (flags & Options.AllowQuoting) != 0;
                    _sparse = (flags & Options.AllowSparse) != 0;
                    _sb = new StringBuilder();
                    _blank = ReadOnlyMemory<char>.Empty;
                    Fields = new FieldSet();
                }

                /// <summary>
                /// Check if the given char is a separator.
                /// </summary>
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                private bool IsSep(char ch)
                {
                    if (ch == _sep0)
                        return true;
                    for (int i = 1; i < _seps.Length; i++)
                    {
                        if (ch == _seps[i])
                            return true;
                    }
                    return false;
                }

                /// <summary>
                /// Process the line of text into fields, stored in the Fields field. Ensures that sparse
                /// don't precede non-sparse. Returns the lim of the src columns.
                /// </summary>
                public int GatherFields(ReadOnlyMemory<char> lineSpan, ReadOnlySpan<char> span, string path = null, long line = 0)
                {
                    Fields.AssertEmpty();

                    ScanInfo scan = new ScanInfo(ref lineSpan, path, line);
                    Contracts.Assert(scan.IchMinBuf <= scan.IchMinNext && scan.IchMinNext <= scan.IchLimBuf);

                    int src = 0;
                    if (!_sparse)
                    {
                        for (; ; )
                        {
                            Contracts.Assert(scan.IchMinBuf <= scan.IchMinNext && scan.IchMinNext <= scan.IchLimBuf);
                            bool more = FetchNextField(ref scan, span);
                            Contracts.Assert(scan.IchMinBuf <= scan.IchMinNext && scan.IchMinNext <= scan.IchLimBuf);
                            Contracts.Assert(scan.Index == -1);

                            if (scan.QuotingError)
                                _stats.LogBadFmt(ref scan, "Illegal quoting");

                            if (!scan.Span.IsEmpty)
                            {
                                Fields.EnsureSpace();
                                Fields.Spans[Fields.Count] = scan.Span;
                                Fields.Indices[Fields.Count++] = src;
                            }
                            if (++src > _srcNeeded || !more)
                                break;
                        }
                        return src;
                    }

                    // Allow sparse. inputSize == 0 means the text specifies the maxDim value.
                    // Note that we always go one past srcNeeded in case what we think is srcNeeded is
                    // actually the sparse length value.
                    int csrcSparse = -1;
                    int indexPrev = -1;
                    int srcLimFixed = -1;
                    int inputSize = _inputSize;
                    int srcNeeded = _srcNeeded;

                    for (; ; )
                    {
                        Contracts.Assert(scan.IchMinBuf <= scan.IchMinNext && scan.IchMinNext <= scan.IchLimBuf);
                        bool more = FetchNextField(ref scan, span);
                        Contracts.Assert(scan.IchMinBuf <= scan.IchMinNext && scan.IchMinNext <= scan.IchLimBuf);
                        Contracts.Assert(scan.Index >= -1);

                        if (scan.QuotingError)
                            _stats.LogBadFmt(ref scan, "Illegal quoting");

                        if (scan.Index < 0)
                        {
                            // Not sparse.
                            if (srcLimFixed >= 0)
                            {
                                _stats.LogBadFmt(ref scan, "Non-sparse formatted value follows sparse formatted value");
                                break;
                            }
                            if (!scan.Span.IsEmpty)
                            {
                                Fields.EnsureSpace();
                                Fields.Spans[Fields.Count] = scan.Span;
                                Fields.Indices[Fields.Count++] = src;
                            }
                            if (src++ > srcNeeded || !more)
                                break;
                            continue;
                        }

                        if (srcLimFixed < 0)
                        {
                            // First sparse item.
                            Contracts.Assert(src - 1 <= srcNeeded);

                            if (inputSize == 0)
                            {
                                // Dimensionality (number of sparse slots) is embedded in the text as the last src value.
                                if (Fields.Count <= 0)
                                {
                                    _stats.LogBadFmt(ref scan, "Missing dimensionality or ambiguous sparse item. Use sparse=- for non-sparse file, and/or quote the value.");
                                    break;
                                }
                                if (Fields.Indices[Fields.Count - 1] != src - 1)
                                {
                                    _stats.LogBadFmt(ref scan, "Missing dimensionality or ambiguous sparse item. Use sparse=- for non-sparse file, and/or quote the value.");
                                    break;
                                }
                                var spanT = Fields.Spans[Fields.Count - 1];

                                // Note that Convert throws exception the text is unparsable.
                                int csrc = default;
                                try
                                {
                                    Conversions.Instance.Convert(ref spanT, ref csrc);
                                }
                                catch
                                {
                                    Contracts.Assert(csrc == default);
                                }

                                if (csrc <= 0)
                                {
                                    _stats.LogBadFmt(ref scan, "Bad dimensionality or ambiguous sparse item. Use sparse=- for non-sparse file, and/or quote the value.");
                                    break;
                                }

                                csrcSparse = csrc;
                                srcLimFixed = Fields.Indices[--Fields.Count];
                                if (csrcSparse >= SrcLim - srcLimFixed)
                                    csrcSparse = SrcLim - srcLimFixed - 1;
                                inputSize = srcLimFixed + csrcSparse;
                                if (srcNeeded >= inputSize)
                                    srcNeeded = inputSize - 1;
                            }
                            else
                            {
                                Contracts.Assert(srcNeeded < inputSize);
                                srcLimFixed = Fields.Count;
                                csrcSparse = inputSize - Fields.Count;
                            }

                            // Retire "src" - it is no longer used once we get to sparse land.
                            src = -1;
                        }

                        // If it's past what we need, stop. Note that this means we require sorted indices!
                        // Note that this test protects against overflow - if we first computed dsrc + srcLimFixed
                        // we would risk overflowing.
                        if (scan.Index > srcNeeded - srcLimFixed)
                        {
                            // If the scan.Index value is bigger than expected, warn the user.
                            if (scan.Index >= csrcSparse)
                            {
                                _stats.LogBadFmt(ref scan, "Sparse item index larger than expected. Is the specified size incorrect?");
                                break;
                            }
                            if (scan.Index > srcNeeded - srcLimFixed + 1)
                                break;
                        }

                        if (indexPrev >= scan.Index)
                        {
                            _stats.LogBadFmt(ref scan, "Sparse indices out of order");
                            break;
                        }
                        indexPrev = scan.Index;

                        // Don't record empties.
                        if (!scan.Span.IsEmpty)
                        {
                            Fields.EnsureSpace();
                            Fields.Spans[Fields.Count] = scan.Span;
                            Fields.Indices[Fields.Count++] = srcLimFixed + scan.Index;
                        }
                        if (!more)
                            break;
                    }

                    if (srcLimFixed < 0)
                    {
                        // Dense
                        return Math.Max(src, inputSize);
                    }

                    // Sparse
                    Contracts.Assert(inputSize > 0);
                    Contracts.Assert(csrcSparse == inputSize - srcLimFixed);
                    return inputSize;
                }

                private bool FetchNextField(ref ScanInfo scan, ReadOnlySpan<char> span)
                {
                    Contracts.Assert(scan.IchMinBuf <= scan.IchMinNext && scan.IchMinNext <= scan.IchLimBuf);

                    var text = scan.TextBuf;
                    int ichLim = scan.IchLimBuf;
                    int ichCur = scan.IchMinNext;
                    if (!_sepContainsSpace)
                    {
                        // Ignore leading spaces
                        while (ichCur < ichLim && span[ichCur] == ' ')
                            ichCur++;
                    }

                    // Initialize the ParseField.
                    scan.QuotingError = false;
                    scan.Index = -1;
                    scan.IchMin = ichCur;

                    if (ichCur >= ichLim)
                    {
                        scan.IchMinNext = scan.IchLim = ichCur;
                        scan.Span = _blank;
                        return false;
                    }

                    int ichMinRaw = ichCur;
                    if (_sparse && (uint)(span[ichCur] - '0') <= 9)
                    {
                        // See if it is sparse. Avoid overflow by limiting the index to 9 digits.
                        // REVIEW: This limits the src index to a billion. Is this acceptable?
                        int ichEnd = Math.Min(ichLim, ichCur + 9);
                        int ichCol = ichCur + 1;
                        Contracts.Assert(ichCol <= ichEnd);
                        while (ichCol < ichEnd && (uint)(span[ichCol] - '0') <= 9)
                            ichCol++;

                        if (ichCol < ichLim && span[ichCol] == ':')
                        {
                            // It is sparse. Compute the index.
                            int ind = 0;
                            for (int ich = ichCur; ich < ichCol; ich++)
                                ind = ind * 10 + (span[ich] - '0');
                            ichCur = ichCol + 1;
                            scan.Index = ind;

                            // Skip spaces again.
                            if (!_sepContainsSpace)
                            {
                                while (ichCur < ichLim && span[ichCur] == ' ')
                                    ichCur++;
                            }

                            if (ichCur >= ichLim)
                            {
                                scan.IchMinNext = scan.IchLim = ichCur;
                                scan.Span = _blank;
                                return false;
                            }
                        }
                    }

                    Contracts.Assert(ichCur < ichLim);
                    if (span[ichCur] == '"' && _quoting)
                    {
                        // Quoted case.
                        ichCur++;
                        _sb.Clear();
                        int ichRun = ichCur;
                        for (; ; ichCur++)
                        {
                            Contracts.Assert(ichCur <= ichLim);
                            if (ichCur >= ichLim)
                            {
                                // Missing close quote!
                                scan.QuotingError = true;
                                break;
                            }
                            if (span[ichCur] == '"')
                            {
                                if (ichCur > ichRun)
                                    _sb.AppendSpan(span.Slice(ichRun, ichCur - ichRun));
                                if (++ichCur >= ichLim)
                                    break;
                                if (span[ichCur] != '"')
                                    break;
                                ichRun = ichCur;
                            }
                        }

                        // Ignore any spaces between here and the next separator. Anything else is a formatting "error".
                        for (; ichCur < ichLim; ichCur++)
                        {
                            if (span[ichCur] == ' ')
                            {
                                // End the loop if space is a sep, otherwise ignore this space.
                                if (_sepContainsSpace)
                                    break;
                            }
                            else
                            {
                                // End the loop if this nonspace char is a sep, otherwise it is an error.
                                if (IsSep(span[ichCur]))
                                    break;
                                scan.QuotingError = true;
                            }
                        }

                        if (scan.QuotingError || _sb.Length == 0)
                            scan.Span = String.Empty.AsMemory();
                        else
                            scan.Span = _sb.ToString().AsMemory();
                    }
                    else
                    {
                        int ichMin = ichCur;

                        // Please note that these branched tight loops are intended and performance critical.
                        if (_seps.Length == 1)
                        {
                            for (; ; ichCur++)
                            {
                                Contracts.Assert(ichCur <= ichLim);
                                if (ichCur >= ichLim)
                                    break;
                                if (_sep0 == span[ichCur])
                                    break;
                            }
                        }
                        else if (_seps.Length == 2)
                        {
                            for (; ; ichCur++)
                            {
                                Contracts.Assert(ichCur <= ichLim);
                                if (ichCur >= ichLim)
                                    break;
                                if (_sep0 == span[ichCur] || _sep1 == span[ichCur])
                                    break;
                            }
                        }
                        else
                        {
                            for (; ; ichCur++)
                            {
                                Contracts.Assert(ichCur <= ichLim);
                                if (ichCur >= ichLim)
                                    break;
                                if (IsSep(span[ichCur]))
                                    break;
                            }
                        }

                        if (ichMin >= ichCur)
                            scan.Span = _blank;
                        else
                            scan.Span = text.Slice(ichMin, ichCur - ichMin);
                    }

                    scan.IchLim = ichCur;
                    if (ichCur >= ichLim)
                    {
                        scan.IchMinNext = ichCur;
                        return false;
                    }

                    Contracts.Assert(_seps.Contains(span[ichCur]));
                    scan.IchMinNext = ichCur + 1;
                    return true;
                }
            }

            private void ProcessItems(RowSet rows, int irow, bool[] active, FieldSet fields, int srcLim, long line)
            {
                Contracts.Assert(active == null | Utils.Size(active) == _infos.Length);
                fields.AssertValid();

                Contracts.Assert(0 <= irow && irow < rows.Count);
                for (int iinfo = 0; iinfo < _infos.Length; iinfo++)
                {
                    if (active != null && !active[iinfo])
                        continue;

                    var info = _infos[iinfo];
                    var v = rows.Pipes[iinfo];
                    Contracts.Assert(v != null);

                    if (!info.ColType.IsVector)
                        ProcessOne(fields, info, v, irow, line);
                    else
                        ProcessVec(srcLim, fields, info, v, irow, line);
                }
            }

            private void ProcessVec(int srcLim, FieldSet fields, ColInfo info, ColumnPipe v, int irow, long line)
            {
                Contracts.Assert(srcLim >= 0);
                Contracts.Assert(info.ColType.IsVector);
                Contracts.Assert(info.SizeBase > 0 || info.IsegVariable >= 0);

                int sizeVar = 0;
                if (info.IsegVariable >= 0)
                {
                    // There is a variable segment. Compute the total size.
                    var seg = info.Segments[info.IsegVariable];
                    if (seg.Min < srcLim)
                        sizeVar = srcLim - seg.Min;
                }
                Contracts.Assert(sizeVar >= 0);
                int size = checked(info.SizeBase + sizeVar);

                v.Reset(irow, size);
                int ivDst = 0;
                for (int i = 0; i < info.Segments.Length; i++)
                {
                    var seg = info.Segments[i];
                    Contracts.Assert(seg.IsVariable == (i == info.IsegVariable));

                    int min = seg.Min;
                    int lim = seg.Lim;
                    if (i == info.IsegVariable)
                    {
                        lim = srcLim;
                        Contracts.Assert(lim == min + sizeVar);
                    }
                    int sizeSeg = lim - min;
                    Contracts.Assert(ivDst <= size - sizeSeg);

                    int isrc = fields.Indices.FindIndexSorted(0, fields.Count, min);
                    if (isrc < fields.Count && fields.Indices[isrc] < lim)
                    {
                        int indexBase = ivDst - min;
                        int isrcLim = fields.Indices.FindIndexSorted(isrc, fields.Count, lim);
                        Contracts.Assert(isrc < isrcLim);
                        for (; isrc < isrcLim; isrc++)
                        {
                            var srcCur = fields.Indices[isrc];
                            Contracts.Assert(min <= srcCur & srcCur < lim);
                            if (!v.Consume(irow, indexBase + srcCur, ref fields.Spans[isrc]))
                                v.Rows.Stats.LogBadValue(line, info.Name, indexBase + srcCur);
                        }
                    }
                    ivDst += sizeSeg;
                }
                Contracts.Assert(ivDst == size);
            }

            private void ProcessOne(FieldSet vs, ColInfo info, ColumnPipe v, int irow, long line)
            {
                Contracts.Assert(!info.ColType.IsVector);
                Contracts.Assert(Utils.Size(info.Segments) == 1);
                Contracts.Assert(info.Segments[0].Lim == info.Segments[0].Min + 1);

                int src = info.Segments[0].Min;
                int isrc = vs.Indices.FindIndexSorted(0, vs.Count, src);
                if (isrc < vs.Count && vs.Indices[isrc] == src)
                {
                    if (!v.Consume(irow, 0, ref vs.Spans[isrc]))
                        v.Rows.Stats.LogBadValue(line, info.Name);
                }
                else
                    v.Reset(irow, 0);
            }

            // This checks for an inconsistent number of features.
            private void VerifyColumnCount(int csrc)
            {
                if (csrc == _csrc)
                    return;

                Interlocked.CompareExchange(ref _csrc, csrc, 0);

                if (csrc == _csrc)
                    return;

                if (Interlocked.Increment(ref _mismatchCount) == 1)
                    Console.WriteLine("Warning: Feature count mismatch: {0} vs {1}", csrc, _csrc);
            }
        }
    }
}
