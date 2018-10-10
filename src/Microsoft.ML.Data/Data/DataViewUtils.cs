// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    public static class DataViewUtils
    {
        /// <summary>
        /// Generate a unique temporary column name for the given schema.
        /// Use tag to independently create multiple temporary, unique column
        /// names for a single transform.
        /// </summary>
        public static string GetTempColumnName(this ISchema schema, string tag = null)
        {
            Contracts.CheckValue(schema, nameof(schema));

            int col;
            if (!string.IsNullOrWhiteSpace(tag) && !schema.TryGetColumnIndex(tag, out col))
                return tag;

            for (int i = 0; ; i++)
            {
                string name = string.IsNullOrWhiteSpace(tag) ?
                    string.Format("temp_{0:000}", i) :
                    string.Format("temp_{0}_{1:000}", tag, i);

                if (!schema.TryGetColumnIndex(name, out col))
                    return name;
            }
        }

        /// <summary>
        /// Generate n unique temporary column names for the given schema.
        /// Use tag to independently create multiple temporary, unique column
        /// names for a single transform.
        /// </summary>
        public static string[] GetTempColumnNames(this ISchema schema, int n, string tag = null)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Check(n > 0, "n");

            var res = new string[n];
            int j = 0;
            for (int i = 0; i < n; i++)
            {
                for (;;)
                {
                    string name = string.IsNullOrWhiteSpace(tag) ?
                        string.Format("temp_{0:000}", j) :
                        string.Format("temp_{0}_{1:000}", tag, j);
                    j++;
                    int col;
                    if (!schema.TryGetColumnIndex(name, out col))
                    {
                        res[i] = name;
                        break;
                    }
                }
            }
            return res;
        }

        /// <summary>
        /// Get the row count from the input view by any means necessary, even explicit enumeration
        /// and counting if <see cref="IDataView.GetRowCount"/> insists on returning <c>null</c>.
        /// </summary>
        public static long ComputeRowCount(IDataView view)
        {
            long? countNullable = view.GetRowCount(lazy: false);
            if (countNullable != null)
                return countNullable.Value;
            long count = 0;
            using (var cursor = view.GetRowCursor(col => false))
            {
                while (cursor.MoveNext())
                    count++;
            }
            return count;
        }

        /// <summary>
        /// Get the target number of threads to use, given a host and another indicator of thread count.
        /// When num > 0, this uses num limited to twice what the host says. Otherwise, if preferOne
        /// is true, it returns 1. Otherwise, it returns what the host says.
        /// </summary>
        public static int GetThreadCount(IHost host, int num = 0, bool preferOne = false)
        {
            Contracts.CheckValue(host, nameof(host));

            int conc = host.ConcurrencyFactor;
            if (conc <= 0)
                conc = Math.Max(2, Environment.ProcessorCount - 1);

            if (num > 0)
                return Math.Min(num, 2 * conc);
            if (preferOne)
                return 1;
            return conc;
        }

        /// <summary>
        /// Try to create a cursor set from upstream and consolidate it here. The host determines
        /// the target cardinality of the cursor set.
        /// </summary>
        public static bool TryCreateConsolidatingCursor(out IRowCursor curs,
            IDataView view, Func<int, bool> predicate, IHost host, IRandom rand)
        {
            Contracts.CheckValue(host, nameof(host));
            host.CheckValue(view, nameof(view));
            host.CheckValue(predicate, nameof(predicate));

            int cthd = GetThreadCount(host);
            host.Assert(cthd > 0);
            if (cthd == 1 || !AllCachable(view.Schema, predicate))
            {
                curs = null;
                return false;
            }

            IRowCursorConsolidator consolidator;
            var inputs = view.GetRowCursorSet(out consolidator, predicate, cthd, rand);
            host.Check(Utils.Size(inputs) > 0);
            host.Check(inputs.Length == 1 || consolidator != null);

            if (inputs.Length == 1)
                curs = inputs[0];
            else
                curs = consolidator.CreateCursor(host, inputs);
            return true;
        }

        /// <summary>
        /// From the given input cursor, split it into a cursor set with the given
        /// cardinality. If not all the active columns are cachable, this will only
        /// produce the given input cursor.
        /// </summary>
        public static IRowCursor[] CreateSplitCursors(out IRowCursorConsolidator consolidator,
            IChannelProvider provider, IRowCursor input, int num)
        {
            Contracts.CheckValue(provider, nameof(provider));
            provider.CheckValue(input, nameof(input));

            consolidator = null;
            if (num <= 1)
                return new IRowCursor[1] { input };

            // If any active columns are not cachable, we can't split.
            if (!AllCachable(input.Schema, input.IsColumnActive))
                return new IRowCursor[1] { input };

            // REVIEW: Should we limit the cardinality to some reasonable size?

            // REVIEW: Ideally a splitter should be owned by a data view
            // we might split, so that we can share the cache pools among multiple
            // cursors.

            // REVIEW: Keep the utility method here, move this splitter stuff
            // to some other file.
            return Splitter.Split(out consolidator, provider, input.Schema, input, num);
        }

        /// <summary>
        /// Return whether all the active columns, as determined by the predicate, are
        /// cachable - either primitive types or vector types.
        /// </summary>
        public static bool AllCachable(ISchema schema, Func<int, bool> predicate)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckValue(predicate, nameof(predicate));

            for (int col = 0; col < schema.ColumnCount; col++)
            {
                if (!predicate(col))
                    continue;
                var type = schema.GetColumnType(col);
                if (!IsCachable(type))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Determine whether the given type is cachable - either a primitive type or a vector type.
        /// </summary>
        public static bool IsCachable(this ColumnType type)
        {
            return type != null && (type.IsPrimitive || type.IsVector);
        }

        /// <summary>
        /// Tests whether the cursors are mutually compatible for consolidation,
        /// that is, they all are non-null, have the same schemas, and the same
        /// set of columns are active.
        /// </summary>
        public static bool SameSchemaAndActivity(IRowCursor[] cursors)
        {
            // There must be something to actually consolidate.
            if (Utils.Size(cursors) == 0)
                return true;
            var firstCursor = cursors[0];
            if (firstCursor == null)
                return false;
            if (cursors.Length == 1)
                return true;
            var schema = firstCursor.Schema;
            // All cursors must have the same schema.
            for (int i = 1; i < cursors.Length; ++i)
            {
                if (cursors[i] == null || cursors[i].Schema != schema)
                    return false;
            }
            // All cursors must have the same columns active.
            for (int c = 0; c < schema.ColumnCount; ++c)
            {
                bool active = firstCursor.IsColumnActive(c);
                for (int i = 1; i < cursors.Length; ++i)
                {
                    if (cursors[i].IsColumnActive(c) != active)
                        return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Given a parallel cursor set, this consolidates them into a single cursor. The batchSize
        /// is a hint used for efficiency.
        /// </summary>
        public static IRowCursor ConsolidateGeneric(IChannelProvider provider, IRowCursor[] inputs, int batchSize)
        {
            Contracts.CheckValue(provider, nameof(provider));
            provider.CheckNonEmpty(inputs, nameof(inputs));
            provider.CheckParam(batchSize >= 0, nameof(batchSize));

            if (inputs.Length == 1)
                return inputs[0];

            object[] pools = null;
            return Splitter.Consolidator.Consolidate(provider, inputs, batchSize, ref pools);
        }

        /// <summary>
        /// A convenience class to facilitate the creation of a split, as well as a convenient
        /// place to store shared resources that can be reused among multiple splits of a cursor
        /// with the same schema. Since splitting also returns a consolidator, this also contains
        /// a consolidating logic.
        ///
        /// In a very rough sense, both the splitters and consolidators are written in the same way:
        /// For all input cursors, and all active columns, an "in pipe" is created. A worker thread
        /// per input cursor busily retrieves values from the cursors and stores them in the "in
        /// pipe." At appropriate times, "batch" objects are synthesized from the inputs consumed
        /// thusfar, and inserted into a blocking collection. The output cursor or cursors likewise
        /// have a set of "out pipe" instances, one per each of the active columns, through which
        /// successive batches are presented for consumption by the user of the output cursors. Of
        /// course, both split and consolidate have many details from which they differ, for example, the
        /// consolidator must accept batches as they come and reconcile them among multiple inputs,
        /// while the splitter is more free.
        ///
        /// It is ideal if a data view that could be split retains one of these objects itself,
        /// so that multiple splittings will have the capability of sharing buffers from cursoring
        /// to cursoring, but this is not required.
        /// </summary>
        private sealed class Splitter
        {
            private readonly ISchema _schema;
            private readonly object[] _cachePools;
            private object[] _consolidateCachePools;

            /// <summary>
            /// Pipes, in addition to column values, will also communicate extra information
            /// enumerated within this. This enum serves the purpose of providing nice readable
            /// indices to these "extra" information in pipes.
            /// </summary>
            private enum ExtraIndex
            {
                Id,
#pragma warning disable MSML_GeneralName // Allow for this private enum.
                _Lim
#pragma warning restore MSML_GeneralName
            }

            private Splitter(ISchema schema)
            {
                Contracts.AssertValue(schema);
                _schema = schema;
                _cachePools = new object[_schema.ColumnCount + (int)ExtraIndex._Lim];
            }

            public sealed class Consolidator : IRowCursorConsolidator
            {
                private readonly Splitter _splitter;

                public Consolidator(Splitter splitter)
                {
                    Contracts.AssertValue(splitter);
                    _splitter = splitter;
                }

                public IRowCursor CreateCursor(IChannelProvider provider, IRowCursor[] inputs)
                {
                    return Consolidate(provider, inputs, 128, ref _splitter._consolidateCachePools);
                }

                public static IRowCursor Consolidate(IChannelProvider provider, IRowCursor[] inputs, int batchSize, ref object[] ourPools)
                {
                    Contracts.AssertValue(provider);
                    using (var ch = provider.Start("Consolidate"))
                    {
                        return ConsolidateCore(provider, inputs, ref ourPools, ch);
                    }
                }

                private static IRowCursor ConsolidateCore(IChannelProvider provider, IRowCursor[] inputs, ref object[] ourPools, IChannel ch)
                {
                    ch.CheckNonEmpty(inputs, nameof(inputs));
                    if (inputs.Length == 1)
                        return inputs[0];
                    ch.CheckParam(SameSchemaAndActivity(inputs), nameof(inputs), "Inputs not compatible for consolidation");

                    IRowCursor cursor = inputs[0];
                    ISchema schema = cursor.Schema;
                    ch.CheckParam(AllCachable(schema, cursor.IsColumnActive), nameof(inputs), "Inputs had some uncachable input columns");

                    int[] activeToCol;
                    int[] colToActive;
                    Utils.BuildSubsetMaps(schema.ColumnCount, cursor.IsColumnActive, out activeToCol, out colToActive);

                    // Because the schema of the consolidator is not necessary fixed, we are merely
                    // opportunistic about buffer sharing, from cursoring to cursoring. If we can do
                    // it easily, great, if not, no big deal.
                    if (Utils.Size(ourPools) != schema.ColumnCount)
                        ourPools = new object[schema.ColumnCount + (int)ExtraIndex._Lim];
                    // Create the out pipes.
                    OutPipe[] outPipes = new OutPipe[activeToCol.Length + (int)ExtraIndex._Lim];
                    for (int i = 0; i < activeToCol.Length; ++i)
                    {
                        int c = activeToCol[i];
                        ColumnType type = schema.GetColumnType(c);
                        var pool = GetPool(type, ourPools, c);
                        outPipes[i] = OutPipe.Create(type, pool);
                    }
                    int idIdx = activeToCol.Length + (int)ExtraIndex.Id;
                    outPipes[idIdx] = OutPipe.Create(NumberType.UG, GetPool(NumberType.UG, ourPools, idIdx));

                    // Create the structures to synchronize between the workers and the consumer.
                    const int toConsumeBound = 4;
                    var toConsume = new BlockingCollection<Batch>(toConsumeBound);
                    var batchColumnPool = new MadeObjectPool<BatchColumn[]>(() => new BatchColumn[outPipes.Length]);
                    Thread[] workers = new Thread[inputs.Length];
                    MinWaiter waiter = new MinWaiter(workers.Length);
                    bool done = false;

                    for (int t = 0; t < workers.Length; ++t)
                    {
                        var localCursor = inputs[t];
                        ch.Assert(localCursor.State == CursorState.NotStarted);
                        // Note that these all take ownership of their respective cursors,
                        // so they all handle their disposal internal to the thread.
                        workers[t] = Utils.CreateBackgroundThread(() =>
                        {
                            // This will be the last batch sent in the finally. If iteration procedes without
                            // error, it will remain null, and be sent as a sentinel. If iteration results in
                            // an exception that we catch, the exception catching block will set this to an
                            // exception bearing block, and that will be passed along as the last block instead.
                            Batch lastBatch = null;
                            try
                            {
                                using (localCursor)
                                {
                                    InPipe[] inPipes = new InPipe[outPipes.Length];
                                    for (int i = 0; i < activeToCol.Length; ++i)
                                        inPipes[i] = outPipes[i].CreateInPipe(RowCursorUtils.GetGetterAsDelegate(localCursor, activeToCol[i]));
                                    inPipes[idIdx] = outPipes[idIdx].CreateInPipe(localCursor.GetIdGetter());

                                    long oldBatch = 0;
                                    int count = 0;
                                    // This event is used to synchronize ourselves using a MinWaiter
                                    // so that we add batches to the consumer queue at the appropriate time.
                                    ManualResetEventSlim waiterEvent = null;

                                    Action pushBatch = () =>
                                    {
                                        if (count > 0)
                                        {
                                            var batchColumns = batchColumnPool.Get();
                                            for (int i = 0; i < inPipes.Length; ++i)
                                                batchColumns[i] = inPipes[i].GetBatchColumnAndReset();
                                            // REVIEW: Is it worth not allocating new Batch object for each batch?
                                            var batch = new Batch(batchColumnPool, batchColumns, count, oldBatch);
                                            count = 0;
                                            // The waiter event should never be null since this is only
                                            // called after a point where waiter.Register has been called.
                                            ch.AssertValue(waiterEvent);
                                            waiterEvent.Wait();
                                            waiterEvent = null;
                                            toConsume.Add(batch);
                                        }
                                    };
                                    // Handle the first one separately, then go into the main loop.
                                    if (localCursor.MoveNext() && !done)
                                    {
                                        oldBatch = localCursor.Batch;
                                        foreach (var pipe in inPipes)
                                            pipe.Fill();
                                        count++;
                                        // Register with the min waiter that we want to wait on this batch number.
                                        waiterEvent = waiter.Register(oldBatch);

                                        while (localCursor.MoveNext() && !done)
                                        {
                                            if (oldBatch != localCursor.Batch)
                                            {
                                                ch.Assert(count == 0 || localCursor.Batch > oldBatch);
                                                pushBatch();
                                                oldBatch = localCursor.Batch;
                                                waiterEvent = waiter.Register(oldBatch);
                                            }
                                            foreach (var pipe in inPipes)
                                                pipe.Fill();
                                            count++;
                                        }
                                        pushBatch();
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                // Whoops, we won't be sending null as the sentinel now.
                                lastBatch = new Batch(ex);
                                toConsume.Add(new Batch(ex));
                            }
                            finally
                            {
                                if (waiter.Retire() == 0)
                                {
                                    if (lastBatch == null)
                                    {
                                        // If it wasn't null, this already sent along an exception bearing batch, in which
                                        // case sending the sentinel is unnecessary and unhelpful.
                                        toConsume.Add(null);
                                    }
                                    toConsume.CompleteAdding();
                                }
                            }
                        });
                        workers[t].Start();
                    }

                    Action quitAction = () =>
                    {
                        done = true;
                        var myOutPipes = outPipes;
                        foreach (var batch in toConsume.GetConsumingEnumerable())
                        {
                            if (batch == null)
                                continue;
                            batch.SetAll(myOutPipes);
                            foreach (var outPipe in myOutPipes)
                                outPipe.Unset();
                        }
                        foreach (Thread thread in workers)
                            thread.Join();
                    };

                    return new Cursor(provider, schema, activeToCol, colToActive, outPipes, toConsume, quitAction);
                }

                private static object GetPool(ColumnType type, object[] pools, int poolIdx)
                {
                    Func<object[], int, object> func = GetPoolCore<int>;
                    var method = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.RawType);
                    return method.Invoke(null, new object[] { pools, poolIdx });
                }

                private static MadeObjectPool<T[]> GetPoolCore<T>(object[] pools, int poolIdx)
                {
                    var pool = pools[poolIdx] as MadeObjectPool<T[]>;
                    if (pool == null)
                        pools[poolIdx] = pool = new MadeObjectPool<T[]>(() => null);
                    return pool;
                }
            }

            public static IRowCursor[] Split(out IRowCursorConsolidator consolidator, IChannelProvider provider, ISchema schema, IRowCursor input, int cthd)
            {
                Contracts.AssertValue(provider, "provider");

                var splitter = new Splitter(schema);
                using (var ch = provider.Start("CursorSplitter"))
                {
                    var result = splitter.SplitCore(out consolidator, provider, input, cthd);
                    return result;
                }
            }

            private IRowCursor[] SplitCore(out IRowCursorConsolidator consolidator, IChannelProvider ch, IRowCursor input, int cthd)
            {
                Contracts.AssertValue(ch);
                ch.AssertValue(input);
                ch.Assert(input.Schema == _schema);
                ch.Assert(cthd >= 2);
                ch.Assert(AllCachable(_schema, input.IsColumnActive));

                // REVIEW: Should the following be configurable?
                // How would we even expose these sorts of parameters to a user?
                const int maxBatchCount = 128;
                const int toConsumeBound = 4;

                // Create the mappings between active column index, and column index.
                int[] activeToCol;
                int[] colToActive;
                Utils.BuildSubsetMaps(_schema.ColumnCount, input.IsColumnActive, out activeToCol, out colToActive);

                Func<IRowCursor, int, InPipe> createFunc = CreateInPipe<int>;
                var inGenMethod = createFunc.GetMethodInfo().GetGenericMethodDefinition();
                object[] arguments = new object[] { input, 0 };
                // Only one set of in-pipes, one per column, as well as for extra side information.
                InPipe[] inPipes = new InPipe[activeToCol.Length + (int)ExtraIndex._Lim];
                // There are as many sets of out pipes as there are output cursors.
                OutPipe[][] outPipes = new OutPipe[cthd][];
                for (int i = 0; i < cthd; ++i)
                    outPipes[i] = new OutPipe[inPipes.Length];

                // For each column, create the InPipe, and all OutPipes per output cursor.
                for (int c = 0; c < activeToCol.Length; ++c)
                {
                    ch.Assert(0 <= activeToCol[c] && activeToCol[c] < _schema.ColumnCount);
                    ch.Assert(c == 0 || activeToCol[c - 1] < activeToCol[c]);
                    ch.Assert(input.IsColumnActive(activeToCol[c]));
                    var type = input.Schema.GetColumnType(activeToCol[c]);
                    ch.Assert(type.IsCachable());
                    arguments[1] = activeToCol[c];
                    var inPipe = inPipes[c] =
                        (InPipe)inGenMethod.MakeGenericMethod(type.RawType).Invoke(this, arguments);
                    for (int i = 0; i < cthd; ++i)
                        outPipes[i][c] = inPipe.CreateOutPipe(type);
                }
                // Beyond the InPipes corresponding to column values, we have extra side info pipes.
                int idIdx = activeToCol.Length + (int)ExtraIndex.Id;
                inPipes[idIdx] = CreateIdInPipe(input);
                for (int i = 0; i < cthd; ++i)
                    outPipes[i][idIdx] = inPipes[idIdx].CreateOutPipe(NumberType.UG);

                var toConsume = new BlockingCollection<Batch>(toConsumeBound);
                var batchColumnPool = new MadeObjectPool<BatchColumn[]>(() => new BatchColumn[inPipes.Length]);
                bool done = false;
                int outputsRunning = cthd;

                // Set up and start the thread that consumes the input, and utilizes the InPipe
                // instances to compose the Batch objects. The thread takes ownership of the
                // cursor, and so handles its disposal.
                Thread thread = Utils.CreateBackgroundThread(
                    () =>
                    {
                        Batch lastBatch = null;
                        try
                        {
                            using (input)
                            {
                                long batchId = 0;
                                int count = 0;
                                Action pushBatch = () =>
                                {
                                    var batchColumns = batchColumnPool.Get();
                                    for (int c = 0; c < inPipes.Length; ++c)
                                        batchColumns[c] = inPipes[c].GetBatchColumnAndReset();
                                    // REVIEW: Is it worth not allocating new Batch object for each batch?
                                    var batch = new Batch(batchColumnPool, batchColumns, count, batchId++);
                                    count = 0;
                                    toConsume.Add(batch);
                                };

                                while (input.MoveNext() && !done)
                                {
                                    foreach (var pipe in inPipes)
                                        pipe.Fill();
                                    if (++count >= maxBatchCount)
                                        pushBatch();
                                }
                                if (count > 0)
                                    pushBatch();
                            }
                        }
                        catch (Exception ex)
                        {
                            lastBatch = new Batch(ex);
                        }
                        finally
                        {
                            // The last batch might be an exception, as in the above case. We pass along the exception
                            // bearing batch as the first of the last batches, so that the first worker to encounter this
                            // will know to throw. The remaining get the regular "stop working" null sentinel.
                            toConsume.Add(lastBatch);
                            for (int i = 1; i < cthd; ++i)
                                toConsume.Add(null);
                            toConsume.CompleteAdding();
                        }
                    });
                thread.Start();

                Action quitAction = () =>
                {
                    int remaining = Interlocked.Decrement(ref outputsRunning);
                    ch.Assert(remaining >= 0);
                    if (remaining == 0)
                    {
                        done = true;
                        // A relatively quick and convenient way to dispose of batches, is to use
                        // set/unset on some output pipes repeatedly. Since all have been disposed,
                        // we may as well use the first set of output pipes.
                        var myOutPipes = outPipes[0];
                        foreach (var batch in toConsume.GetConsumingEnumerable())
                        {
                            if (batch == null)
                                continue;
                            batch.SetAll(myOutPipes);
                            foreach (var outPipe in myOutPipes)
                                outPipe.Unset();
                        }
                        thread.Join();
                    }
                };

                var cursors = new Cursor[cthd];
                for (int i = 0; i < cthd; ++i)
                    cursors[i] = new Cursor(ch, _schema, activeToCol, colToActive, outPipes[i], toConsume, quitAction);
                consolidator = new Consolidator(this);
                return cursors;
            }

            /// <summary>
            /// An in pipe creator intended to be used from the splitter only.
            /// </summary>
            private InPipe CreateInPipe<T>(IRow input, int col)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= col && col < _schema.ColumnCount);
                return CreateInPipeCore(col, input.GetGetter<T>(col));
            }

            /// <summary>
            /// An in pipe creator intended to be used from the splitter only.
            /// </summary>
            private InPipe CreateIdInPipe(IRow input)
            {
                Contracts.AssertValue(input);
                return CreateInPipeCore(_schema.ColumnCount + (int)ExtraIndex.Id, input.GetIdGetter());
            }

            private InPipe CreateInPipeCore<T>(int poolIdx, ValueGetter<T> getter)
            {
                Contracts.Assert(0 <= poolIdx && poolIdx < _cachePools.Length);
                Contracts.AssertValue(getter);
                var pool = (MadeObjectPool<T[]>)_cachePools[poolIdx];
                if (pool == null)
                {
                    // REVIEW: If we changed our InPipe behavior so that it only worked over a maximum size
                    // in all scenarios, both during splitting and consolidating, it would be possible to set this
                    // to be of fixed size so we don't have to do reallocation.
                    Interlocked.CompareExchange(ref _cachePools[poolIdx], new MadeObjectPool<T[]>(() => null), null);
                    pool = (MadeObjectPool<T[]>)_cachePools[poolIdx];
                }
                return InPipe.Create(pool, getter);
            }

            /// <summary>
            /// There is one of these created per input cursor, per "channel" of information
            /// (necessary channels include values from active columns, as well as additional
            /// side information), in both splitting and consolidating. This is a running buffer
            /// of the input cursor's values. It is used to create <see cref="BatchColumn"/> objects.
            /// </summary>
            private abstract class InPipe
            {
                public int Count { get; protected set; }

                private InPipe()
                {
                }

                public abstract void Fill();

                public abstract BatchColumn GetBatchColumnAndReset();

                public static InPipe Create<T>(MadeObjectPool<T[]> pool, ValueGetter<T> getter)
                {
                    return new Impl<T>(pool, getter);
                }

                /// <summary>
                /// Creates an out pipe corresponding to the in pipe. This is useful for the splitter,
                /// when we are creating an in pipe.
                /// </summary>
                public abstract OutPipe CreateOutPipe(ColumnType type);

                private sealed class Impl<T> : InPipe
                {
                    private readonly MadeObjectPool<T[]> _pool;
                    private readonly ValueGetter<T> _getter;

                    private T[] _values;

                    public Impl(MadeObjectPool<T[]> pool, ValueGetter<T> getter)
                    {
                        Contracts.AssertValue(pool);
                        Contracts.AssertValue(getter);

                        _pool = pool;
                        _getter = getter;
                        _values = _pool.Get();
                        Contracts.AssertValueOrNull(_values);
                    }

                    public override void Fill()
                    {
                        Utils.EnsureSize(ref _values, Count + 1, keepOld: true);
                        _getter(ref _values[Count++]);
                    }

                    public override BatchColumn GetBatchColumnAndReset()
                    {
                        // REVIEW: Is it worth avoiding an allocation of these new BatchColumn objects?
                        var retval = new BatchColumn.Impl<T>(_values, Count);
                        _values = _pool.Get();
                        Count = 0;
                        return retval;
                    }

                    public override OutPipe CreateOutPipe(ColumnType type)
                    {
                        Contracts.AssertValue(type);
                        Contracts.Assert(typeof(T) == type.RawType);
                        return OutPipe.Create(type, _pool);
                    }
                }
            }

            /// <summary>
            /// These are objects continuously created by the <see cref="InPipe"/> to spin off the
            /// values they have collected. They are collected into a <see cref="Batch"/>
            /// object, and eventually one is consumed by an <see cref="OutPipe"/> instance.
            /// </summary>
            private abstract class BatchColumn
            {
                public readonly int Count;

                private BatchColumn(int count)
                {
                    Contracts.Assert(count > 0);
                    Count = count;
                }

                public sealed class Impl<T> : BatchColumn
                {
                    public readonly T[] Values;

                    public Impl(T[] values, int count)
                        : base(count)
                    {
                        Contracts.Assert(Utils.Size(values) >= count);
                        Values = values;
                    }
                }
            }

            /// <summary>
            /// This holds a collection of <see cref="BatchColumn"/> objects, which together hold all
            /// the values from a set of rows from the input cursor. These are produced as needed
            /// by the input cursor reader, and consumed by each of the output cursors.
            ///
            /// This class also serves a secondary role in marshalling exceptions thrown in the workers
            /// producing batches, into the threads consuming these batches.
            /// <see cref="HasException"/> lets us know if this is one of these "special" batches.
            /// If it is, then the <see cref="SetAll"/> method will throw whenever it is called, by the
            /// consumer of the batches.
            /// </summary>
            private sealed class Batch
            {
                private readonly MadeObjectPool<BatchColumn[]> _pool;
                private readonly BatchColumn[] _batchColumns;
                public readonly int Count;
                public readonly long BatchId;

                private readonly Exception _ex;

                public bool HasException { get { return _ex != null; } }

                /// <summary>
                /// Construct a batch object to communicate the <see cref="BatchColumn"/> objects to consumers.
                /// </summary>
                public Batch(MadeObjectPool<BatchColumn[]> pool, BatchColumn[] batchColumns, int count, long batchId)
                {
                    Contracts.AssertValue(pool);
                    Contracts.AssertValue(batchColumns);
                    Contracts.Assert(count > 0);
                    Contracts.Assert(batchId >= 0);
                    Contracts.Assert(batchColumns.All(bc => bc.Count == count));

                    _pool = pool;
                    _batchColumns = batchColumns;
                    Count = count;
                    BatchId = batchId;
                    Contracts.Assert(!HasException);
                }

                /// <summary>
                /// Construct a batch object to communicate that something went wrong. In this case all other fields
                /// will have default values.
                /// </summary>
                public Batch(Exception ex)
                {
                    Contracts.AssertValue(ex);
                    _ex = ex;
                    Contracts.Assert(HasException);
                }

                /// <summary>
                /// Gives all of the batch columns to the output pipes. This should be called only once,
                /// per batch object, because the the batch columns will be returned to the pool.
                ///
                /// If this was an exception bearing batch, that exception will be propagated and thrown
                /// in this.
                /// </summary>
                public void SetAll(OutPipe[] pipes)
                {
                    if (_ex != null)
                        throw Contracts.Except(_ex, "Splitter/consolidator worker encountered exception while consuming source data");
                    Contracts.Assert(Utils.Size(pipes) == _batchColumns.Length);
                    for (int p = 0; p < _batchColumns.Length; ++p)
                    {
                        pipes[p].Set(_batchColumns[p]);
                        _batchColumns[p] = null;
                    }
                    _pool.Return(_batchColumns);
                }
            }

            /// <summary>
            /// This helps a cursor present the results of a <see cref="BatchColumn"/>. Practically its role
            /// really is to just provide a stable delegate for the <see cref="IRow.GetGetter{T}(int)"/>.
            /// There is one of these created per column, per output cursor, i.e., in splitting
            /// there are <c>n</c> of these created per column, and when consolidating only one of these
            /// is created per column.
            /// </summary>
            private abstract class OutPipe
            {
                private int _count;
                private int _index;

                public int Remaining => _count - _index;

                private OutPipe()
                {
                }

                public static OutPipe Create(ColumnType type, object pool)
                {
                    Contracts.AssertValue(type);
                    Contracts.AssertValue(pool);

                    Type pipeType;
                    if (type.IsVector)
                        pipeType = typeof(ImplVec<>).MakeGenericType(type.ItemType.RawType);
                    else
                    {
                        Contracts.Assert(type.IsPrimitive);
                        pipeType = typeof(ImplOne<>).MakeGenericType(type.RawType);
                    }
                    var constructor = pipeType.GetConstructor(new Type[] { typeof(object) });
                    return (OutPipe)constructor.Invoke(new object[] { pool });
                }

                /// <summary>
                /// Creates an in pipe corresponding to this out pipe. Useful for the consolidator,
                /// when we are creating many in pipes from a single out pipe.
                /// </summary>
                public abstract InPipe CreateInPipe(Delegate getter);

                /// <summary>
                /// Sets this <see cref="OutPipe"/> to start presenting the output of a batch column.
                /// Note that this positions the output on the first item, not before the first item,
                /// so it is not necessary to call <see cref="MoveNext"/> to get the first value.
                /// </summary>
                /// <param name="batchCol">The batch column whose values we should start presenting.</param>
                public abstract void Set(BatchColumn batchCol);

                public abstract void Unset();

                public abstract Delegate GetGetter();

                /// <summary>
                /// Moves to the next value. Note that this should be called only when we are certain that
                /// we have a next value to move to, that is, when <see cref="Remaining"/> is non-zero.
                /// </summary>
                public void MoveNext()
                {
                    Contracts.Assert(_index < _count);
                    ++_index;
                }

                private abstract class Impl<T> : OutPipe
                {
                    private readonly MadeObjectPool<T[]> _pool;
                    protected T[] Values;

                    public Impl(object pool)
                    {
                        Contracts.Assert(pool is MadeObjectPool<T[]>);
                        _pool = (MadeObjectPool<T[]>)pool;
                    }

                    public override InPipe CreateInPipe(Delegate getter)
                    {
                        Contracts.AssertValue(getter);
                        Contracts.Assert(getter is ValueGetter<T>);
                        return InPipe.Create<T>(_pool, (ValueGetter<T>)getter);
                    }

                    public override void Set(BatchColumn batchCol)
                    {
                        Contracts.AssertValue(batchCol);
                        Contracts.Assert(batchCol is BatchColumn.Impl<T>);
                        // In all possible scenarios, there is never cause for an output pipe
                        // to end early while the cursor itself has more rows, I believe, except
                        // if we at some point decide to optimize move many.
                        Contracts.Assert(_count == 0 || (_index == _count - 1));
                        // REVIEW: This sort of loose typing makes me angry. Roar!
                        var batchColTyped = (BatchColumn.Impl<T>)batchCol;
                        if (Values != null)
                            _pool.Return(Values);
                        Values = batchColTyped.Values;
                        _count = batchColTyped.Count;
                        _index = 0;
                        Contracts.Assert(_count <= Utils.Size(Values));
                    }

                    public override void Unset()
                    {
                        Contracts.Assert(_index <= _count);
                        if (Values != null)
                            _pool.Return(Values);
                        Values = null;
                        _count = 0;
                        _index = 0;
                    }

                    public override Delegate GetGetter()
                    {
                        ValueGetter<T> getter = Getter;
                        return getter;
                    }

                    protected abstract void Getter(ref T value);
                }

                private sealed class ImplVec<T> : Impl<VBuffer<T>>
                {
                    public ImplVec(object pool)
                        : base(pool)
                    {
                    }

                    protected override void Getter(ref VBuffer<T> value)
                    {
                        Contracts.Check(_index < _count, "Cannot get value as the cursor is not in a good state");
                        Values[_index].CopyTo(ref value);
                    }
                }

                private sealed class ImplOne<T> : Impl<T>
                {
                    public ImplOne(object pool)
                        : base(pool)
                    {
                    }

                    protected override void Getter(ref T value)
                    {
                        Contracts.Check(_index < _count, "Cannot get value as the cursor is not in a good state");
                        value = Values[_index];
                    }
                }
            }

            /// <summary>
            /// A cursor used by both the splitter and consolidator, that iteratively consumes
            /// <see cref="Batch"/> objects from the input blocking collection, and yields the
            /// values stored therein through the help of <see cref="OutPipe"/> objects.
            /// </summary>
            private sealed class Cursor : RootCursorBase, IRowCursor
            {
                private readonly ISchema _schema;
                private readonly int[] _activeToCol;
                private readonly int[] _colToActive;
                private readonly OutPipe[] _pipes;
                private readonly Delegate[] _getters;
                private readonly ValueGetter<UInt128> _idGetter;
                private readonly BlockingCollection<Batch> _batchInputs;
                private readonly Action _quitAction;

                private int _remaining;
                private long _batch;
                private bool _disposed;

                public ISchema Schema
                {
                    get { return _schema; }
                }

                public override long Batch
                {
                    get { return _batch; }
                }

                /// <summary>
                /// Constructs one of the split cursors.
                /// </summary>
                /// <param name="provider">The channel provider.</param>
                /// <param name="schema">The schema.</param>
                /// <param name="activeToCol">The mapping from active indices, to input column indices.</param>
                /// <param name="colToActive">The reverse mapping from input column indices to active indices,
                /// where -1 is present if this column is not active.</param>
                /// <param name="pipes">The output pipes, one per channel of information</param>
                /// <param name="batchInputs"></param>
                /// <param name="quitAction"></param>
                public Cursor(IChannelProvider provider, ISchema schema, int[] activeToCol, int[] colToActive,
                    OutPipe[] pipes, BlockingCollection<Batch> batchInputs, Action quitAction)
                    : base(provider)
                {
                    Ch.AssertValue(schema);
                    Ch.AssertValue(activeToCol);
                    Ch.AssertValue(colToActive);
                    Ch.AssertValue(pipes);
                    Ch.AssertValue(batchInputs);
                    Ch.AssertValueOrNull(quitAction);
                    Ch.Assert(colToActive.Length == schema.ColumnCount);
                    Ch.Assert(activeToCol.Length + (int)ExtraIndex._Lim == pipes.Length);
                    Ch.Assert(pipes.All(p => p != null));
                    // Could also confirm the inverse mappiness of activeToCol/colToActive, but that seems like a bit much.
                    _schema = schema;
                    _activeToCol = activeToCol;
                    _colToActive = colToActive;
                    _pipes = pipes;
                    _getters = new Delegate[pipes.Length];
                    for (int i = 0; i < activeToCol.Length; ++i)
                        _getters[i] = _pipes[i].GetGetter();
                    _idGetter = (ValueGetter<UInt128>)_pipes[activeToCol.Length + (int)ExtraIndex.Id].GetGetter();
                    _batchInputs = batchInputs;
                    _batch = -1;
                    _quitAction = quitAction;
                }

                public override void Dispose()
                {
                    if (!_disposed)
                    {
                        foreach (var pipe in _pipes)
                            pipe.Unset();
                        _disposed = true;
                        if (_quitAction != null)
                            _quitAction();
                    }
                    base.Dispose();
                }

                public override ValueGetter<UInt128> GetIdGetter()
                {
                    return _idGetter;
                }

                protected override bool MoveNextCore()
                {
                    Ch.Assert(!_disposed && State != CursorState.Done);
                    if (--_remaining > 0)
                    {
                        // We are still consuming the current output pipes.
                        foreach (var pipe in _pipes)
                        {
                            pipe.MoveNext();
                            Ch.Assert(pipe.Remaining == _remaining);
                        }
                    }
                    else
                    {
                        // We are done with the current output pipe or we are just getting started.

                        // REVIEW: Before testing I had a solution based on consuming enumerables, but the
                        // consuming enumerable did not do the "right thing," consistent with how it is documented,
                        // or indeed consistent with how I've ever seen it work in the past. All but one of the
                        // consuming enumerables would exit immediately, despite the underlying collection not
                        // being add completed? The below "Take" based mechanism with a sentinel does work, but I
                        // find the fact that the first solution did not work very troubling.
                        var nextBatch = _batchInputs.Take();
                        if (nextBatch == null)
                            return false;
                        Ch.Assert(nextBatch.HasException || nextBatch.BatchId > Batch);
                        Ch.Assert(nextBatch.HasException || nextBatch.Count > 0);
                        _batch = nextBatch.BatchId;
                        _remaining = nextBatch.Count;
                        // Note that setting an out pipe sets it on the first item, not before it, so it is not
                        // necessary to move the pipe.
                        nextBatch.SetAll(_pipes);
                        Ch.Assert(!nextBatch.HasException);
                    }
                    return true;
                }

                public bool IsColumnActive(int col)
                {
                    Ch.CheckParam(0 <= col && col < _colToActive.Length, nameof(col));
                    return _colToActive[col] >= 0;
                }

                public ValueGetter<TValue> GetGetter<TValue>(int col)
                {
                    Ch.CheckParam(IsColumnActive(col), nameof(col), "requested column not active");
                    var getter = _getters[_colToActive[col]] as ValueGetter<TValue>;
                    if (getter == null)
                        throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                    return getter;
                }
            }
        }

        /// <summary>
        /// This is a consolidating cursor that is usable even with cursors that are uncachable,
        /// at the cost of being totally synchronous, that is, there is no parallel benefit from
        /// having split the input cursors.
        /// </summary>
        internal sealed class SynchronousConsolidatingCursor : RootCursorBase, IRowCursor
        {
            private readonly IRowCursor[] _cursors;
            private readonly Delegate[] _getters;

            private readonly ISchema _schema;
            private readonly Heap<CursorStats> _mins;
            private readonly int[] _activeToCol;
            private readonly int[] _colToActive;
            private readonly MethodInfo _methInfo;

            // The batch number of the current input cursor, or -1 if this cursor is not in Good state.
            private long _batch;
            // Index into _cursors array pointing to the current cursor, or -1 if this cursor is not in Good state.
            private int _icursor;
            // If this cursor is in Good state then this should equal _cursors[_icursor], else null.
            private IRowCursor _currentCursor;
            private bool _disposed;

            private struct CursorStats
            {
                public readonly long Batch;
                public readonly int CursorIdx;

                public CursorStats(long batch, int idx)
                {
                    Batch = batch;
                    CursorIdx = idx;
                }
            }

            // REVIEW: It is not *really* necessary that we actually pass along the
            // input batch as our own batch. Should we suppress it?
            public override long Batch { get { return _batch; } }

            public ISchema Schema { get { return _schema; } }

            public SynchronousConsolidatingCursor(IChannelProvider provider, IRowCursor[] cursors)
                : base(provider)
            {
                Ch.CheckNonEmpty(cursors, nameof(cursors));
                _cursors = cursors;
                _schema = _cursors[0].Schema;

                Utils.BuildSubsetMaps(_schema.ColumnCount, _cursors[0].IsColumnActive, out _activeToCol, out _colToActive);

                Func<int, Delegate> func = CreateGetter<int>;
                _methInfo = func.GetMethodInfo().GetGenericMethodDefinition();

                _getters = new Delegate[_activeToCol.Length];
                for (int i = 0; i < _activeToCol.Length; ++i)
                    _getters[i] = CreateGetter(_activeToCol[i]);
                _icursor = -1;
                _batch = -1;

                _mins = new Heap<CursorStats>((s1, s2) => s1.Batch > s2.Batch);
                InitHeap();
            }

            private void InitHeap()
            {
                for (int i = 0; i < _cursors.Length; ++i)
                {
                    IRowCursor cursor = _cursors[i];
                    Ch.Assert(cursor.State == CursorState.NotStarted);
                    if (cursor.MoveNext())
                        _mins.Add(new CursorStats(cursor.Batch, i));
                }
            }

            public override void Dispose()
            {
                if (!_disposed)
                {
                    _disposed = true;
                    _batch = -1;
                    _icursor = -1;
                    _currentCursor = null;

                    foreach (var cursor in _cursors)
                        cursor.Dispose();
                }
                base.Dispose();
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                ValueGetter<UInt128>[] idGetters = new ValueGetter<UInt128>[_cursors.Length];
                for (int i = 0; i < _cursors.Length; ++i)
                    idGetters[i] = _cursors[i].GetIdGetter();
                return
                    (ref UInt128 val) =>
                    {
                        Ch.Check(_icursor >= 0, "Cannot call ID getter in current state");
                        idGetters[_icursor](ref val);
                    };
            }

            private Delegate CreateGetter(int col)
            {
                var methInfo = _methInfo.MakeGenericMethod(Schema.GetColumnType(col).RawType);
                return (Delegate)methInfo.Invoke(this, new object[] { col });
            }

            private Delegate CreateGetter<T>(int col)
            {
                ValueGetter<T>[] getters = new ValueGetter<T>[_cursors.Length];
                var type = Schema.GetColumnType(col);
                for (int i = 0; i < _cursors.Length; ++i)
                {
                    var cursor = _cursors[i];
                    Ch.AssertValue(cursor);
                    Ch.Assert(col < cursor.Schema.ColumnCount);
                    Ch.Assert(cursor.IsColumnActive(col));
                    Ch.Assert(type.Equals(cursor.Schema.GetColumnType(col)));
                    getters[i] = _cursors[i].GetGetter<T>(col);
                }
                ValueGetter<T> mine =
                    (ref T value) =>
                    {
                        Ch.Check(_icursor >= 0, "Cannot get value as the cursor is not in a good state");
                        getters[_icursor](ref value);
                    };
                return mine;
            }

            protected override bool MoveNextCore()
            {
                Ch.Assert(!_disposed && State != CursorState.Done);
                if (State == CursorState.Good && _currentCursor.MoveNext())
                {
                    // If we're still in this batch, no need to do anything, yet.
                    if (_currentCursor.Batch == _batch)
                        return true;
                    // We've run past the end of our batch, but not past the end of our
                    // cursor. Put this cursor back into the heap, and prepare to select
                    // a new minimum batch cursor.
                    Ch.Assert(_currentCursor.Batch > _batch);
                    _mins.Add(new CursorStats(_currentCursor.Batch, _icursor));
                }
                // This will happen if none of the cursors have any output rows left.
                if (_mins.Count == 0)
                    return false;
                // This is either the first call, or a time when we've run past the end of
                // some batch with some cursors with more rows. Because we only know the
                // batch ID once we've moved into a row, we do not need to, at this time.
                var stats = _mins.Pop();
                Ch.Assert(State == CursorState.NotStarted || stats.Batch > _batch);
                _icursor = stats.CursorIdx;
                _currentCursor = _cursors[stats.CursorIdx];
                _batch = _currentCursor.Batch;
                Ch.Assert(stats.Batch == _currentCursor.Batch);
                return true;
            }

            public bool IsColumnActive(int col)
            {
                Ch.CheckParam(0 <= col && col < _colToActive.Length, nameof(col));
                return _colToActive[col] >= 0;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.CheckParam(IsColumnActive(col), nameof(col), "requested column not active");
                var getter = _getters[_colToActive[col]] as ValueGetter<TValue>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }
        }

        public static ValueGetter<ReadOnlyMemory<char>>[] PopulateGetterArray(IRowCursor cursor, List<int> colIndices)
        {
            var n = colIndices.Count;
            var getters = new ValueGetter<ReadOnlyMemory<char>>[n];

            for (int i = 0; i < n; i++)
            {
                ValueGetter<ReadOnlyMemory<char>> getter;
                var srcColIndex = colIndices[i];

                var colType = cursor.Schema.GetColumnType(srcColIndex);
                if (colType.IsVector)
                {
                    getter = Utils.MarshalInvoke(GetVectorFlatteningGetter<int>, colType.ItemType.RawType,
                        cursor, srcColIndex, colType.ItemType);
                }
                else
                {
                    getter = Utils.MarshalInvoke(GetSingleValueGetter<int>, colType.ItemType.RawType,
                        cursor, srcColIndex, colType);
                }

                getters[i] = getter;
            }

            return getters;
        }

        public static ValueGetter<ReadOnlyMemory<char>> GetSingleValueGetter<T>(IRow cursor, int i, ColumnType colType)
        {
            var floatGetter = cursor.GetGetter<T>(i);
            T v = default(T);
            ValueMapper<T, StringBuilder> conversion;
            if (!Conversions.Instance.TryGetStringConversion<T>(colType, out conversion))
            {
                var error = $"Cannot display {colType}";
                conversion = (ref T src, ref StringBuilder builder) =>
                {
                    if (builder == null)
                        builder = new StringBuilder();
                    else
                        builder.Clear();
                    builder.Append(error);
                };
            }

            StringBuilder dst = null;
            ValueGetter<ReadOnlyMemory<char>> getter =
                (ref ReadOnlyMemory<char> value) =>
                {
                    floatGetter(ref v);
                    conversion(ref v, ref dst);
                    string text = dst.ToString();
                    value = text.AsMemory();
                };
            return getter;
        }

        public static ValueGetter<ReadOnlyMemory<char>> GetVectorFlatteningGetter<T>(IRow cursor, int colIndex, ColumnType colType)
        {
            var vecGetter = cursor.GetGetter<VBuffer<T>>(colIndex);
            var vbuf = default(VBuffer<T>);
            const int previewValues = 100;
            ValueMapper<T, StringBuilder> conversion;
            Conversions.Instance.TryGetStringConversion<T>(colType, out conversion);
            StringBuilder dst = null;
            ValueGetter<ReadOnlyMemory<char>> getter =
                (ref ReadOnlyMemory<char> value) =>
                {
                    vecGetter(ref vbuf);

                    bool isLong = vbuf.Length > previewValues;
                    var suffix = isLong ? string.Format(",...(+{0})", vbuf.Length - previewValues) : "";
                    var stringRep = string.Join(",", vbuf.Items(true).Take(previewValues)
                        .Select(
                            x =>
                            {
                                var v = x.Value;
                                conversion(ref v, ref dst);
                                return dst.ToString();
                            }));
                    value = string.Format("<{0}{1}>", stringRep, suffix).AsMemory();
                };
            return getter;
        }
    }
}
