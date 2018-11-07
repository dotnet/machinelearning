// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

[assembly: LoadableClass(ShuffleTransform.Summary, typeof(ShuffleTransform), typeof(ShuffleTransform.Arguments), typeof(SignatureDataTransform),
    "Shuffle Transform", "ShuffleTransform", "Shuffle", "shuf")]

[assembly: LoadableClass(ShuffleTransform.Summary, typeof(ShuffleTransform), null, typeof(SignatureLoadDataTransform),
    "Shuffle Transform", ShuffleTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This is a transform that, given any input dataview (even an unshufflable one) will,
    /// when we construct a randomized cursor attempt to perform a rude version of shuffling
    /// using a pool. A pool of a given number of rows will be constructed from the first
    /// rows in the input cursor, and then, successively, the output cursor will yield one
    /// of these rows and replace it with another row from the input.
    /// </summary>
    public sealed class ShuffleTransform : RowToRowTransformBase
    {
        private static class Defaults
        {
            public const int PoolRows = 1000;
            public const bool PoolOnly = false;
            public const bool ForceShuffle = false;
        }

        public sealed class Arguments
        {
            // REVIEW: A more intelligent heuristic, based on the expected size of the inputs, perhaps?
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The pool will have this many rows", ShortName = "rows")]
            public int PoolRows = Defaults.PoolRows;

            // REVIEW: Come up with a better way to specify the desired set of functionality.
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If true, the transform will not attempt to shuffle the input cursor but only shuffle based on the pool. This parameter has no effect if the input data was not itself shufflable.", ShortName = "po")]
            public bool PoolOnly = Defaults.PoolOnly;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If true, the transform will always provide a shuffled view.", ShortName = "force")]
            public bool ForceShuffle = Defaults.ForceShuffle;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If true, the transform will always shuffle the input. The default value is the same as forceShuffle.", ShortName = "forceSource")]
            public bool? ForceShuffleSource;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The random seed to use for forced shuffling.", ShortName = "seed")]
            public int? ForceShuffleSeed;
        }

        internal const string Summary = "Reorders rows in the dataset by pseudo-random shuffling.";

        public const string LoaderSignature = "ShuffleTrans";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SHUFFLET",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Force shuffle source saving
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ShuffleTransform).Assembly.FullName);
        }

        private const string RegistrationName = "Shuffle";

        private readonly int _poolRows;
        private readonly bool _poolOnly;
        private readonly bool _forceShuffle;
        private readonly bool _forceShuffleSource;
        private readonly int _forceShuffleSeed;
        // This field is possibly distinct from _input, with any non-vector or non-primitive
        // types removed, since we do not support these since the implementation does not
        // know how to copy other types of values.
        private readonly IDataView _subsetInput;

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="poolRows">The pool will have this many rows</param>
        /// <param name="poolOnly">If true, the transform will not attempt to shuffle the input cursor but only shuffle based on the pool. This parameter has no effect if the input data was not itself shufflable.</param>
        /// <param name="forceShuffle">If true, the transform will always provide a shuffled view.</param>
        public ShuffleTransform(IHostEnvironment env,
            IDataView input,
            int poolRows = Defaults.PoolRows,
            bool poolOnly = Defaults.PoolOnly,
            bool forceShuffle = Defaults.ForceShuffle)
            : this(env, new Arguments() { PoolRows = poolRows, PoolOnly = poolOnly, ForceShuffle = forceShuffle }, input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public ShuffleTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));

            Host.CheckUserArg(args.PoolRows > 0, nameof(args.PoolRows), "pool size must be positive");
            _poolRows = args.PoolRows;
            _poolOnly = args.PoolOnly;
            _forceShuffle = args.ForceShuffle;
            _forceShuffleSource = args.ForceShuffleSource ?? (!_poolOnly && _forceShuffle);
            Host.CheckUserArg(!(_poolOnly && _forceShuffleSource),
                nameof(args.ForceShuffleSource), "Cannot set both poolOnly and forceShuffleSource");

            if (_forceShuffle || _forceShuffleSource)
                _forceShuffleSeed = args.ForceShuffleSeed ?? Host.Rand.NextSigned();

            _subsetInput = SelectCachableColumns(input, env);
        }

        private ShuffleTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: number of rows for the sample pool
            // bool(as byte): whether randomize only this transform, and not its input
            // bool(as byte): whether present a shuffled cursor even when a shuffled cursor is not requested
            // bool(as byte): whether the input cursor is always shuffled even when a shuffled cursor is not requested
            // int, only present if one of the previous two were true: seed for the random number generator to use
            //   when a shuffled cursor was not requested
            _poolRows = ctx.Reader.ReadInt32();
            Host.CheckDecode(_poolRows > 0);
            _poolOnly = ctx.Reader.ReadBoolByte();
            _forceShuffle = ctx.Reader.ReadBoolByte();
            _forceShuffleSource = ctx.Reader.ReadBoolByte();
            Host.CheckDecode(!(_poolOnly && _forceShuffleSource));
            if (_forceShuffle || _forceShuffleSource)
                _forceShuffleSeed = ctx.Reader.ReadInt32();
            _subsetInput = SelectCachableColumns(input, host);
        }

        public static ShuffleTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model", ch => new ShuffleTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of rows for the sample pool
            // bool(as byte): whether randomize only this transform, and not its input
            // bool(as byte): whether present a shuffled cursor even when a shuffled cursor is not requested
            // bool(as byte): whether the input cursor is always shuffled even when a shuffled cursor is not requested
            // int, only present if one of the previous two were true: seed for the random number generator to use
            //   when a shuffled cursor was not requested
            ctx.Writer.Write(_poolRows);
            ctx.Writer.WriteBoolByte(_poolOnly);
            ctx.Writer.WriteBoolByte(_forceShuffle);
            ctx.Writer.WriteBoolByte(_forceShuffleSource);
            if (_forceShuffle || _forceShuffleSource)
                ctx.Writer.Write(_forceShuffleSeed);
        }

        /// <summary>
        /// Since shuffling requires serving up items potentially out of order we need to know
        /// how to save and then copy out values that we read. This transform knows how to save
        /// and copy out only primitive and vector valued columns, but nothing else, so any
        /// other columns are dropped.
        /// </summary>
        private static IDataView SelectCachableColumns(IDataView data, IHostEnvironment env)
        {
            List<int> columnsToDrop = null;
            var schema = data.Schema;
            for (int c = 0; c < schema.ColumnCount; ++c)
            {
                var type = schema.GetColumnType(c);
                if (!type.IsCachable())
                    Utils.Add(ref columnsToDrop, c);
            }
            if (Utils.Size(columnsToDrop) == 0)
                return data;

            var args = new ChooseColumnsByIndexTransform.Arguments();
            args.Drop = true;
            args.Index = columnsToDrop.ToArray();
            return new ChooseColumnsByIndexTransform(env, args, data);
        }

        /// <summary>
        /// Utility to check whether all types in an input schema are shufflable.
        /// </summary>
        internal static bool CanShuffleAll(ISchema schema)
        {
            for (int c = 0; c < schema.ColumnCount; ++c)
            {
                var type = schema.GetColumnType(c);
                if (!type.IsCachable())
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Utility to take a cursor, and get a shuffled version of this cursor.
        /// </summary>
        public static IRowCursor GetShuffledCursor(IChannelProvider provider, int poolRows, IRowCursor cursor, IRandom rand)
        {
            Contracts.CheckValue(provider, nameof(provider));

            provider.CheckParam(poolRows > 0, nameof(poolRows), "Must be positive");
            provider.CheckValue(cursor, nameof(cursor));
            // REVIEW: In principle, we could limit this check to only active columns,
            // if we extend the use of this utility.
            provider.CheckParam(CanShuffleAll(cursor.Schema), nameof(cursor), "Cannot shuffle a cursor with some uncachable columns");
            provider.CheckValue(rand, nameof(rand));

            if (poolRows == 1)
                return cursor;
            return new RowCursor(provider, poolRows, cursor, rand);
        }

        public override bool CanShuffle { get { return true; } }

        public override Schema Schema { get { return _subsetInput.Schema; } }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");
            return false;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            // REVIEW: This is slightly interesting. Our mechanism for inducing
            // randomness in the source cursor is this Random object, but this can change
            // from release to release. The correct solution, it seems, is to instead have
            // randomness injected into cursor creation by using IRandom (or something akin
            // to it), vs. just a straight system Random.

            // The desired functionality is to support some permutations of whether we allow
            // shuffling at the source level, or not.
            //
            // Pool       | Source   | Options
            // -----------+----------+--------
            // Randonly   | Never    | poolOnly+
            //    "       | Randonly | (default)
            //    "       | Always   | forceSource+
            // Always     | Never    | force+ poolOnly+
            // Always     | Randonly | force+ forceSource-
            // Always     | Always   | force+

            bool shouldShuffleMe = _forceShuffle || rand != null;
            bool shouldShuffleSource = _forceShuffleSource || (!_poolOnly && rand != null);

            IRandom myRandom = rand ?? (shouldShuffleMe || shouldShuffleSource ? RandomUtils.Create(_forceShuffleSeed) : null);
            if (shouldShuffleMe)
                rand = myRandom;
            IRandom sourceRand = shouldShuffleSource ? RandomUtils.Create(myRandom) : null;

            var input = _subsetInput.GetRowCursor(predicate, sourceRand);
            // If rand is null (so we're not doing pool shuffling) or number of pool rows is 1
            // (so any pool shuffling, if attempted, would be trivial anyway), just return the
            // source cursor.
            if (rand == null || _poolRows == 1)
                return input;
            return new RowCursor(Host, _poolRows, input, rand);
        }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);
            consolidator = null;
            return new IRowCursor[] { GetRowCursorCore(predicate, rand) };
        }

        /// <summary>
        /// This describes the row cursor. Let's imagine we instantiated our shuffle transform
        /// over a pool of size P. Logically, externally, the cursor acts as if you have this pool
        /// P and whenever you randomly sample and yield a row from it, that row is then discarded
        /// and replaced with the next row from the input source cursor.
        ///
        /// It would also be possible to implement in a way that cleaves closely to this logical
        /// interpretation, but this would be inefficient. We instead have a buffer of larger size
        /// P+B. A consumer (running presumably in the main thread) sampling and fetching items and a
        /// producer (running in a task, which may be running in a different thread) filling the buffer
        /// with items to sample, utilizing this extra space to enable an efficient possibly
        /// multithreaded scheme.
        ///
        /// The consumer, for its part, at any given time "owns" a contiguous portion of this buffer.
        /// (A contiguous portion of this buffer we consider to be able to wrap around, from the end
        /// to the beginning. The buffer is accessed in a "circular" fashion.) Consider that this portion
        /// is broken into three distinct regions: there is a sort of middle "sampling" region
        /// (usually of size P but possibly smaller when we've reached the end of the input and so are
        /// running out of rows to sample), a region before this sampling region composed of already
        /// sampled "dead" rows, and a "presampling" region after this sampling region composed of
        /// rows ready to be sampled in future iterations, but that we are not sampling yet (in order
        /// to behave equivalently to the simple logical model of at any given time sampling P items).
        /// The producer owns the complement of the portion owned by the consumer.
        ///
        /// As the cursor progresses, the producer fills in successive items in its portion of the
        /// buffer it owns, and passes them off to the consumer (not one item at a time, but rather in
        /// batches, to keep down the amount of intertask communication). The consumer in addition to
        /// taking ownership of these items, will also periodically pass dead items back to the producer
        /// (again, not one dead item at a time, but in batches when the number of dead items reaches
        /// a certain threshold).
        ///
        /// This communication is accomplished using a pair of BufferBlock instances, through which
        /// the producer and consumer are notified how many additional items they can take ownership
        /// of.
        ///
        /// As the consumer "selects" a row from the pool of selectable rows each time it moves to
        /// the next row, this randomly selected row is considered to be the "first" index, since this
        /// makes its subsequent transition to being a dead row much simpler. It would be inefficient to
        /// swap all the values in each column's buffer to accomplish this to make the selected row
        /// first, of course, so one rather swaps an index, so that these nicely behavior contiguous
        /// circular indices, get mapped in an index within the buffers, through a permutation maintained
        /// in the pipeIndices array.
        ///
        /// The result is something functionally equivalent to but but considerably faster than the
        /// simple implementation described in the first paragraph.
        /// </summary>
        private sealed class RowCursor : RootCursorBase, IRowCursor
        {
            /// <summary>
            /// Pipes, in addition to column values, will also communicate extra information
            /// enumerated within this. This enum serves the purpose of providing nice readable
            /// indices to these "extra" information in pipes.
            /// </summary>
            private enum ExtraIndex
            {
                Id,
                Lim
            }

            /// <summary>
            /// There is one of these created per active column plus any extra info, and is a mechanism
            /// through which the producer is able to ingest and store this data from the source cursor,
            /// and the consumer able to fetch data stored at particular indices.
            /// </summary>
            private abstract class ShufflePipe
            {
                private static volatile Type[] _pipeConstructorTypes;

                /// <summary>
                /// Creates a shuffle pipe, given a value getter.
                /// </summary>
                /// <param name="bufferSize">The size of the internal array.</param>
                /// <param name="type">The column type, which determines what type of pipe is created</param>
                /// <param name="getter">A getter that should be a value getter corresponding to the
                /// column type</param>
                /// <returns>An appropriate <see cref="ShufflePipe{T}"/></returns>
                public static ShufflePipe Create(int bufferSize, ColumnType type, Delegate getter)
                {
                    Contracts.Assert(bufferSize > 0);
                    Contracts.AssertValue(type);
                    Contracts.AssertValue(getter);

                    Type pipeType;
                    if (type.IsVector)
                        pipeType = typeof(ImplVec<>).MakeGenericType(type.ItemType.RawType);
                    else
                    {
                        Contracts.Assert(type.IsPrimitive);
                        pipeType = typeof(ImplOne<>).MakeGenericType(type.RawType);
                    }
                    if (_pipeConstructorTypes == null)
                        Interlocked.CompareExchange(ref _pipeConstructorTypes, new Type[] { typeof(int), typeof(Delegate) }, null);
                    var constructor = pipeType.GetConstructor(_pipeConstructorTypes);
                    return (ShufflePipe)constructor.Invoke(new object[] { bufferSize, getter });
                }

                /// <summary>
                /// Reads the cursor column's current value, and store it in the indicated index,
                /// in the internal array.
                /// </summary>
                public abstract void Fill(int idx);

                private sealed class ImplVec<T> : ShufflePipe<VBuffer<T>>
                {
                    public ImplVec(int bufferSize, Delegate getter)
                        : base(bufferSize, getter)
                    {
                    }

                    protected override void Copy(in VBuffer<T> src, ref VBuffer<T> dst)
                    {
                        src.CopyTo(ref dst);
                    }
                }

                private sealed class ImplOne<T> : ShufflePipe<T>
                {
                    public ImplOne(int bufferSize, Delegate getter)
                        : base(bufferSize, getter)
                    {
                    }

                    protected override void Copy(in T src, ref T dst)
                    {
                        dst = src;
                    }
                }
            }

            private abstract class ShufflePipe<T> : ShufflePipe
            {
                private readonly ValueGetter<T> _getter;
                protected readonly T[] Buffer;

                public ShufflePipe(int bufferSize, Delegate getter)
                {
                    Contracts.AssertValue(getter);
                    Contracts.Assert(getter is ValueGetter<T>);
                    _getter = (ValueGetter<T>)getter;
                    Buffer = new T[bufferSize];
                }

                public override void Fill(int idx)
                {
                    Contracts.Assert(0 <= idx && idx < Buffer.Length);
                    _getter(ref Buffer[idx]);
                }

                /// <summary>
                /// Copies the values stored at an index through a previous <see cref="Fill"/> method,
                /// call to a value.
                /// </summary>
                public void Fetch(int idx, ref T value)
                {
                    Contracts.Assert(0 <= idx && idx < Buffer.Length);
                    Copy(in Buffer[idx], ref value);
                }

                protected abstract void Copy(in T src, ref T dst);
            }

            // The number of examples to have in each synchronization block. This should be >= 1.
            private const int _blockSize = 16;
            // The number of spare blocks to keep the filler worker busy on. This should be >= 1.
            private const int _bufferDepth = 3;

            private readonly int _poolRows;
            private readonly IRowCursor _input;
            private readonly IRandom _rand;

            // This acts as mapping from the "circular" index to the actual index within the pipe.
            private readonly int[] _pipeIndices;
            // These shuffle pipes are the actual internal type-specific buffers. There is one of
            // these per active column, as well as those for additional side information.
            private readonly ShufflePipe[] _pipes;
            // Each delegate here corresponds to a pipe holding column data.
            private readonly Delegate[] _getters;
            // This delegate corresponds to the pipe holding ID data.
            private readonly ValueGetter<UInt128> _idGetter;

            // The current position of the output cursor in circular "space".
            private int _circularIndex;
            // The current position of the output cursor in pipe "space".
            private int _pipeIndex;
            // This indicates the current number of "dead" items at the head, prior
            // to the start of the circular index.
            private int _deadCount;
            // This indicates the current number of available items.
            private int _liveCount;
            private bool _doneConsuming;

            private readonly BufferBlock<int> _toProduce;
            private readonly BufferBlock<int> _toConsume;
            private readonly Task _producerTask;
            private Exception _producerTaskException;

            private readonly int[] _colToActivesIndex;

            public Schema Schema { get { return _input.Schema; } }

            public override long Batch
            {
                // REVIEW: Implement cursor set support.
                get { return 0; }
            }

            public RowCursor(IChannelProvider provider, int poolRows, IRowCursor input, IRandom rand)
                : base(provider)
            {
                Ch.AssertValue(input);
                Ch.AssertValue(rand);

                Ch.Assert(_blockSize > 0);
                Ch.Assert(_bufferDepth > 0);
                Ch.Assert(poolRows > 0);

                _poolRows = poolRows;
                _input = input;
                _rand = rand;

                _pipeIndices = Utils.GetIdentityPermutation(_poolRows - 1 + _bufferDepth * _blockSize);

                int colLim = Schema.ColumnCount;
                int numActive = 0;
                _colToActivesIndex = new int[colLim];
                for (int c = 0; c < colLim; ++c)
                    _colToActivesIndex[c] = _input.IsColumnActive(c) ? numActive++ : -1;
                _pipes = new ShufflePipe[numActive + (int)ExtraIndex.Lim];
                _getters = new Delegate[numActive];
                for (int c = 0; c < colLim; ++c)
                {
                    int ia = _colToActivesIndex[c];
                    if (ia < 0)
                        continue;
                    _pipes[ia] = ShufflePipe.Create(_pipeIndices.Length,
                        input.Schema.GetColumnType(c), RowCursorUtils.GetGetterAsDelegate(input, c));
                    _getters[ia] = CreateGetterDelegate(c);
                }
                var idPipe = _pipes[numActive + (int)ExtraIndex.Id] = ShufflePipe.Create(_pipeIndices.Length, NumberType.UG, input.GetIdGetter());
                _idGetter = CreateGetterDelegate<UInt128>(idPipe);
                // Initially, after the preamble to MoveNextCore, we want:
                // liveCount=0, deadCount=0, circularIndex=0. So we set these
                // funky values accordingly.
                _pipeIndex = _circularIndex = _pipeIndices.Length - 1;
                _deadCount = -1;
                _liveCount = 1;

                // Set up the producer worker.
                _toConsume = new BufferBlock<int>();
                _toProduce = new BufferBlock<int>();
                // First request the pool - 1 + block size rows, to get us going.
                PostAssert(_toProduce, _poolRows - 1 + _blockSize);
                // Queue up the remaining capacity.
                for (int i = 1; i < _bufferDepth; ++i)
                    PostAssert(_toProduce, _blockSize);

                _producerTask = LoopProducerWorker();
            }

            public override void Dispose()
            {
                if (_producerTask.Status == TaskStatus.Running)
                {
                    _toProduce.Post(0);
                    _producerTask.Wait();
                }
                base.Dispose();
            }

            public static void PostAssert<T>(ITargetBlock<T> target, T item)
            {
                bool retval = target.Post(item);
                Contracts.Assert(retval);
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                return _idGetter;
            }

            private async Task LoopProducerWorker()
            {
                try
                {
                    int circularIndex = 0;
                    for (; ; )
                    {
                        int requested = await _toProduce.ReceiveAsync();
                        if (requested == 0)
                        {
                            // We had some sort of early exit. Just go out, do not post even the
                            // sentinel to the consumer, as nothing will be consumed any more.
                            return;
                        }
                        Ch.Assert(requested >= _blockSize);
                        int numRows;
                        for (numRows = 0; numRows < requested; ++numRows)
                        {
                            Ch.Assert(0 <= circularIndex & circularIndex < _pipeIndices.Length);
                            if (!_input.MoveNext())
                                break;
                            int pipeIndex = _pipeIndices[circularIndex++];
                            Ch.Assert(0 <= pipeIndex & pipeIndex < _pipeIndices.Length);
                            for (int c = 0; c < _pipes.Length; ++c)
                                _pipes[c].Fill(pipeIndex);
                            if (circularIndex == _pipeIndices.Length)
                                circularIndex = 0;
                        }
                        PostAssert(_toConsume, numRows);
                        if (numRows < requested)
                        {
                            // We've reached the end of the cursor. Send the sentinel, then exit.
                            // This assumes that the receiver will receive things in Post order
                            // (so that the sentinel is received, after the last Post).
                            if (numRows > 0)
                                PostAssert(_toConsume, 0);
                            return;
                        }
                    }
                }
                catch (Exception ex)
                {
                    _producerTaskException = ex;
                    // Send the sentinel in this case as well, the field will be checked.
                    PostAssert(_toConsume, 0);
                }
            }

            protected override bool MoveNextCore()
            {
                Ch.Assert(_liveCount > 0);
                Ch.Assert(_deadCount < _blockSize || _doneConsuming);
                Ch.Assert(0 <= _circularIndex & _circularIndex < _pipeIndices.Length);

                if (++_circularIndex == _pipeIndices.Length)
                    _circularIndex = 0;
                --_liveCount;
                if (++_deadCount >= _blockSize && !_doneConsuming)
                {
                    // We should let the producer know it can give us more stuff.
                    // It is possible for int values to be sent beyond the
                    // end of the sentinel, but we suppose this is irrelevant.
                    PostAssert(_toProduce, _deadCount);
                    _deadCount = 0;
                }

                while (_liveCount < _poolRows && !_doneConsuming)
                {
                    // We are under capacity. Try to get some more.
                    int got = _toConsume.Receive();
                    if (got == 0)
                    {
                        // We've reached the end sentinel. There's no reason
                        // to attempt further communication with the producer.
                        // Check whether something horrible happened.
                        if (_producerTaskException != null)
                            throw Ch.Except(_producerTaskException, "Shuffle input cursor reader failed with an exception");
                        _doneConsuming = true;
                        break;
                    }
                    _liveCount += got;
                }
                if (_liveCount == 0)
                    return false;
                int circularSwapIndex = (_rand.Next(Math.Min(_liveCount, _poolRows)) + _circularIndex) % _pipeIndices.Length;
                _pipeIndex = _pipeIndices[circularSwapIndex];
                _pipeIndices[circularSwapIndex] = _pipeIndices[_circularIndex];
                _pipeIndices[_circularIndex] = _pipeIndex;
                return true;
            }

            public bool IsColumnActive(int col)
            {
                Ch.CheckParam(0 <= col && col < _colToActivesIndex.Length, nameof(col));
                Ch.Assert((_colToActivesIndex[col] >= 0) == _input.IsColumnActive(col));
                return _input.IsColumnActive(col);
            }

            private Delegate CreateGetterDelegate(int col)
            {
                Ch.Assert(0 <= col && col < _colToActivesIndex.Length);
                Ch.Assert(_colToActivesIndex[col] >= 0);
                Func<int, Delegate> createDel = CreateGetterDelegate<int>;
                return Utils.MarshalInvoke(createDel, Schema.GetColumnType(col).RawType, col);
            }

            private Delegate CreateGetterDelegate<TValue>(int col)
            {
                Ch.Assert(0 <= col && col < _colToActivesIndex.Length);
                Ch.Assert(_colToActivesIndex[col] >= 0);
                Ch.Assert(Schema.GetColumnType(col).RawType == typeof(TValue));
                return CreateGetterDelegate<TValue>(_pipes[_colToActivesIndex[col]]);
            }

            private ValueGetter<TValue> CreateGetterDelegate<TValue>(ShufflePipe pipe)
            {
                Ch.AssertValue(pipe);
                Ch.Assert(pipe is ShufflePipe<TValue>);
                var pipe2 = (ShufflePipe<TValue>)pipe;
                ValueGetter<TValue> getter =
                    (ref TValue value) =>
                    {
                        Ch.Assert(_pipeIndex == _pipeIndices[_circularIndex]);
                        pipe2.Fetch(_pipeIndex, ref value);
                    };
                return getter;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.CheckParam(0 <= col && col < _colToActivesIndex.Length, nameof(col));
                Ch.CheckParam(_colToActivesIndex[col] >= 0, nameof(col), "requested column not active");
                ValueGetter<TValue> getter = _getters[_colToActivesIndex[col]] as ValueGetter<TValue>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }
        }
    }
}
