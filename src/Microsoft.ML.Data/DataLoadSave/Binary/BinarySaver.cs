// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(BinarySaver.Summary, typeof(BinarySaver), typeof(BinarySaver.Arguments), typeof(SignatureDataSaver),
    "Binary Saver", "BinarySaver", "Binary")]

namespace Microsoft.ML.Runtime.Data.IO
{
    using Stopwatch = System.Diagnostics.Stopwatch;

    public sealed class BinarySaver : IDataSaver
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The compression scheme to use for the blocks", ShortName = "comp")]
            public CompressionKind Compression = CompressionKind.Default;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The block-size heuristic will choose no more than this many rows to have per block, can be set to null to indicate that there is no inherent limit", ShortName = "rpb")]
            public int? MaxRowsPerBlock = 1 << 13; // ~8 thousand.

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The block-size heuristic will attempt to have about this many bytes across all columns per block, can be set to null to accept the inidcated max-rows-per-block as the number of rows per block", ShortName = "bpb")]
            public long? MaxBytesPerBlock = 80 << 20; // ~80 megabytes.

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If true, this forces a deterministic block order during writing", ShortName = "det")]
            public bool DeterministicBlockOrder = false; // REVIEW: Should this be true? Should we have multiple layout schemes?

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Suppress any info output (not warnings or errors)", Hide = true)]
            public bool Silent;
        }

        internal const string Summary = "Writes data into a native binary IDV file.";

        private readonly IHost _host;
        private readonly CodecFactory _factory;
        private readonly MemoryStreamPool _memPool;

        private readonly CompressionKind _compression;
        private readonly int? _maxRowsPerBlock;
        private readonly long? _maxBytesPerBlock;
        private readonly bool _deterministicBlockOrder;
        private readonly bool _silent;

        private long _rowCount;

        /// <summary>
        /// This is a simple struct to associate a source index with a codec, without having to have
        /// parallel structures everywhere.
        /// </summary>
        private struct ColumnCodec
        {
            public readonly int SourceIndex;
            public readonly IValueCodec Codec;

            public ColumnCodec(int sourceIndex, IValueCodec codec)
            {
                SourceIndex = sourceIndex;
                Codec = codec;
            }
        }

        private abstract class WritePipe
        {
            protected readonly BinarySaver Parent;

            protected WritePipe(BinarySaver parent)
            {
                Contracts.AssertValue(parent);
                Parent = parent;
            }

            /// <summary>
            /// Returns an appropriate generic <c>WritePipe{T}</c> for the given column.
            /// </summary>
            public static WritePipe Create(BinarySaver parent, IRowCursor cursor, ColumnCodec col)
            {
                Type writePipeType = typeof(WritePipe<>).MakeGenericType(col.Codec.Type.RawType);
                return (WritePipe)Activator.CreateInstance(writePipeType, parent, cursor, col);
            }

            public abstract void BeginBlock();

            public abstract void FetchAndWrite();

            public abstract MemoryStream EndBlock();
        }

        private sealed class WritePipe<T> : WritePipe
        {
            private ValueGetter<T> _getter;
            private IValueCodec<T> _codec;
            private IValueWriter<T> _writer;
            private MemoryStream _currentStream;
            private T _value;

            public WritePipe(BinarySaver parent, IRowCursor cursor, ColumnCodec col)
                : base(parent)
            {
                var codec = col.Codec as IValueCodec<T>;
                Contracts.AssertValue(codec);
                _codec = codec;
                _getter = cursor.GetGetter<T>(col.SourceIndex);
            }

            public override void BeginBlock()
            {
                Contracts.Assert(_writer == null);
                _currentStream = Parent._memPool.Get();
                _writer = _codec.OpenWriter(_currentStream);
            }

            public override void FetchAndWrite()
            {
                Contracts.Assert(_writer != null);
                _getter(ref _value);
                _writer.Write(ref _value);
            }

            public override MemoryStream EndBlock()
            {
                Contracts.Assert(_writer != null);
                _writer.Commit();
                _writer = null;
                var retval = _currentStream;
                _currentStream = null;
                return retval;
            }
        }

        /// <summary>
        /// A class useful for encapsulating both compressed and uncompressed block data.
        /// As the mechanism the compress workers communicate with the writer worker, they
        /// also have a dual usage if <see cref="Exception"/> is non-null of indicating
        /// a source worker threw an exception.
        /// </summary>
        private struct Block
        {
            /// <summary>
            /// Take one guess.
            /// </summary>
            public readonly MemoryStream BlockData;
            /// <summary>
            /// The length of the block if uncompressed. This quantity is only intended to be
            /// meaningful if the block data is compressed.
            /// </summary>
            public readonly int UncompressedLength;
            /// <summary>
            /// The column index, which is the index of the column as being written, which
            /// may be less than the column from the source dataview if there were preceeding
            /// columns being dropped.
            /// </summary>
            public readonly int ColumnIndex;
            /// <summary>
            /// The block sequence number for this column, starting consecutively from 0.
            /// </summary>
            public readonly long BlockIndex;

            public Block(MemoryStream data, int colindex, long blockIndex, int uncompLength = 0)
            {
                BlockData = data;
                ColumnIndex = colindex;
                BlockIndex = blockIndex;
                UncompressedLength = uncompLength;
            }
        }

        /// <summary>
        /// Constructs a saver for a data view.
        /// </summary>
        public BinarySaver(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("BinarySaver");

            _host.CheckUserArg(!args.MaxRowsPerBlock.HasValue || args.MaxRowsPerBlock > 0, nameof(args.MaxRowsPerBlock), "Must be positive.");
            _host.CheckUserArg(!args.MaxBytesPerBlock.HasValue || args.MaxBytesPerBlock > 0, nameof(args.MaxBytesPerBlock), "Must be positive.");

            _host.CheckUserArg(args.MaxRowsPerBlock.HasValue || args.MaxBytesPerBlock.HasValue, nameof(args.MaxBytesPerBlock),
                "Either " + nameof(args.MaxRowsPerBlock) + " or " + nameof(args.MaxBytesPerBlock) + " must have a value.");

            _memPool = new MemoryStreamPool();
            _factory = new CodecFactory(_host, _memPool);
            _compression = args.Compression;

            _maxRowsPerBlock = args.MaxRowsPerBlock;
            _maxBytesPerBlock = args.MaxBytesPerBlock;
            _deterministicBlockOrder = args.DeterministicBlockOrder;
            _silent = args.Silent;
        }

        private void CompressionWorker(BlockingCollection<Block> toCompress, BlockingCollection<Block> toWrite, int columns, OrderedWaiter waiter,
            ExceptionMarshaller exMarshaller)
        {
            Contracts.AssertValue(exMarshaller);
            try
            {
                _host.AssertValue(toCompress);
                _host.AssertValue(toWrite);
                _host.Assert(columns > 0);
                _host.Assert(_deterministicBlockOrder == (waiter != null));

                foreach (Block block in toCompress.GetConsumingEnumerable(exMarshaller.Token))
                {
                    MemoryStream compressed = _memPool.Get();
                    int uncompLength;
                    using (Stream stream = _compression.CompressStream(compressed))
                    {
                        MemoryStream uncompressed = block.BlockData;
                        uncompLength = (int)uncompressed.Length;
                        ArraySegment<byte> buffer;
                        bool tmp = uncompressed.TryGetBuffer(out buffer);
                        Contracts.Assert(tmp);
                        stream.Write(buffer.Array, buffer.Offset, buffer.Count);
                        _memPool.Return(ref uncompressed);
                    }
                    if (_deterministicBlockOrder)
                        waiter.Wait((long)columns * block.BlockIndex + block.ColumnIndex, exMarshaller.Token);
                    toWrite.Add(new Block(compressed, block.ColumnIndex, block.BlockIndex, uncompLength), exMarshaller.Token);
                    if (_deterministicBlockOrder)
                        waiter.Increment();
                }
            }
            catch (Exception ex)
            {
                exMarshaller.Set("compressing", ex);
            }
        }

        /// <summary>
        /// A helper method to query and write metadata to the stream.
        /// </summary>
        /// <param name="writer">A binary writer, which if metadata exists for the
        /// indicated column the base stream will be positioned just past the end of
        /// the written metadata table of contents, and if metadata does not exist
        /// remains unchanged</param>
        /// <param name="schema">The schema to query for metadat</param>
        /// <param name="col">The column we are attempting to get metadata for</param>
        /// <param name="ch">The channel to which we write any diagnostic information</param>
        /// <returns>The offset of the metadata table of contents, or 0 if there was
        /// no metadata</returns>
        private long WriteMetadata(BinaryWriter writer, ISchema schema, int col, IChannel ch)
        {
            _host.AssertValue(writer);
            _host.AssertValue(schema);
            _host.Assert(0 <= col && col < schema.ColumnCount);

            int count = 0;
            WriteMetadataCoreDelegate del = WriteMetadataCore<int>;
            MethodInfo methInfo = del.GetMethodInfo().GetGenericMethodDefinition();
            object[] args = new object[] { writer.BaseStream, schema, col, null, null, null };

            List<long> offsets = new List<long>();
            offsets.Add(writer.BaseStream.Position);
            var metadataInfos = new List<Tuple<string, IValueCodec, CompressionKind>>();
            var kinds = new HashSet<string>();

            // Write all metadata blocks for this column to the file, one after the other, keeping
            // track of the location and size of each for when we write the metadata table of contents.
            // (To be clear, this specific layout is not required by the format.)

            foreach (var pair in schema.GetMetadataTypes(col))
            {
                _host.Check(!string.IsNullOrEmpty(pair.Key), "Metadata with null or empty kind detected, disallowed");
                _host.Check(pair.Value != null, "Metadata with null type detected, disallowed");
                if (!kinds.Add(pair.Key))
                    throw _host.Except("Metadata with duplicate kind '{0}' encountered, disallowed", pair.Key, schema.GetColumnName(col));
                args[3] = pair.Key;
                args[4] = pair.Value;
                IValueCodec codec = (IValueCodec)methInfo.MakeGenericMethod(pair.Value.RawType).Invoke(this, args);
                if (codec == null)
                {
                    // Nothing was written.
                    ch.Warning("Could not get codec for type {0}, dropping column '{1}' index {2} metadata kind '{3}'",
                        pair.Value, schema.GetColumnName(col), col, pair.Key);
                    continue;
                }
                offsets.Add(writer.BaseStream.Position);
                _host.CheckIO(offsets[offsets.Count - 1] > offsets[offsets.Count - 2], "Bad offsets detected during write");
                metadataInfos.Add(Tuple.Create(pair.Key, codec, (CompressionKind)args[5]));
                count++;
            }
            if (metadataInfos.Count == 0)
            {
                _host.CheckIO(writer.BaseStream.Position == offsets[0], "unexpected offset after no writing of metadata");
                return 0;
            }
            // Write the metadata table of contents just past the end of the last metadata block.

            // *** Metadata TOC format ***
            // LEB128 int: Number of metadata TOC entries
            // Metadata TOC entries: As many of these as indicated by the count above

            long expectedPosition = offsets[metadataInfos.Count];
            writer.WriteLeb128Int((ulong)metadataInfos.Count);
            expectedPosition += Utils.Leb128IntLength((ulong)metadataInfos.Count);
            for (int i = 0; i < metadataInfos.Count; ++i)
            {
                // *** Metadata TOC entry format ***
                // string: metadata kind
                // codec definition: metadata codec
                // CompressionKind(byte): block compression strategy
                // long: Offset into the stream of the start of the metadata block
                // LEB128 int: Byte size of the metadata block in the file

                writer.Write(metadataInfos[i].Item1);
                int stringLen = Encoding.UTF8.GetByteCount(metadataInfos[i].Item1);
                expectedPosition += Utils.Leb128IntLength((ulong)stringLen) + stringLen;
                _host.CheckIO(writer.BaseStream.Position == expectedPosition, "unexpected offsets after metadata table of contents kind");

                expectedPosition += _factory.WriteCodec(writer.BaseStream, metadataInfos[i].Item2);
                _host.CheckIO(writer.BaseStream.Position == expectedPosition, "unexpected offsets after metadata table of contents type description");

                writer.Write((byte)metadataInfos[i].Item3);
                expectedPosition++;

                writer.Write(offsets[i]);
                expectedPosition += sizeof(long);

                long blockSize = offsets[i + 1] - offsets[i];
                writer.WriteLeb128Int((ulong)blockSize);
                expectedPosition += Utils.Leb128IntLength((ulong)blockSize);
                _host.CheckIO(writer.BaseStream.Position == expectedPosition, "unexpected offsets after metadata table of contents location");
            }
            _host.Assert(metadataInfos.Count == offsets.Count - 1);
            return offsets[metadataInfos.Count];
        }

        private delegate IValueCodec WriteMetadataCoreDelegate(Stream stream, ISchema schema, int col, string kind, ColumnType type, out CompressionKind compression);

        private IValueCodec WriteMetadataCore<T>(Stream stream, ISchema schema, int col, string kind, ColumnType type, out CompressionKind compressionKind)
        {
            _host.Assert(typeof(T) == type.RawType);
            IValueCodec generalCodec;
            if (!_factory.TryGetCodec(type, out generalCodec))
            {
                compressionKind = default(CompressionKind);
                return null;
            }
            IValueCodec<T> codec = (IValueCodec<T>)generalCodec;
            T value = default(T);
            schema.GetMetadata(kind, col, ref value);

            // Metadatas will often be pretty small, so that compression makes no sense.
            // We try both a compressed and uncompressed version of metadata and
            // opportunistically pick whichever is smallest.
            MemoryStream uncompressedMem = _memPool.Get();
            using (IValueWriter<T> writer = codec.OpenWriter(uncompressedMem))
            {
                writer.Write(ref value);
                writer.Commit();
            }
            MemoryStream compressedMem = _memPool.Get();
            ArraySegment<byte> buffer;
            bool tmp = uncompressedMem.TryGetBuffer(out buffer);
            _host.Assert(tmp);
            using (Stream compressStream = _compression.CompressStream(compressedMem))
                compressStream.Write(buffer.Array, buffer.Offset, buffer.Count);
            if (uncompressedMem.Length <= compressedMem.Length)
            {
                // Write uncompressed.
                compressionKind = CompressionKind.None;
            }
            else
            {
                // Write compressed.
                compressionKind = _compression;
                tmp = compressedMem.TryGetBuffer(out buffer);
                _host.Assert(tmp);
            }
            stream.Write(buffer.Array, buffer.Offset, buffer.Count);
            _memPool.Return(ref uncompressedMem);
            _memPool.Return(ref compressedMem);
            return codec;
        }

        private void WriteWorker(Stream stream, BlockingCollection<Block> toWrite, ColumnCodec[] activeColumns,
            ISchema sourceSchema, int rowsPerBlock, IChannelProvider cp, ExceptionMarshaller exMarshaller)
        {
            _host.AssertValue(exMarshaller);
            try
            {
                _host.AssertValue(cp);
                cp.AssertValue(stream);
                cp.AssertValue(toWrite);
                cp.AssertValue(activeColumns);
                cp.AssertValue(sourceSchema);
                cp.Assert(rowsPerBlock > 0);

                using (IChannel ch = cp.Start("Write"))
                {
                    var blockLookups = new List<BlockLookup>[activeColumns.Length];
                    for (int c = 0; c < blockLookups.Length; ++c)
                        blockLookups[c] = new List<BlockLookup>();
                    var deadLookups = new int[activeColumns.Length];

                    // Reserve space for the header at the start. This will be filled
                    // in with valid values once writing has completed.
                    ch.CheckIO(stream.Position == 0);
                    stream.Write(new byte[Header.HeaderSize], 0, Header.HeaderSize);
                    ch.CheckIO(stream.Position == Header.HeaderSize);
                    long expectedPosition = stream.Position;
                    BlockLookup deadLookup = new BlockLookup();
                    foreach (Block block in toWrite.GetConsumingEnumerable(exMarshaller.Token))
                    {
                        ch.CheckIO(stream.Position == expectedPosition);
                        MemoryStream compressed = block.BlockData;
                        ArraySegment<byte> buffer;
                        bool tmp = compressed.TryGetBuffer(out buffer);
                        ch.Assert(tmp);
                        stream.Write(buffer.Array, buffer.Offset, buffer.Count);
                        BlockLookup currLookup = new BlockLookup(expectedPosition, (int)compressed.Length, block.UncompressedLength);
                        expectedPosition += compressed.Length;
                        _memPool.Return(ref compressed);
                        ch.CheckIO(stream.Position == expectedPosition);

                        // Record the position. We have this "lookups" list per column. Yet, it may be that sometimes
                        // the writer receives things out of order.
                        // REVIEW: The format and the rest of the pipeline supposedly supports a long number
                        // of blocks, but the writing scheme does not yet support that.
                        int blockIndex = (int)block.BlockIndex;
                        var lookups = blockLookups[block.ColumnIndex];
                        if (lookups.Count == block.BlockIndex) // Received in order.
                            lookups.Add(currLookup);
                        else if (lookups.Count < block.BlockIndex) // Received a block a little bit early.
                        {
                            // Add a bunch of dead filler lookups, until these late blocks come in.
                            int deadToAdd = (int)block.BlockIndex - lookups.Count;
                            for (int i = 0; i < deadToAdd; ++i)
                                lookups.Add(deadLookup);
                            deadLookups[block.ColumnIndex] += deadToAdd;
                            ch.Assert(lookups.Count == block.BlockIndex);
                            lookups.Add(currLookup);
                        }
                        else // Received a block a little bit late.
                        {
                            // This should be a dead block unless the compressors are buggy and somehow
                            // yielding duplicate blocks or something.
                            ch.Assert(lookups[blockIndex].BlockOffset == 0);
                            deadLookups[block.ColumnIndex]--;
                            lookups[blockIndex] = currLookup;
                        }
                    }

                    // We have finished writing all blocks. We will now write the block lookup tables (so we can
                    // find the blocks), the slot names (for any columns that have them), the column table of
                    // contents (so we know how to decode the blocks, and where the lookups and names are),
                    // and the header (so we know dataview wide information and where to find the table of
                    // contents) in that order.
                    long[] lookupOffsets = new long[blockLookups.Length];
                    using (BinaryWriter writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
                    {
                        // Write the block lookup directories. These are referenced from the table of contents,
                        // so that someone knows where to look for some block data.
                        for (int c = 0; c < blockLookups.Length; ++c)
                        {
                            ch.Assert(deadLookups[c] == 0);
                            // The block lookup directories are written uncompressed and in fixed length
                            // to enable rapid seeking.
                            lookupOffsets[c] = stream.Position;
                            foreach (BlockLookup lookup in blockLookups[c])
                            {
                                // *** Lookup table entry format ***
                                // long: Offset to the start of a block
                                // int: Byte length of block as written
                                // int: Byte length of block when uncompressed

                                ch.Assert(lookup.BlockOffset > 0);
                                writer.Write(lookup.BlockOffset);
                                writer.Write(lookup.BlockLength);
                                writer.Write(lookup.DecompressedBlockLength);
                            }
                            ch.CheckIO(stream.Position == lookupOffsets[c] + (16 * blockLookups[c].Count),
                                "unexpected offsets after block lookup table write");
                        }
                        // Write the metadata for each column.
                        long[] metadataTocOffsets = new long[activeColumns.Length];
                        for (int c = 0; c < activeColumns.Length; ++c)
                            metadataTocOffsets[c] = WriteMetadata(writer, sourceSchema, activeColumns[c].SourceIndex, ch);

                        // Write the table of contents.
                        long tocOffset = stream.Position;
                        {
                            int c = 0;
                            expectedPosition = stream.Position;
                            foreach (var active in activeColumns)
                            {
                                // *** Column TOC entry format ***
                                // string: column name
                                // codec (as interpretable by CodecFactory.TryGetCodec): column block codec
                                // CompressionKind(byte): block compression strategy
                                // LEB128 int: Rows per block
                                // long: Offset to the start of the lookup table
                                // long: Offset to the start of the metadata TOC entries, or 0 if this has no metadata

                                string name = sourceSchema.GetColumnName(active.SourceIndex);
                                writer.Write(name);
                                int nameLen = Encoding.UTF8.GetByteCount(name);
                                expectedPosition += Utils.Leb128IntLength((uint)nameLen) + nameLen;
                                ch.CheckIO(stream.Position == expectedPosition, "unexpected offsets after table of contents name");
                                expectedPosition += _factory.WriteCodec(stream, active.Codec);
                                ch.CheckIO(stream.Position == expectedPosition, "unexpected offsets after table of contents type description");
                                writer.Write((byte)_compression);
                                expectedPosition++;
                                // REVIEW: Right now the number of rows per block is fixed, so we
                                // write the same value each time. In some future state, it may be that this
                                // is relaxed, with possibly some tradeoffs (for example, inability to randomly seek).
                                writer.WriteLeb128Int((ulong)rowsPerBlock);
                                expectedPosition += Utils.Leb128IntLength((uint)rowsPerBlock);
                                // Offset of the lookup table.
                                writer.Write(lookupOffsets[c]);
                                expectedPosition += sizeof(long);
                                // Offset of the metadata table of contents.
                                writer.Write(metadataTocOffsets[c]);
                                expectedPosition += sizeof(long);
                                ch.CheckIO(stream.Position == expectedPosition, "unexpected offsets after table of contents");
                                c++;
                            }
                        }
                        // Write the tail signature.
                        long tailOffset = stream.Position;
                        writer.Write(Header.TailSignatureValue);

                        // Now move back to the beginning of the stream, and write out the now completed header.
                        Header header = new Header()
                        {
                            Signature = Header.SignatureValue,
                            Version = Header.WriterVersion,
                            CompatibleVersion = Header.CanBeReadByVersion,
                            TableOfContentsOffset = tocOffset,
                            TailOffset = tailOffset,
                            RowCount = _rowCount,
                            ColumnCount = activeColumns.Length
                        };
                        byte[] headerBytes = new byte[Header.HeaderSize];
                        unsafe
                        {
                            Marshal.Copy(new IntPtr(&header), headerBytes, 0, Marshal.SizeOf(typeof(Header)));
                        }
                        writer.Seek(0, SeekOrigin.Begin);
                        writer.Write(headerBytes);
                    }
                }
            }
            catch (Exception ex)
            {
                exMarshaller.Set("writing", ex);
            }
        }

        private void FetchWorker(BlockingCollection<Block> toCompress, IDataView data,
            ColumnCodec[] activeColumns, int rowsPerBlock, Stopwatch sw, IChannel ch, IProgressChannel pch, ExceptionMarshaller exMarshaller)
        {
            Contracts.AssertValue(ch);
            Contracts.AssertValueOrNull(pch);
            ch.AssertValue(exMarshaller);
            try
            {
                ch.AssertValue(toCompress);
                ch.AssertValue(data);
                ch.AssertValue(activeColumns);
                ch.AssertValue(sw);
                ch.Assert(rowsPerBlock > 0);

                // The main thread handles fetching from the cursor, and storing it into blocks passed to toCompress.
                HashSet<int> activeSet = new HashSet<int>(activeColumns.Select(col => col.SourceIndex));
                long blockIndex = 0;
                int remainingInBlock = rowsPerBlock;
                using (IRowCursor cursor = data.GetRowCursor(activeSet.Contains))
                {
                    WritePipe[] pipes = new WritePipe[activeColumns.Length];
                    for (int c = 0; c < activeColumns.Length; ++c)
                        pipes[c] = WritePipe.Create(this, cursor, activeColumns[c]);
                    for (int c = 0; c < pipes.Length; ++c)
                        pipes[c].BeginBlock();

                    long rows = 0;
                    if (pch != null)
                        pch.SetHeader(new ProgressHeader(new[] { "rows" }), e => e.SetProgress(0, rows));

                    while (cursor.MoveNext())
                    {
                        for (int c = 0; c < pipes.Length; ++c)
                            pipes[c].FetchAndWrite();
                        if (--remainingInBlock == 0)
                        {
                            for (int c = 0; c < pipes.Length; ++c)
                            {
                                // REVIEW: It may be better if EndBlock got moved to a different worker thread.
                                toCompress.Add(new Block(pipes[c].EndBlock(), c, blockIndex), exMarshaller.Token);
                                pipes[c].BeginBlock();
                            }
                            remainingInBlock = rowsPerBlock;
                            blockIndex++;
                        }

                        rows++;
                    }
                    if (remainingInBlock < rowsPerBlock)
                    {
                        for (int c = 0; c < pipes.Length; ++c)
                            toCompress.Add(new Block(pipes[c].EndBlock(), c, blockIndex), exMarshaller.Token);
                    }

                    Contracts.Assert(rows == (blockIndex + 1) * rowsPerBlock - remainingInBlock);
                    _rowCount = rows;
                    if (pch != null)
                        pch.Checkpoint(rows);
                }

                toCompress.CompleteAdding();
            }
            catch (Exception ex)
            {
                exMarshaller.Set("cursoring", ex);
            }
        }

        public bool IsColumnSavable(ColumnType type)
        {
            IValueCodec codec;
            return _factory.TryGetCodec(type, out codec);
        }

        public void SaveData(Stream stream, IDataView data, params int[] colIndices)
        {
            _host.CheckValue(stream, nameof(stream));
            _host.CheckValue(data, nameof(data));
            _host.CheckValueOrNull(colIndices);
            _host.CheckParam(stream.CanWrite, nameof(stream), "cannot save to non-writable stream");
            _host.CheckParam(stream.CanSeek, nameof(stream), "cannot save to non-seekable stream");
            _host.CheckParam(stream.Position == 0, nameof(stream), "stream must be positioned at head of stream");

            using (IChannel ch = _host.Start("Saving"))
            using (ExceptionMarshaller exMarshaller = new ExceptionMarshaller())
            {
                var toWrite = new BlockingCollection<Block>(16);
                var toCompress = new BlockingCollection<Block>(16);
                var activeColumns = GetActiveColumns(data.Schema, colIndices);
                int rowsPerBlock = RowsPerBlockHeuristic(data, activeColumns);
                ch.Assert(rowsPerBlock > 0);
                Stopwatch sw = new Stopwatch();

                // Set up the compression and write workers that consume the input information first.
                Task compressionTask = null;
                if (activeColumns.Length > 0)
                {
                    OrderedWaiter waiter = _deterministicBlockOrder ? new OrderedWaiter() : null;
                    Thread[] compressionThreads = new Thread[Environment.ProcessorCount];
                    for (int i = 0; i < compressionThreads.Length; ++i)
                    {
                        compressionThreads[i] = Utils.CreateBackgroundThread(
                            () => CompressionWorker(toCompress, toWrite, activeColumns.Length, waiter, exMarshaller));
                        compressionThreads[i].Start();
                    }
                    compressionTask = new Task(() =>
                    {
                        foreach (Thread t in compressionThreads)
                            t.Join();
                    });
                    compressionTask.Start();
                }

                // While there is an advantage to putting the IO into a separate thread, there is not an
                // advantage to having more than one worker.
                Thread writeThread = Utils.CreateBackgroundThread(
                    () => WriteWorker(stream, toWrite, activeColumns, data.Schema, rowsPerBlock, _host, exMarshaller));
                writeThread.Start();
                sw.Start();

                // REVIEW: For now the fetch worker just works in the main thread. If it's
                // a fairly large view through, it may be advantageous to consider breaking up the
                // fetchwrite operations on the pipes, somehow.
                // Despite running in the main thread for now, the fetch worker follows the same
                // pattern of utilizing exMarshaller.
                using (var pch = _silent ? null : _host.StartProgressChannel("BinarySaver"))
                {
                    FetchWorker(toCompress, data, activeColumns, rowsPerBlock, sw, ch, pch, exMarshaller);
                }

                _host.Assert(compressionTask != null || toCompress.IsCompleted);
                if (compressionTask != null)
                    compressionTask.Wait();
                toWrite.CompleteAdding();

                writeThread.Join();
                exMarshaller.ThrowIfSet(ch);
                if (!_silent)
                    ch.Info("Wrote {0} rows across {1} columns in {2}", _rowCount, activeColumns.Length, sw.Elapsed);
                // When we dispose the exception marshaller, this will set the cancellation token when we internally
                // dispose the cancellation token source, so one way or another those threads are being cancelled, even
                // if an exception is thrown in the main body of this function.
            }
        }

        private ColumnCodec[] GetActiveColumns(ISchema schema, int[] colIndices)
        {
            _host.AssertValue(schema);
            _host.AssertValueOrNull(colIndices);

            ColumnCodec[] activeSourceColumns = new ColumnCodec[Utils.Size(colIndices)];
            if (Utils.Size(colIndices) == 0)
                return activeSourceColumns;

            for (int c = 0; c < colIndices.Length; ++c)
            {
                ColumnType type = schema.GetColumnType(colIndices[c]);
                IValueCodec codec;
                if (!_factory.TryGetCodec(type, out codec))
                    throw _host.Except("Could not get codec for requested column {0} of type {1}", schema.GetColumnName(c), type);
                _host.Assert(type.Equals(codec.Type));
                activeSourceColumns[c] = new ColumnCodec(colIndices[c], codec);
            }
            return activeSourceColumns;
        }

        private int RowsPerBlockHeuristic(IDataView data, ColumnCodec[] actives)
        {
            // If we did not set a size bound, return the old bound.
            if (!_maxBytesPerBlock.HasValue)
            {
                _host.Assert(_maxRowsPerBlock.HasValue && _maxRowsPerBlock.Value > 0); // argument validation should have ensured this
                return _maxRowsPerBlock.Value;
            }
            long maxBytes = _maxBytesPerBlock.Value;

            // First get the cursor.
            HashSet<int> active = new HashSet<int>(actives.Select(cc => cc.SourceIndex));
            IRandom rand = data.CanShuffle ? new TauswortheHybrid(_host.Rand) : null;
            // Get the estimators.
            EstimatorDelegate del = EstimatorCore<int>;
            MethodInfo methInfo = del.GetMethodInfo().GetGenericMethodDefinition();

            using (IRowCursor cursor = data.GetRowCursor(active.Contains, rand))
            {
                object[] args = new object[] { cursor, null, null, null };
                var writers = new IValueWriter[actives.Length];
                var estimators = new Func<long>[actives.Length];
                for (int c = 0; c < actives.Length; ++c)
                {
                    var col = actives[c];
                    args[1] = col;
                    methInfo.MakeGenericMethod(col.Codec.Type.RawType).Invoke(this, args);
                    estimators[c] = (Func<long>)args[2];
                    writers[c] = (IValueWriter)args[3];
                }

                int rows = 0;
                // We can't really support more than this.
                int maxRowsPerBlock = _maxRowsPerBlock.GetValueOrDefault(int.MaxValue);
                while (rows < maxRowsPerBlock)
                {
                    if (!cursor.MoveNext())
                        break; // We'll just have one block for each column.
                    long totalEstimate = estimators.Sum(c => c());
                    if (totalEstimate > maxBytes)
                        break;
                    rows++;
                }
                return Math.Max(1, rows); // Possible that even a single row exceeds the "limit".
            }
        }

        private delegate void EstimatorDelegate(IRowCursor cursor, ColumnCodec col,
            out Func<long> fetchWriteEstimator, out IValueWriter writer);

        private void EstimatorCore<T>(IRowCursor cursor, ColumnCodec col,
            out Func<long> fetchWriteEstimator, out IValueWriter writer)
        {
            ValueGetter<T> getter = cursor.GetGetter<T>(col.SourceIndex);
            IValueCodec<T> codec = col.Codec as IValueCodec<T>;
            _host.AssertValue(codec);
            IValueWriter<T> specificWriter = codec.OpenWriter(Stream.Null);
            writer = specificWriter;
            T val = default(T);
            fetchWriteEstimator = () =>
            {
                getter(ref val);
                specificWriter.Write(ref val);
                return specificWriter.GetCommitLengthEstimate();
            };
        }

        /// <summary>
        /// A utility method to save a column type to a stream, if we have a codec for that.
        /// </summary>
        /// <param name="stream">The stream to save the description to</param>
        /// <param name="type">The type to save</param>
        /// <param name="bytesWritten">The number of bytes written to the stream, which will
        /// be zero if we could not save the stream</param>
        /// <returns>Returns if have the ability to save this column type. If we do, we write
        /// the description to the stream. If we do not, nothing is written to the stream and
        /// the stream is not advanced.</returns>
        public bool TryWriteTypeDescription(Stream stream, ColumnType type, out int bytesWritten)
        {
            _host.CheckValue(stream, nameof(stream));
            _host.CheckValue(type, nameof(type));

            IValueCodec codec;
            if (!_factory.TryGetCodec(type, out codec))
            {
                bytesWritten = 0;
                return false;
            }
            bytesWritten = _factory.WriteCodec(stream, codec);
            return true;
        }

        /// <summary>
        /// Attempts to load a type description from a stream. In all cases, in the event
        /// of a properly formatted stream, even if the type-descriptor is not recognized,
        /// the stream will be at the end of that type descriptor. Note that any detected
        /// format errors will result in a throw.
        /// </summary>
        /// <param name="stream">The stream to load the type description from</param>
        /// <returns>A non-null value if the type descriptor was recognized, or null if
        /// it was not</returns>
        public ColumnType LoadTypeDescriptionOrNull(Stream stream)
        {
            _host.CheckValue(stream, nameof(stream));

            IValueCodec codec;
            if (!_factory.TryReadCodec(stream, out codec))
                return null;
            return codec.Type;
        }

        /// <summary>
        /// A utility method to save a column type and value to a stream, if we have a codec for that.
        /// </summary>
        /// <param name="stream">The stream to write the type and value to</param>
        /// <param name="type">The type of the codec to write and utilize</param>
        /// <param name="value">The value to encode and write</param>
        /// <param name="bytesWritten">The number of bytes written</param>
        /// <returns>Whether the write was successful or not</returns>
        public bool TryWriteTypeAndValue<T>(Stream stream, ColumnType type, ref T value, out int bytesWritten)
        {
            _host.CheckValue(stream, nameof(stream));
            _host.CheckValue(type, nameof(type));
            _host.CheckParam(value.GetType() == type.RawType, nameof(value), "Value doesn't match type");

            IValueCodec codec;
            if (!_factory.TryGetCodec(type, out codec))
            {
                bytesWritten = 0;
                return false;
            }

            IValueCodec<T> codecT = (IValueCodec<T>)codec;

            bytesWritten = _factory.WriteCodec(stream, codec);

            using (var writer = codecT.OpenWriter(stream))
            {
                writer.Write(ref value);
                bytesWritten += (int)writer.GetCommitLengthEstimate();
                writer.Commit();
            }
            return true;
        }

        /// <summary>
        /// Attempts to load a type description and a value of that type from a stream.
        /// </summary>
        /// <param name="stream">The stream to load the type description and value from</param>
        /// <param name="type">A non-null value if the type descriptor was recognized, or null if
        /// it was not</param>
        /// <param name="value">A non-null value if the type descriptor was recognized and a value
        /// read, or null if the type descriptor was not recognized</param>
        /// <returns>Whether the load of a type description and value was successful</returns>
        public bool TryLoadTypeAndValue(Stream stream, out ColumnType type, out object value)
        {
            _host.CheckValue(stream, nameof(stream));

            IValueCodec codec;
            if (!_factory.TryReadCodec(stream, out codec))
            {
                type = null;
                value = null;
                return false;
            }
            type = codec.Type;

            Func<Stream, IValueCodec<int>, object> func = LoadValue<int>;
            var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(codec.Type.RawType);
            value = (meth.Invoke(this, new object[] { stream, codec }));
            return true;
        }

        /// <summary>
        /// Deserializes and returns a value given a stream and codec.
        /// </summary>
        private object LoadValue<T>(Stream stream, IValueCodec<T> codec)
        {
            _host.Assert(typeof(T) == codec.Type.RawType);
            T value = default(T);
            using (var reader = codec.OpenReader(stream, 1))
            {
                reader.MoveNext();
                reader.Get(ref value);
            }
            return value;
        }
    }
}
