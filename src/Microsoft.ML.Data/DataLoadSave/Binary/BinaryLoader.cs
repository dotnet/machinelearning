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
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(BinaryLoader.Summary, typeof(BinaryLoader), typeof(BinaryLoader.Arguments), typeof(SignatureDataLoader),
    "Binary Loader",
    BinaryLoader.LoadName,
    "Binary",
    "Bin")]

[assembly: LoadableClass(BinaryLoader.Summary, typeof(BinaryLoader), null, typeof(SignatureLoadDataLoader),
    "Binary Data View Loader", BinaryLoader.LoaderSignature)]

[assembly: LoadableClass(typeof(BinaryLoader.InfoCommand), typeof(BinaryLoader.InfoCommand.Arguments), typeof(SignatureCommand),
    "", BinaryLoader.InfoCommand.LoadName, "idv")]

namespace Microsoft.ML.Runtime.Data.IO
{
    public sealed class BinaryLoader : IDataLoader, IDisposable
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The number of worker decompressor threads to use", ShortName = "t")]
            public int? Threads;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If specified, the name of a column to generate and append, providing a U8 key-value indicating the index of the row within the binary file", ShortName = "rowIndex", Hide = true)]
            public string RowIndexName;

            // REVIEW: Is this the right knob? The other thing we could do is have a bound on number
            // of MB, based on an analysis of average block size.
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "When shuffling, the number of blocks worth of data to keep in the shuffle pool. " +
                "Larger values will make the shuffling more random, but use more memory. Set to 0 to use only block shuffling.", ShortName = "pb")]
            public Double PoolBlocks = _defaultShuffleBlocks;
        }

        /// <summary>
        /// Each column corresponds to a table of contents entry, describing information about the column
        /// and how values may be extracted. For columns represented physically within the stream this will
        /// include its location within the stream and a codec to decode the bytestreams, and for generated
        /// columns procedures to create them. This structure is used both for those columns that
        /// we know how to access (called alive columns), and those columns we do not know how to access
        /// (either because the value codec or compressions scheme is unrecognized, called a dead column).
        /// </summary>
        private sealed class TableOfContentsEntry
        {
            /// <summary>
            /// The name of the column.
            /// </summary>
            public readonly string Name;

            /// <summary>
            /// The codec we will use to read the values from the stream. This will be null if
            /// and only if this is a dead or generated column.
            /// </summary>
            public readonly IValueCodec Codec;

            /// <summary>
            /// The column type of the column. This will be null if and only if this is a dead
            /// column.
            /// </summary>
            public readonly ColumnType Type;

            /// <summary>
            /// The compression scheme used on this column's blocks.
            /// </summary>
            public readonly CompressionKind Compression;

            /// <summary>
            /// The number of rows in each block (except for the last one).
            /// </summary>
            public readonly int RowsPerBlock;

            /// <summary>
            /// The offset into the stream where the lookup table for this column is stored.
            /// </summary>
            public readonly long LookupOffset;

            /// <summary>
            /// The offset into the stream where the metadata TOC entries for this column are
            /// stored. This will be 0 if there is no metadata for this column.
            /// </summary>
            public readonly long MetadataTocOffset;

            /// <summary>
            /// The index of the column. Note that if there are dead columns, this value may
            /// differ from the corresponding column index as reported by the dataview.
            /// </summary>
            public readonly int ColumnIndex;

            // Non-null only for generated columns.
            private readonly Delegate _generatorDelegate;

            private readonly BinaryLoader _parent;
            private readonly IExceptionContext _ectx;

            // Initially null, but lazily constructed by GetLookup.
            private volatile BlockLookup[] _lookup;

            // Both initially -1, but lazily constructed by GetLookup.
            private volatile int _maxCompLen;
            private volatile int _maxDecompLen;

            // Initially null, but lazily constructed by GetMetadataTOC.
            private volatile MetadataTableOfContentsEntry[] _metadataToc;

            // Initially null, but lazily constructed by GetMetadataTOC. This contains
            // the descriptions of uninterpretable metadata, akin to the _deadColumns
            // array in the loader.
            private volatile MetadataTableOfContentsEntry[] _deadMetadataToc;

            // Initially null, but lazily constructed by GetMetadataTOCEntryOrNull.
            private volatile Dictionary<string, MetadataTableOfContentsEntry> _metadataMap;

            private long _metadataTocEnd;

            /// <summary>
            /// Whether this is a generated column, that is, something dependent on no actual block data
            /// in the file.
            /// </summary>
            public bool IsGenerated { get { return ColumnIndex == -1; } }

            public TableOfContentsEntry(BinaryLoader parent, int index, string name, IValueCodec codec,
                CompressionKind compression, int rowsPerBlock, long lookupOffset, long metadataTocOffset)
            {
                Contracts.AssertValue(parent, "parent");
                Contracts.AssertValue(parent._host, "parent");
                _parent = parent;
                _ectx = _parent._host;

                _ectx.Assert(0 <= index && index < parent._header.ColumnCount);
                _ectx.AssertValue(name);
                _ectx.AssertValueOrNull(codec);
                _ectx.Assert(metadataTocOffset == 0 || Header.HeaderSize <= metadataTocOffset);
                // REVIEW: Should we allow lookup offset to be 0, if the binary file has no rows?
                _ectx.Assert(Header.HeaderSize <= lookupOffset);

                ColumnIndex = index;
                Name = name;
                Codec = codec;
                Type = Codec != null ? Codec.Type : null;
                Compression = compression;
                RowsPerBlock = rowsPerBlock;
                LookupOffset = lookupOffset;
                MetadataTocOffset = metadataTocOffset;

                _maxCompLen = -1;
                _maxDecompLen = -1;

                _ectx.Assert(!IsGenerated);
            }

            /// <summary>
            /// Constructor for a generated column, which corresponds to no column in the original file,
            /// and has no stored blocks associated with it. The input <paramref name="valueMapper"/> must
            /// be a <c>ValueMapper</c> mapping a <c>long</c> zero based row index, to some value with the
            /// same type as the raw type in <paramref name="type"/>.
            /// </summary>
            public TableOfContentsEntry(BinaryLoader parent, string name, ColumnType type, Delegate valueMapper)
            {
                Contracts.AssertValue(parent, "parent");
                Contracts.AssertValue(parent._host, "parent");
                _parent = parent;
                _ectx = _parent._host;

                _ectx.AssertValue(name);
                _ectx.AssertValue(type);
                _ectx.AssertValue(valueMapper);

                ColumnIndex = -1;
                Name = name;
                Type = type;
                _generatorDelegate = valueMapper;

                _maxCompLen = 0;
                _maxDecompLen = 0;

                _ectx.Assert(IsGenerated);
#if DEBUG
                Action del = AssertGeneratorValid<int>;
                var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(Type.RawType);
                meth.Invoke(this, null);
#endif
            }

#if DEBUG
            private void AssertGeneratorValid<T>()
            {
                _ectx.Assert(IsGenerated);
                _ectx.AssertValue(_generatorDelegate);
                _ectx.AssertValue(Type);
                ValueMapper<long, T> del = _generatorDelegate as ValueMapper<long, T>;
                _ectx.AssertValue(del);
            }
#endif

            /// <summary>
            /// Returns the value mapper for a generated column. Only a valid call if
            /// <typeparamref name="T"/> is the same type as <see cref="ColumnType.RawType"/>.
            /// </summary>
            public ValueMapper<long, T> GetValueMapper<T>()
            {
                _ectx.Assert(IsGenerated);
                _ectx.Assert(typeof(T) == Type.RawType);
                return (ValueMapper<long, T>)_generatorDelegate;
            }

            /// <summary>
            /// Gets an array, one for each block of this column, describing its location within the file.
            /// This will return null if and only if this is a generated column.
            /// </summary>
            public BlockLookup[] GetLookup()
            {
                _ectx.Assert(!IsGenerated == (LookupOffset > 0));
                if (LookupOffset > 0 && _maxCompLen == -1)
                {
                    Stream stream = _parent._stream;
                    lock (stream)
                    {
                        if (_maxCompLen == -1)
                        {
                            long rc = _parent._header.RowCount;
                            if (rc == 0)
                                return _lookup = new BlockLookup[0];
                            long numBlocks = (rc - 1) / RowsPerBlock + 1;
                            // By the format it's perfectly legal, but we don't yet support reading so many blocks.
                            if (numBlocks > int.MaxValue)
                                throw _ectx.ExceptNotSupp("This version of the software does not support {0} blocks", numBlocks);
                            var lookup = new BlockLookup[numBlocks];
                            stream.Seek(LookupOffset, SeekOrigin.Begin);
                            var reader = _parent._reader;
                            int maxCompLen = 0;
                            int maxDecompLen = 0;
                            for (int b = 0; b < numBlocks; ++b)
                            {
                                long offset = reader.ReadInt64();
                                int compLen = reader.ReadInt32();
                                int decompLen = reader.ReadInt32();
                                _ectx.CheckDecode(0 <= compLen, "negative compressed block length detected");
                                _ectx.CheckDecode(0 <= decompLen, "negative decompressed block length detected");
                                if (maxCompLen < compLen)
                                    maxCompLen = compLen;
                                if (maxDecompLen < decompLen)
                                    maxDecompLen = decompLen;
                                _ectx.CheckDecode(Header.HeaderSize <= offset && offset <= _parent._header.TailOffset - compLen, "block offset out of range");
                                lookup[b] = new BlockLookup(offset, compLen, decompLen);
                            }
                            _lookup = lookup;
                            _metadataTocEnd = stream.Position;
                            // Assign in this order since tests of validity are on comp len.
                            _maxDecompLen = maxDecompLen;
                            _maxCompLen = maxCompLen;
                        }
                    }
                    _ectx.AssertValue(_lookup);
                }
                _ectx.Assert(_maxDecompLen >= 0);
                _ectx.Assert(_maxCompLen >= 0);
                return _lookup;
            }

            /// <summary>
            /// Fetches the maximum block sizes for both the compressed and decompressed
            /// block sizes, for this column. If there are no blocks associated with this
            /// column, for whatever reason (for example, a data view with no rows, or a generated
            /// column), this will return 0 in both vlaues.
            /// </summary>
            /// <param name="compressed">The maximum value of the compressed block size
            /// (that is, the actual size of the block in stream) among all blocks for this
            /// column</param>
            /// <param name="decompressed">The maximum value of the block size when
            /// decompressed among all blocks for this column</param>
            public void GetMaxBlockSizes(out int compressed, out int decompressed)
            {
                if (_maxCompLen == -1)
                    GetLookup();
                _ectx.Assert(0 <= _maxCompLen);
                _ectx.Assert(0 <= _maxDecompLen);
                compressed = _maxCompLen;
                decompressed = _maxDecompLen;
            }

            private void EnsureMetadataStructuresInitialized()
            {
                if (MetadataTocOffset <= 0 || _metadataToc != null)
                    return;

                Stream stream = _parent._stream;
                lock (stream)
                {
                    if (_metadataToc != null)
                        return;

                    using (var ch = _parent._host.Start("Metadata TOC Read"))
                    {
                        ReadTocMetadata(ch, stream);
                        ch.Done();
                    }
                }
            }

            private void ReadTocMetadata(IChannel ch, Stream stream)
            {
                _ectx.AssertValue(ch);
                ch.AssertValue(stream);

                stream.Seek(MetadataTocOffset, SeekOrigin.Begin);
                var reader = _parent._reader;

                ulong mtocCount = reader.ReadLeb128Int();
                ch.CheckDecode(0 < mtocCount && mtocCount < int.MaxValue,
                    "Bad number of metadata TOC entries read");
                var mtocEntries = new List<MetadataTableOfContentsEntry>();
                var deadMtocEntries = new List<MetadataTableOfContentsEntry>();
                var map = new Dictionary<string, MetadataTableOfContentsEntry>();
                // This may have more entries than map if some metadata blocks are uninterpretable.
                var kinds = new HashSet<string>();

                for (int i = 0; i < (int)mtocCount; ++i)
                {
                    string kind = reader.ReadString();
                    ch.CheckDecode(!string.IsNullOrEmpty(kind), "Metadata kind must be non-empty string");
                    ch.CheckDecode(kinds.Add(kind), "Duplicate metadata kind read from file");
                    IValueCodec codec;
                    bool gotCodec = _parent._factory.TryReadCodec(stream, out codec);
                    // Even in the case where we did not succeed loading the codec, we still
                    // want to skip to the next table of contents entry, so keep reading.
                    CompressionKind compression = (CompressionKind)reader.ReadByte();
                    bool knowCompression = Enum.IsDefined(typeof(CompressionKind), compression);
                    long blockOffset = reader.ReadInt64();
                    ch.CheckDecode(Header.HeaderSize <= blockOffset && blockOffset <= _parent._header.TailOffset,
                        "Metadata block offset out of range");
                    ulong ublockSize = reader.ReadLeb128Int();
                    ch.CheckDecode(ublockSize <= long.MaxValue, "Metadata block size out of range");
                    long blockSize = (long)ublockSize;
                    ch.CheckDecode(0 < blockSize && blockSize <= _parent._header.TailOffset - blockOffset,
                        "Metadata block size out of range");
                    if (gotCodec && knowCompression)
                    {
                        var entry = MetadataTableOfContentsEntry.Create(_parent, kind, codec, compression, blockOffset, blockSize);
                        mtocEntries.Add(entry);
                        map[kind] = entry;
                    }
                    else
                    {
                        ch.Warning("Cannot interpret metadata of kind '{0}' because {1} unrecognized",
                            gotCodec ? "compression" : (knowCompression ? "codec" : "codec and compression"));
                        var entry = MetadataTableOfContentsEntry.CreateDead(_parent, kind, codec, compression, blockOffset, blockSize);
                        deadMtocEntries.Add(entry);
                    }
                }
                // It is possible for this to be empty but non-null if the associated codec
                // or compression schemes for all pieces of metadata is unknown. We want to
                // keep it empty but non-null so we can distinguish between the two cases of
                // "couldn't read anything" vs. "didn't read anything," lest we attempt to
                // re-read the metadata TOC on every request.
                _metadataToc = mtocEntries.ToArray();
                _deadMetadataToc = deadMtocEntries.ToArray();
                _metadataMap = map;
                _metadataTocEnd = stream.Position;
            }

            /// <summary>
            /// Gets an array containing the metadata TOC entries. This will return null if there
            /// are no entries stored at all, and empty if there is metadata, but none of it was
            /// readable. (To inspect attributes of the unreadable metadata, if any, see
            /// <see cref="GetDeadMetadataTocArray"/>.) All entries will point to metadata with
            /// known codecs and compression schemes.
            /// </summary>
            public MetadataTableOfContentsEntry[] GetMetadataTocArray()
            {
                EnsureMetadataStructuresInitialized();
                _ectx.Assert((MetadataTocOffset == 0) == (_metadataToc == null));
                _ectx.Assert((MetadataTocOffset == 0) == (_deadMetadataToc == null));
                return _metadataToc;
            }

            /// <summary>
            /// Gets an array containing the metadata TOC entries for all "dead" pieces of metadata. This
            /// will return null if there are no metadata stored at all either readable or unreadable, and
            /// empty if there is no unreadable piece of metadata. A piece of metadata is considered "dead"
            /// if either its codec or compression kind is unknown. This is primarily for diagnostic purposes.
            /// </summary>
            public MetadataTableOfContentsEntry[] GetDeadMetadataTocArray()
            {
                EnsureMetadataStructuresInitialized();
                _ectx.Assert((MetadataTocOffset == 0) == (_metadataToc == null));
                _ectx.Assert((MetadataTocOffset == 0) == (_deadMetadataToc == null));
                return _deadMetadataToc;
            }

            /// <summary>
            /// Returns the entry for a valid "live" piece of metadata given a kind.
            /// </summary>
            public MetadataTableOfContentsEntry GetMetadataTocEntryOrNull(string kind)
            {
                _ectx.AssertNonEmpty(kind);
                EnsureMetadataStructuresInitialized();
                if (_metadataMap == null)
                {
                    _ectx.Assert(MetadataTocOffset == 0);
                    return null;
                }
                MetadataTableOfContentsEntry retval;
                _metadataMap.TryGetValue(kind, out retval);
                return retval;
            }

            /// <summary>
            /// Returns the location in the stream just past the end of the metadata table of contents.
            /// If this column has no metadata table of contents defined, this will return 0. This is
            /// primarily for diagnostic purposes.
            /// </summary>
            /// <returns></returns>
            public long GetMetadataTocEndOffset()
            {
                EnsureMetadataStructuresInitialized();
                _ectx.Assert((_metadataTocEnd == 0) == (MetadataTocOffset == 0));
                return _metadataTocEnd;
            }
        }

        /// <summary>
        /// A column can be associated with metadata, in which case it will have one or more table of contents entries,
        /// each represented by one of these entries.
        /// </summary>
        private abstract class MetadataTableOfContentsEntry
        {
            /// <summary>
            /// The kind of the metadata, an identifying name.
            /// </summary>
            public readonly string Kind;

            /// <summary>
            /// The codec we will use to read the metadata value. If this is <c>null</c>,
            /// the metadata is considered "dead," that is, uninterpretable.
            /// </summary>
            public abstract IValueCodec Codec { get; }

            /// <summary>
            /// The compression scheme used on the metadata block. If this is an unknown
            /// type, the metadata is considered "dead," that is, uninterpretable.
            /// </summary>
            public readonly CompressionKind Compression;

            /// <summary>
            /// The offset into the stream where the metadata block begins.
            /// </summary>
            public readonly long BlockOffset;

            /// <summary>
            /// The number of bytes used to store the metadata block.
            /// </summary>
            public readonly long BlockSize;

            protected readonly BinaryLoader Parent;

            protected MetadataTableOfContentsEntry(BinaryLoader parent, string kind,
                CompressionKind compression, long blockOffset, long blockSize)
            {
                Contracts.AssertValue(parent, "Parent");
                Contracts.AssertValue(parent._host, "parent");
                Contracts.AssertNonEmpty(kind);
                Contracts.Assert(Header.HeaderSize <= blockOffset);
                Contracts.Assert(0 <= blockSize);

                Parent = parent;
                Kind = kind;
                Compression = compression;
                BlockOffset = blockOffset;
                BlockSize = blockSize;
            }

            public static MetadataTableOfContentsEntry Create(BinaryLoader parent, string kind, IValueCodec codec,
                CompressionKind compression, long blockOffset, long blockSize)
            {
                Contracts.AssertValue(parent, "parent");
                Contracts.AssertValue(parent._host, "parent");
                IExceptionContext ectx = parent._host;
                ectx.AssertValue(codec);
                ectx.Assert(Enum.IsDefined(typeof(CompressionKind), compression));

                var type = codec.Type;
                Type entryType;
                if (type.IsVector)
                {
                    Type valueType = type.RawType;
                    ectx.Assert(valueType.IsGenericEx(typeof(VBuffer<>)));
                    Type[] args = valueType.GetGenericArguments();
                    ectx.Assert(args.Length == 1);
                    entryType = typeof(MetadataTableOfContentsEntry.ImplVec<>).MakeGenericType(args);
                }
                else
                {
                    entryType = typeof(MetadataTableOfContentsEntry.ImplOne<>).MakeGenericType(type.RawType);
                }
                var result = (MetadataTableOfContentsEntry)Activator.CreateInstance(entryType,
                    parent, kind, codec, compression, blockOffset, blockSize);
                ectx.AssertValue(result.Codec);
                return result;
            }

            public static MetadataTableOfContentsEntry CreateDead(BinaryLoader parent, string kind, IValueCodec codec,
                CompressionKind compression, long blockOffset, long blockSize)
            {
                // We should be creating "dead" metadata only if we either couldn't interpret the codec,
                // or the compression kind, I should expect.
                Contracts.Assert((codec == null) || !Enum.IsDefined(typeof(CompressionKind), compression));
                return new ImplDead(parent, kind, codec, compression, blockOffset, blockSize);
            }

            /// <summary>
            /// Information on a metadata that could not be interpreted for some reason.
            /// </summary>
            private sealed class ImplDead : MetadataTableOfContentsEntry
            {
                private readonly IValueCodec _codec;

                public override IValueCodec Codec { get { return _codec; } }

                public ImplDead(BinaryLoader parent, string kind, IValueCodec codec,
                    CompressionKind compression, long blockOffset, long blockSize)
                    : base(parent, kind, compression, blockOffset, blockSize)
                {
                    _codec = codec;
                }
            }

            private sealed class ImplOne<T> : MetadataTableOfContentsEntry<T>
            {
                public ImplOne(BinaryLoader parent, string kind, IValueCodec<T> codec,
                    CompressionKind compression, long blockOffset, long blockSize)
                    : base(parent, kind, codec, compression, blockOffset, blockSize)
                {
                }

                public override void Get(ref T value)
                {
                    EnsureValue();
                    value = Value;
                }
            }

            private sealed class ImplVec<T> : MetadataTableOfContentsEntry<VBuffer<T>>
            {
                public ImplVec(BinaryLoader parent, string kind, IValueCodec<VBuffer<T>> codec,
                    CompressionKind compression, long blockOffset, long blockSize)
                    : base(parent, kind, codec, compression, blockOffset, blockSize)
                {
                }

                public override void Get(ref VBuffer<T> value)
                {
                    EnsureValue();
                    Value.CopyTo(ref value);
                }
            }
        }

        private abstract class MetadataTableOfContentsEntry<T> : MetadataTableOfContentsEntry
        {
            private bool _fetched;
            private readonly IValueCodec<T> _codec;
            protected T Value;

            public override IValueCodec Codec { get { return _codec; } }

            protected MetadataTableOfContentsEntry(BinaryLoader parent, string kind, IValueCodec<T> codec,
                CompressionKind compression, long blockOffset, long blockSize)
                : base(parent, kind, compression, blockOffset, blockSize)
            {
                // REVIEW: Do we want to have the capability to track "dead" pieces of
                // metadata, that is, metadata for which we have an unrecognized codec?
                Contracts.AssertValue(codec);
                _codec = codec;
            }

            protected void EnsureValue()
            {
                if (!_fetched)
                {
                    Stream stream = Parent._stream;
                    lock (stream)
                    {
                        if (!_fetched)
                        {
                            stream.Seek(BlockOffset, SeekOrigin.Begin);
                            using (var subset = new SubsetStream(stream, BlockSize))
                            using (var decompressed = Compression.DecompressStream(subset))
                            using (var valueReader = _codec.OpenReader(decompressed, 1))
                            {
                                valueReader.MoveNext();
                                valueReader.Get(ref Value);
                            }
                            _fetched = true;
                        }
                    }
                }
            }

            public abstract void Get(ref T value);
        }

        private sealed class SchemaImpl : ISchema
        {
            private readonly TableOfContentsEntry[] _toc;
            private readonly Dictionary<string, int> _name2col;
            private readonly IExceptionContext _ectx;

            public SchemaImpl(BinaryLoader parent)
            {
                Contracts.AssertValue(parent, "parent");
                Contracts.AssertValue(parent._host, "parent");
                _ectx = parent._host;

                _name2col = new Dictionary<string, int>();
                _toc = parent._aliveColumns;
                for (int c = 0; c < _toc.Length; ++c)
                    _name2col[_toc[c].Name] = c;
            }

            public int ColumnCount { get { return _toc.Length; } }

            public bool TryGetColumnIndex(string name, out int col)
            {
                _ectx.CheckValueOrNull(name);
                if (name == null)
                {
                    col = default(int);
                    return false;
                }
                return _name2col.TryGetValue(name, out col);
            }

            public string GetColumnName(int col)
            {
                _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return _toc[col].Name;
            }

            public ColumnType GetColumnType(int col)
            {
                _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return _toc[col].Type;
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                var metadatas = _toc[col].GetMetadataTocArray();
                if (Utils.Size(metadatas) > 0)
                    return metadatas.Select(e => new KeyValuePair<string, ColumnType>(e.Kind, e.Codec.Type));
                return Enumerable.Empty<KeyValuePair<string, ColumnType>>();
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                _ectx.CheckNonEmpty(kind, nameof(kind));
                _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                var entry = _toc[col].GetMetadataTocEntryOrNull(kind);
                return entry == null ? null : entry.Codec.Type;
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                _ectx.CheckNonEmpty(kind, nameof(kind));
                _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));

                var entry = _toc[col].GetMetadataTocEntryOrNull(kind) as MetadataTableOfContentsEntry<TValue>;
                if (entry == null)
                    throw MetadataUtils.ExceptGetMetadata();
                entry.Get(ref value);
            }
        }

        private readonly Stream _stream;
        private readonly BinaryReader _reader;
        private readonly CodecFactory _factory;
        private readonly Header _header;
        private readonly SchemaImpl _schema;
        private readonly bool _autodeterminedThreads;
        private readonly int _threads;
        private readonly string _generatedRowIndexName;
        private bool _disposed;

        private readonly TableOfContentsEntry[] _aliveColumns;
        // We still want to be able to access information about the columns we could not read, like their
        // name, where they are, how much space they're taking, etc. Conceivably for some operations (for example,
        // column filtering) whether or not we can interpret the values in the column is totally irrelevant.
        private readonly TableOfContentsEntry[] _deadColumns;

        // The number of rows per block. The format supports having different rows per block in each column,
        // but in software we do not yet support this.
        private readonly int _rowsPerBlock;

        // The stream offset at the end of the table of contents. This is useful for diagnostic purposes.
        private readonly long _tocEndLim;

        private readonly MemoryStreamCollection _bufferCollection;

        private readonly IHost _host;

        // The number of blocks worth of data to keep in the shuffle pool.
        private readonly Double _shuffleBlocks;
        // The actual more convenient number of rows to use in the pool, calculated from the shuffle
        // count. This is not serialized to the data model, since it depends on the block size
        // which can change from input data file to data file, as the same data model is applied to
        // different data files.
        private readonly int _randomShufflePoolRows;
        private const Double _defaultShuffleBlocks = 4;

        /// <summary>
        /// Upper inclusive bound of versions this reader can read.
        /// </summary>
        private const ulong ReaderVersion = StandardDataTypesVersion;

        /// <summary>
        /// The first version that removes DvTypes and uses .NET standard
        /// data types.
        /// </summary>
        private const ulong StandardDataTypesVersion = 0x0001000100010006;

        /// <summary>
        /// The first version of the format that accomodated DvText.NA.
        /// </summary>
        private const ulong MissingTextVersion = 0x0001000100010005;

        /// <summary>
        /// The first version of the format that accomodated arbitrary metadata.
        /// </summary>
        private const ulong MetadataVersion = 0x0001000100010004;

        /// <summary>
        /// The first version of the format that accomodated slot names.
        /// </summary>
        private const ulong SlotNamesVersion = 0x0001000100010003;

        /// <summary>
        /// Lower inclusive bound of versions this reader can read.
        /// </summary>
        private const ulong ReaderFirstVersion = 0x0001000100010002;

        public ISchema Schema { get { return _schema; } }

        private long RowCount { get { return _header.RowCount; } }

        public long? GetRowCount(bool lazy = true) { return RowCount; }

        public bool CanShuffle { get { return true; } }

        internal const string Summary = "Loads native Binary IDV data file.";
        internal const string LoadName = "BinaryLoader";

        internal const string LoaderSignature = "BinaryLoader";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "BINLOADR",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Generated row index column
                verWrittenCur: 0x00010003, // Number of blocks to put in the shuffle pool
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(BinaryLoader).Assembly.FullName);
        }

        private BinaryLoader(Arguments args, IHost host, Stream stream, bool leaveOpen)
        {
            Contracts.AssertValue(host, "host");
            _host = host;

            _host.CheckValue(args, nameof(args));
            _host.CheckValue(stream, nameof(stream));
            _host.CheckParam(stream.CanRead, nameof(stream), "input stream must be readable");
            _host.CheckParam(stream.CanSeek, nameof(stream), "input stream must be seekable");
            _host.CheckParam(stream.Position == 0, nameof(stream), "input stream must be at head");
            _host.CheckUserArg(0 <= args.PoolBlocks, nameof(args.PoolBlocks), "must be non-negative");

            using (var ch = _host.Start("Initializing"))
            {
                _stream = stream;
                _reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen);
                _factory = new CodecFactory(_host);

                _header = InitHeader();
                _autodeterminedThreads = args.Threads == null;
                _threads = Math.Max(1, args.Threads ?? (Environment.ProcessorCount / 2));
                _generatedRowIndexName = string.IsNullOrWhiteSpace(args.RowIndexName) ? null : args.RowIndexName;
                InitToc(ch, out _aliveColumns, out _deadColumns, out _rowsPerBlock, out _tocEndLim);
                _schema = new SchemaImpl(this);
                _host.Assert(_schema.ColumnCount == Utils.Size(_aliveColumns));
                _bufferCollection = new MemoryStreamCollection();
                if (Utils.Size(_deadColumns) > 0)
                    ch.Warning("BinaryLoader does not know how to interpret {0} columns", Utils.Size(_deadColumns));
                _shuffleBlocks = args.PoolBlocks;
                CalculateShufflePoolRows(ch, out _randomShufflePoolRows);
                ch.Done();
            }
        }

        /// <summary>
        /// Constructs a new data view reader.
        /// </summary>
        /// <param name="stream">A seekable, readable stream. Note that the data view reader assumes
        /// that it is the exclusive owner of this stream.</param>
        /// <param name="args">Arguments</param>
        /// <param name="env">Host enviroment</param>
        /// <param name="leaveOpen">Whether to leave the input stream open</param>
        public BinaryLoader(IHostEnvironment env, Arguments args, Stream stream, bool leaveOpen = true)
            : this(args, env.Register(LoadName), stream, leaveOpen)
        {
        }

        public BinaryLoader(IHostEnvironment env, Arguments args, string filename)
            : this(env, args, OpenStream(filename), leaveOpen: false)
        {
        }

        public BinaryLoader(IHostEnvironment env, Arguments args, IMultiStreamSource file)
            : this(env, args, OpenStream(file), leaveOpen: false)
        {
        }

        /// <summary>
        /// Creates a binary loader from a <see cref="ModelLoadContext"/>. Since the loader code
        /// opens the file, this will always take ownership of the stream, that is, this is always
        /// akin to <c>leaveOpen</c> in the other constructor being false.
        /// </summary>
        private BinaryLoader(IHost host, ModelLoadContext ctx, Stream stream)
        {
            Contracts.AssertValue(host, "host");
            _host = host;

            _host.AssertValue(ctx);
            _host.CheckValue(stream, nameof(stream));
            _host.CheckParam(stream.CanRead, nameof(stream), "input stream must be readable");
            _host.CheckParam(stream.CanSeek, nameof(stream), "input stream must be seekable");
            _host.CheckParam(stream.Position == 0, nameof(stream), "input stream must be at head");

            // *** Binary format **
            // int: Number of threads if explicitly defined, or 0 if the
            //      number of threads was automatically determined
            // int: Id of the generated row index name (can be null)

            using (var ch = _host.Start("Initializing"))
            {
                _stream = stream;
                if (ctx.Header.ModelVerWritten >= 0x00010002)
                {
                    _threads = ctx.Reader.ReadInt32();
                    ch.CheckDecode(_threads >= 0);
                    if (_threads == 0)
                    {
                        _autodeterminedThreads = true;
                        _threads = Math.Max(1, Environment.ProcessorCount / 2);
                    }

                    _generatedRowIndexName = ctx.LoadStringOrNull();
                    ch.CheckDecode(_generatedRowIndexName == null || !string.IsNullOrWhiteSpace(_generatedRowIndexName));
                }
                else
                {
                    _threads = Math.Max(1, Environment.ProcessorCount / 2);
                    _generatedRowIndexName = null;
                }

                if (ctx.Header.ModelVerWritten >= 0x00010003)
                {
                    _shuffleBlocks = ctx.Reader.ReadDouble();
                    ch.CheckDecode(0 <= _shuffleBlocks);
                }
                else
                    _shuffleBlocks = _defaultShuffleBlocks;

                _reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: false);
                _factory = new CodecFactory(_host);

                _header = InitHeader();
                InitToc(ch, out _aliveColumns, out _deadColumns, out _rowsPerBlock, out _tocEndLim);
                _schema = new SchemaImpl(this);
                ch.Assert(_schema.ColumnCount == Utils.Size(_aliveColumns));
                _bufferCollection = new MemoryStreamCollection();
                if (Utils.Size(_deadColumns) > 0)
                    ch.Warning("BinaryLoader does not know how to interpret {0} columns", Utils.Size(_deadColumns));

                CalculateShufflePoolRows(ch, out _randomShufflePoolRows);
                ch.Done();
            }
        }

        public static BinaryLoader Create(IHostEnvironment env, ModelLoadContext ctx, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(LoadName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(files, nameof(files));

            return h.Apply("Loading Model",
                ch =>
                {
                    if (files.Count == 0)
                    {
                        BinaryLoader retVal = null;
                        // In the case where we have no input streams, but we have an input schema from
                        // the model repository, we still want to surface ourselves as being a binary loader
                        // with the existing schema. The loader "owns" this stream.
                        if (ctx.TryLoadBinaryStream("Schema.idv",
                            r => retVal = new BinaryLoader(h, ctx, HybridMemoryStream.CreateCache(r.BaseStream))))
                        {
                            h.AssertValue(retVal);
                            h.CheckDecode(retVal.RowCount == 0);
                            // REVIEW: Do we want to be a bit more restrictive around uninterpretable columns?
                            return retVal;
                        }
                        h.Assert(retVal == null);
                        // Fall through, allow the failure to be on OpenStream.
                    }
                    return new BinaryLoader(h, ctx, OpenStream(files));
                });
        }

        /// <summary>
        /// Creates a binary loader from a stream that is not owned by the loader.
        /// This creates its own independent copy of input stream for the binary loader.
        /// </summary>
        public static BinaryLoader Create(IHostEnvironment env, ModelLoadContext ctx, Stream stream)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(LoadName);
            return new BinaryLoader(h, ctx, HybridMemoryStream.CreateCache(stream));
        }

        private static Stream OpenStream(IMultiStreamSource files)
        {
            Contracts.CheckValue(files, nameof(files));
            Contracts.CheckParam(files.Count == 1, nameof(files), "binary loader must be created with one file");
            return files.Open(0);
        }

        private static Stream OpenStream(string filename)
        {
            Contracts.CheckNonEmpty(filename, nameof(filename));
            var files = new MultiFileSource(filename);
            return OpenStream(files);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            _host.Assert(_threads >= 1);
            SaveParameters(ctx, _autodeterminedThreads ? 0 : _threads, _generatedRowIndexName, _shuffleBlocks);

            int[] unsavable;
            SaveSchema(_host, ctx, Schema, out unsavable);
            _host.Assert(Utils.Size(unsavable) == 0);
        }

        /// <summary>
        /// Write the parameters of a loader to the save context. Can be called by <see cref="SaveInstance"/>, where there's no actual
        /// loader, only default parameters.
        /// </summary>
        private static void SaveParameters(ModelSaveContext ctx, int threads, string generatedRowIndexName, Double shuffleBlocks)
        {
            // *** Binary format **
            // int: Number of threads if explicitly defined, or 0 if the
            //      number of threads was automatically determined
            // int: Id of the generated row index name (can be null)
            // Double: The randomness coefficient.

            Contracts.Assert(threads >= 0);
            ctx.Writer.Write(threads);
            Contracts.Assert(generatedRowIndexName == null || !string.IsNullOrWhiteSpace(generatedRowIndexName));
            ctx.SaveStringOrNull(generatedRowIndexName);
            Contracts.Assert(0 <= shuffleBlocks);
            ctx.Writer.Write(shuffleBlocks);
        }

        /// <summary>
        /// Save a zero-row dataview that will be used to infer schema information, used in the case
        /// where the binary loader is instantiated with no input streams.
        /// </summary>
        private static void SaveSchema(IHostEnvironment env, ModelSaveContext ctx, ISchema schema, out int[] unsavableColIndices)
        {
            Contracts.AssertValue(env, "env");
            var h = env.Register(LoadName);

            h.AssertValue(ctx);
            h.AssertValue(schema);

            var noRows = new EmptyDataView(h, schema);
            h.Assert(noRows.GetRowCount() == 0);

            var saverArgs = new BinarySaver.Arguments();
            saverArgs.Silent = true;
            var saver = new BinarySaver(env, saverArgs);

            var cols = Enumerable.Range(0, schema.ColumnCount)
                .Select(x => new { col = x, isSavable = saver.IsColumnSavable(schema.GetColumnType(x)) });
            int[] toSave = cols.Where(x => x.isSavable).Select(x => x.col).ToArray();
            unsavableColIndices = cols.Where(x => !x.isSavable).Select(x => x.col).ToArray();
            ctx.SaveBinaryStream("Schema.idv", w => saver.SaveData(w.BaseStream, noRows, toSave));
        }

        /// <summary>
        /// Given the schema and a model context, save an imaginary instance of a binary loader with the
        /// specified schema. Deserialization from this context should produce a real binary loader that
        /// has the specified schema.
        ///
        /// This is used in an API scenario, when the data originates from something other than a loader.
        /// Since our model file requires a loader at the beginning, we have to construct a bogus 'binary' loader
        /// to begin the pipe with, with the assumption that the user will bypass the loader at deserialization
        /// time by providing a starting data view.
        /// </summary>
        public static void SaveInstance(IHostEnvironment env, ModelSaveContext ctx, ISchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoadName);

            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(ctx, nameof(schema));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            SaveParameters(ctx, 0, null, _defaultShuffleBlocks);

            int[] unsavable;
            SaveSchema(env, ctx, schema, out unsavable);
            // REVIEW: we silently ignore unsavable columns.
            // This method is invoked only in an API scenario, where we need to save a loader but we only have a schema.
            // In this case, the API user is likely not subscribed to our environment's channels. Also, in this case, the presence of
            // unsavable columns is not necessarily a bad thing: the user typically provides his own data when loading the transforms,
            // thus bypassing the bogus loader.
        }

        private unsafe Header InitHeader()
        {
            byte[] headerBytes = new byte[Header.HeaderSize];
            int cb = _reader.Read(headerBytes, 0, Header.HeaderSize);
            if (cb != Header.HeaderSize)
            {
                throw _host.ExceptDecode("Read only {0} bytes in file, expected header size of {1}",
                    cb, Header.HeaderSize);
            }
            Header header;
            unsafe
            {
                Marshal.Copy(headerBytes, 0, (IntPtr)(&header), Header.HeaderSize);
            }

            // Validate the header before returning. CheckDecode is used for incorrect
            // formatting.

            _host.CheckDecode(header.Signature == Header.SignatureValue,
                "This does not appear to be a binary dataview file");

            // Obviously the compatibility version can't exceed the true version of the file.
            if (header.CompatibleVersion > header.Version)
            {
                throw _host.ExceptDecode("Compatibility version {0} cannot be greater than file version {1}",
                    Header.VersionToString(header.CompatibleVersion), Header.VersionToString(header.Version));
            }

            if (header.Version < ReaderFirstVersion)
            {
                throw _host.ExceptDecode("Unexpected version {0} encountered, earliest expected here was {1}",
                    Header.VersionToString(header.Version), Header.VersionToString(ReaderFirstVersion));
            }
            // Check the versions.
            if (header.CompatibleVersion < MetadataVersion)
            {
                // This is distinct from the earlier message semantically in that the check
                // against ReaderFirstVersion is an indication of format impurity, whereas this
                // is simply a matter of software support.
                throw _host.Except("Cannot read version {0} data, earliest that can be handled is {1}",
                    Header.VersionToString(header.CompatibleVersion), Header.VersionToString(MetadataVersion));
            }
            if (header.CompatibleVersion > ReaderVersion)
            {
                throw _host.Except("Cannot read version {0} data, latest that can be handled is {1}",
                    Header.VersionToString(header.CompatibleVersion), Header.VersionToString(ReaderVersion));
            }

            _host.CheckDecode(header.RowCount >= 0, "Row count cannot be negative");
            _host.CheckDecode(header.ColumnCount >= 0, "Column count cannot be negative");
            // Check the table of contents offset, though we do not at this time have the contents themselves.
            if (header.ColumnCount != 0 && header.TableOfContentsOffset < Header.HeaderSize)
                throw _host.ExceptDecode("Table of contents offset {0} less than header size, impossible", header.TableOfContentsOffset);

            // Check the tail signature.
            if (header.TailOffset < Header.HeaderSize)
                throw _host.ExceptDecode("Tail offset {0} less than header size, impossible", header.TailOffset);
            _stream.Seek(header.TailOffset, SeekOrigin.Begin);
            ulong tailSig = _reader.ReadUInt64();
            _host.CheckDecode(tailSig == Header.TailSignatureValue, "Incorrect tail signature");
            return header;
        }

        private void InitToc(IChannel ch, out TableOfContentsEntry[] aliveColumns, out TableOfContentsEntry[] deadColumns, out int allRowsPerBlock, out long tocEndOffset)
        {
            if (_header.ColumnCount > 0)
                _stream.Seek(_header.TableOfContentsOffset, SeekOrigin.Begin);
            // Failure to recognize a codec is not by itself an error condition. It only
            // means we cannot read the associated columns.
            List<TableOfContentsEntry> aliveList = new List<TableOfContentsEntry>();
            List<TableOfContentsEntry> deadList = new List<TableOfContentsEntry>();

            allRowsPerBlock = 0;
            for (int c = 0; c < _header.ColumnCount; ++c)
            {
                string name = _reader.ReadString();
                IValueCodec codec;
                bool gotCodec = _factory.TryReadCodec(_stream, out codec);
                // Even in the case where we did not succeed loading the codec, we still
                // want to skip to the next table of contents entry, so keep reading.
                CompressionKind compression = (CompressionKind)_reader.ReadByte();
                bool knowCompression = Enum.IsDefined(typeof(CompressionKind), compression);
                int rowsPerBlock = (int)_reader.ReadLeb128Int();
                // 0 is only a valid blocksize if there are no rows.
                if (!(0 < rowsPerBlock || (rowsPerBlock == 0 && _header.RowCount == 0)))
                    throw ch.ExceptDecode("Bad number of rows per block {0} read", rowsPerBlock);
                // Even though the format allows it, we do not (yet?) support different block sizes across columns.
                if (c == 0)
                    allRowsPerBlock = rowsPerBlock;
                else if (allRowsPerBlock != rowsPerBlock)
                {
                    throw ch.ExceptNotSupp("Different rows per block per column not supported yet, encountered {0} and {1}",
                        allRowsPerBlock, rowsPerBlock);
                }

                long lookupOffset = _reader.ReadInt64();
                if (_header.RowCount > 0)
                {
                    // What is the number of element in the lookup table?
                    long lookupLen = (_header.RowCount - 1) / rowsPerBlock + 1;
                    ch.CheckDecode(Header.HeaderSize <= lookupOffset && lookupOffset <= _header.TailOffset - 16 * lookupLen,
                        "Lookup table offset out of range");
                }
                long metadataTocOffset = _reader.ReadInt64();
                ch.CheckDecode(metadataTocOffset == 0 || Header.HeaderSize <= metadataTocOffset && metadataTocOffset <= _header.TailOffset,
                    "Metadata TOC offset out of range");
                var entry = new TableOfContentsEntry(this, c, name, codec,
                    compression, rowsPerBlock, lookupOffset, metadataTocOffset);
                if (gotCodec && knowCompression)
                    aliveList.Add(entry);
                else
                {
                    ch.Warning("Cannot interpret column '{0}' at index {1} because {2} unrecognized",
                        name, c,
                        gotCodec ? "compression" : (knowCompression ? "codec" : "codec and compression"));
                    deadList.Add(entry);
                }
            }
            tocEndOffset = _stream.Position;
            if (_generatedRowIndexName != null)
            {
                ch.Trace("Creating generated column to hold row index, named '{0}'", _generatedRowIndexName);
                aliveList.Add(CreateRowIndexEntry(_generatedRowIndexName));
            }
            aliveColumns = aliveList.ToArray();
            deadColumns = deadList.ToArray();
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                _reader.Dispose();
            }
        }

        private void CalculateShufflePoolRows(IChannel ch, out int poolRows)
        {
            if (!ShuffleTransform.CanShuffleAll(Schema))
            {
                // This will only happen if we expand the set of types we can serialize,
                // without expanding the set of types we can cache. That is entirely
                // possible, but is not true at the current time.
                ch.Warning("Not adding implicit shuffle, as we did not know how to copy some types of values");
                poolRows = 0;
            }
            var poolSize = Math.Ceiling(_shuffleBlocks * _rowsPerBlock);
            ch.Assert(poolSize >= 0);
            // A pool size of 0 or 1 is like having no pool at all.
            if (poolSize < 2)
            {
                ch.Trace("Not adding implicit shuffle, as it is unnecessary");
                poolRows = 0;
                return;
            }
            const int maxPoolSize = 1 << 28;
            if (poolSize > maxPoolSize)
                poolSize = maxPoolSize;
            if (poolSize > _header.RowCount)
                poolSize = _header.RowCount;
            poolRows = checked((int)poolSize);
            ch.Trace("Implicit shuffle will have pool size {0}", poolRows);
        }

        private TableOfContentsEntry CreateRowIndexEntry(string rowIndexName)
        {
            _host.Assert(!string.IsNullOrWhiteSpace(rowIndexName));
            // REVIEW: Having a row count of 0 means that there are no valid output key values here,
            // so this should be a key with *genuinely* a count of 0. However, a count of a key of 0 means
            // that the key length is unknown. Unsure of how to reconcile this. Is the least harmful thing
            // to do, if RowCount=0, to set count to some value like 1?
            int count = _header.RowCount <= int.MaxValue ? (int)_header.RowCount : 0;
            KeyType type = new KeyType(DataKind.U8, 0, count);
            // We are mapping the row index as expressed as a long, into a key value, so we must increment by one.
            ValueMapper<long, ulong> mapper = (ref long src, ref ulong dst) => dst = (ulong)(src + 1);
            var entry = new TableOfContentsEntry(this, rowIndexName, type, mapper);
            return entry;
        }

        private IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            if (rand != null && _randomShufflePoolRows > 0)
            {
                // Don't bother with block shuffling, if the shuffle cursor is just going to hold
                // the entire dataset in memory anyway.
                var ourRand = _randomShufflePoolRows == _header.RowCount ? null : rand;
                var cursor = new Cursor(this, predicate, ourRand);
                return ShuffleTransform.GetShuffledCursor(_host, _randomShufflePoolRows, cursor, rand);
            }
            return new Cursor(this, predicate, rand);
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            _host.CheckValue(predicate, nameof(predicate));
            _host.CheckValueOrNull(rand);
            return GetRowCursorCore(predicate, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            _host.CheckValue(predicate, nameof(predicate));
            _host.CheckValueOrNull(rand);
            consolidator = null;
            return new IRowCursor[] { GetRowCursorCore(predicate, rand) };
        }

        private sealed class Cursor : RootCursorBase, IRowCursor
        {
            private const string _badCursorState = "cursor is either not started or is ended, and cannot get values";

            private readonly BinaryLoader _parent;
            private readonly int[] _colToActivesIndex;
            private readonly TableOfContentsEntry[] _actives;
            private readonly int _numBlocks;
            private readonly int _rowsPerBlock;
            private readonly int _rowsInLastBlock;
            private readonly ReadPipe[] _pipes;
            private readonly Delegate[] _pipeGetters;
            private readonly long _lastValidCounter;
            // This may be null, in the event that we are not shuffling.
            private readonly int[] _blockShuffleOrder;

            private readonly Thread _readerThread;
            private readonly Task _pipeTask;
            private readonly ExceptionMarshaller _exMarshaller;

            private volatile bool _disposed;

            public ISchema Schema { get { return _parent.Schema; } }

            public override long Batch
            {
                // REVIEW: Implement cursor set support.
                get { return 0; }
            }

            public Cursor(BinaryLoader parent, Func<int, bool> predicate, IRandom rand)
                : base(parent._host)
            {
                _parent = parent;
                Ch.AssertValue(predicate);
                Ch.AssertValueOrNull(rand);

                SchemaImpl schema = _parent._schema;
                _exMarshaller = new ExceptionMarshaller();

                TableOfContentsEntry[] toc = _parent._aliveColumns;
                int[] activeIndices;
                Utils.BuildSubsetMaps(toc.Length, predicate, out activeIndices, out _colToActivesIndex);
                _actives = new TableOfContentsEntry[activeIndices.Length];
                for (int i = 0; i < activeIndices.Length; ++i)
                    _actives[i] = toc[activeIndices[i]];

                _lastValidCounter = _parent._header.RowCount - 1;

                // Set up those evil pipes.
                _pipes = new ReadPipe[parent.RowCount > 0 ? _actives.Length : 0];
                _pipeGetters = new Delegate[_actives.Length];

                // The uniformity of blocksize has already been checked during ToC read.
                // However, if we have only generated columns, then we will have no defined
                // block size. The case of only generated columns is perhaps not terribly
                // likely or useful, but it is *possible*.
                _rowsPerBlock = _parent._rowsPerBlock;
                if (_rowsPerBlock == 0)
                {
                    // This should happen if and only if all columns are generated.
                    // Just pick some value.
                    _rowsPerBlock = int.MaxValue;
                }
                _rowsInLastBlock = _parent.RowCount == 0 ? 0 : (int)(_parent.RowCount % _rowsPerBlock);
                if (_rowsInLastBlock == 0)
                    _rowsInLastBlock = _rowsPerBlock;

                _numBlocks = checked((int)((_parent.RowCount - 1) / _rowsPerBlock + 1));
                _blockShuffleOrder = rand == null || _numBlocks == 0 ? null : Utils.GetRandomPermutation(rand, _numBlocks);

                if (_pipes.Length == 0)
                {
                    // Even in the case where we have no rows, and pipes have not
                    // been created, we still need getter delegates. These won't do
                    // anything but complain about the cursor being in a bad state,
                    // but they still need to exist.
                    for (int c = 0; c < _pipeGetters.Length; ++c)
                        _pipeGetters[c] = GetNoRowGetter(_actives[c].Type);
                    return;
                }

                // The following initalized fields should be used only by code that
                // assumes we have some active columns and more than zero rows.

                // How many buffers per pipe? Figure something based on ceildiv(threads / pipes), plus
                // some thread wiggle room. (Double it, mayhap?)
                int pipeBuffers = 2 * ((_parent._threads + _pipes.Length - 1) / _pipes.Length);

                for (int c = 0; c < _pipes.Length; ++c)
                {
                    _pipes[c] = ReadPipe.Create(this, c, pipeBuffers);
                    _pipeGetters[c] = _pipes[c].GetGetter();
                }
                // The data structures are initialized. Now set up the workers.
                _readerThread = Utils.CreateBackgroundThread(ReaderWorker);
                _readerThread.Start();

                _pipeTask = SetupDecompressTask();
            }

            public override void Dispose()
            {
                if (!_disposed && _readerThread != null)
                {
                    // We should reach this block only in the event of a dispose
                    // before all rows have been iterated upon.

                    // First set the flag on the cursor. The stream-reader and the
                    // pipe-decompressor workers will detect this, stop their work,
                    // and do whatever "cleanup" is natural for them to perform.
                    _disposed = true;

                    // In the disk read -> decompress -> codec read pipeline, we
                    // clean up in reverse order.
                    // 1. First we clear out any pending codec readers, for each pipe.
                    // 2. Then we join the pipe worker threads, which in turn should
                    // have cleared out all of the pending blocks to decompress.
                    // 3. Then finally we join against the reader thread.

                    // This code is analogous to the stuff in MoveNextCore, except
                    // nothing is actually done with the resulting blocks.

                    try
                    {
                        for (; ; )
                        {
                            // This cross-block-index access pattern is deliberate, as
                            // by having a consistent access pattern everywhere we can
                            // have much greater confidence this will never deadlock.
                            bool anyTrue = false;
                            for (int c = 0; c < _pipes.Length; ++c)
                                anyTrue |= _pipes[c].MoveNextCleanup();
                            if (!anyTrue)
                                break;
                        }
                    }
                    catch (OperationCanceledException ex)
                    {
                        // REVIEW: Encountering this here means that we did not encounter
                        // the exception during normal cursoring, but at some later point. I feel
                        // we should not be tolerant of this, and should throw, though it might be
                        // an ambiguous point.
                        Contracts.Assert(ex.CancellationToken == _exMarshaller.Token);
                        _exMarshaller.ThrowIfSet(Ch);
                        Contracts.Assert(false);
                    }
                    finally
                    {
                        _pipeTask.Wait();
                        _readerThread.Join();
                    }
                }

                base.Dispose();
            }

            private Task SetupDecompressTask()
            {
                Thread[] pipeWorkers = new Thread[_parent._threads];
                long decompressSequence = -1;
                long decompressSequenceLim = (long)_numBlocks * _actives.Length;
                for (int w = 0; w < pipeWorkers.Length; ++w)
                {
                    Thread worker = pipeWorkers[w] = Utils.CreateBackgroundThread(() =>
                    {
                        try
                        {
                            for (; ; )
                            {
                                long seq = Interlocked.Increment(ref decompressSequence);
                                int pipeIndex = (int)(seq % _pipes.Length);
                                // If we ever return false, then we know we are past the block sequence
                                // with all the sentinel blocks. Since we are kicking off all blocks in
                                // order, then we know that all the sentinel block handling has been
                                // handled or is in the process of being handled by some worker, so we
                                // may safely exit.
                                if (!_pipes[pipeIndex].DecompressOne())
                                    return;
                            }
                        }
                        catch (Exception ex)
                        {
                            _exMarshaller.Set("decompressing", ex);
                        }
                    });
                    worker.Start();
                }
                Task pipeTask = new Task(() =>
                {
                    foreach (Thread worker in pipeWorkers)
                        worker.Join();
                });
                pipeTask.Start();
                return pipeTask;
            }

            private void ReaderWorker()
            {
                try
                {
                    int blockSteps = checked((int)((_parent.RowCount - 1) / _rowsPerBlock + 1));

                    Stream stream = _parent._stream;
                    int b;
                    for (b = 0; b < blockSteps && !_disposed; ++b)
                    {
                        int bi = _blockShuffleOrder == null ? b : _blockShuffleOrder[b];
                        int rows = bi == blockSteps - 1 ? _rowsInLastBlock : _rowsPerBlock;
                        for (int c = 0; c < _pipes.Length; ++c)
                            _pipes[c].PrepAndSendCompressedBlock(bi, b, rows);
                    }
                    // Add the end sentinel blocks, for all pipes, guaranteeing the useful simplifying
                    // invariant that that all sentinel blocks have the same block sequence index.
                    for (int c = 0; c < _pipes.Length; ++c)
                        _pipes[c].SendSentinelBlock(b);
                }
                catch (Exception ex)
                {
                    _exMarshaller.Set("reading", ex);
                }
            }

            private abstract class ReadPipe
            {
                protected readonly int ColumnIndex;
                protected readonly Cursor Parent;

                protected ExceptionMarshaller ExMarshaller { get { return Parent._exMarshaller; } }

                protected IExceptionContext Ectx { get { return Parent.Ch; } }

                public static ReadPipe Create(Cursor parent, int columnIndex, int bufferSize)
                {
                    Contracts.AssertValue(parent);
                    var entry = parent._actives[columnIndex];
                    Contracts.AssertValue(entry);
                    Type genType = entry.IsGenerated ? typeof(ReadPipeGenerated<>) : typeof(ReadPipe<>);
                    genType = genType.MakeGenericType(entry.Type.RawType);
                    return (ReadPipe)Activator.CreateInstance(
                        genType, parent, columnIndex, bufferSize);
                }

                protected ReadPipe(Cursor parent, int columnIndex)
                {
                    Contracts.AssertValue(parent);
                    Parent = parent;
                    Ectx.Assert(0 <= columnIndex & columnIndex < Utils.Size(parent._actives));
                    ColumnIndex = columnIndex;
                }

                public abstract void PrepAndSendCompressedBlock(long blockIndex, long blockSequence, int rowCount);

                public abstract void SendSentinelBlock(long blockSequence);

                /// <summary>
                /// This will attempt to extract a compressed block from the
                /// <see cref="ReadPipe{T}._toDecompress"/> queue. This returns true if and only if it
                /// succeeded in extracting an item from the queue (even a sentinel block);
                /// that is, if it returns false, then there are no more items to extract
                /// (though, continuing to call this method is entirely possible, and legal,
                /// if convenient).
                /// </summary>
                public abstract bool DecompressOne();

                public abstract bool MoveNext();

                /// <summary>
                /// Necessary to be called in the event of a premature exiting. This executes
                /// the same recycle-fetch block cycle as <see cref="MoveNext"/>, except that
                /// nothing is actually done with the resulting block. This should be called
                /// in a similar fashion as the cursor calls <see cref="MoveNext"/>.
                /// </summary>
                public abstract bool MoveNextCleanup();

                public abstract Delegate GetGetter();
            }

            private sealed class ReadPipeGenerated<T> : ReadPipe
            {
                // REVIEW: I cleave to the invariants and behavior of the stream-based readpipe, even
                // though in this generated case managing the memory buffers is not an issue. What is more
                // important than efficiency in this generated case, however, is maintaining the invariants
                // of thread and blocking behavior upon which we rely for our understanding of the lack of
                // deadlock. Nevertheless, at some point it may be valuable to see if there are some harmless
                // divergences from the stream-based buffer and block model. However, this should be undertaken
                // with great care. It would also be nice to see if, so long as we're enforcing consistency,
                // there's some way we can share more code.

                private const int _bufferSize = 4;
                private readonly BlockingCollection<Block> _toDecompress;
                private readonly IEnumerator<Block> _toDecompressEnumerator;
                private readonly BlockingCollection<Block> _toRead;
                private readonly IEnumerator<Block> _toReadEnumerator;

                // The waiter on insertions to toRead. Any "add" or "complete adding" must depend on this waiter.
                private readonly OrderedWaiter _waiter;
                private readonly ValueMapper<long, T> _mapper;

                private Block _curr;
                private int _remaining;

                private sealed class Block
                {
                    public readonly long BlockSequence;
                    public readonly long RowIndexMin;
                    public readonly long RowIndexLim;

                    /// <summary>
                    /// This indicates that this block does not contain any actual information, or
                    /// correspond to an actual block, but it will still contain the
                    /// <see cref="BlockSequence"/> index. Sentinel blocks are used to indicate that
                    /// there will be no more blocks to be decompressed along a particular pipe,
                    /// allowing the pipe worker to perform necessary cleanup.
                    /// </summary>
                    public bool IsSentinel { get { return RowIndexMin == -1; } }

                    public int Rows { get { return (int)(RowIndexLim - RowIndexMin); } }

                    public Block(long blockSequence, long min, long lim)
                    {
                        Contracts.Assert(blockSequence >= 0);
                        Contracts.Assert(0 <= min & min <= lim);
                        Contracts.Assert(lim - min <= int.MaxValue);
                        BlockSequence = blockSequence;
                        RowIndexMin = min;
                        RowIndexLim = lim;
                        Contracts.Assert(!IsSentinel);
                    }

                    /// <summary>
                    /// Constructor for a sentinel compressed block. (For example,
                    /// the pipe's last block, which contains no valid data.)
                    /// </summary>
                    public Block(long blockSequence)
                    {
                        Contracts.Assert(blockSequence >= 0);
                        BlockSequence = blockSequence;
                        RowIndexMin = RowIndexLim = -1;
                        Contracts.Assert(IsSentinel);
                    }
                }

                public ReadPipeGenerated(Cursor parent, int columnIndex, int bufferSize)
                    : base(parent, columnIndex)
                {
                    Contracts.AssertValue(parent);
                    Contracts.AssertValue(parent.Ch);

                    TableOfContentsEntry entry = parent._actives[ColumnIndex];
                    Ectx.AssertValue(entry);
                    Ectx.Assert(entry.IsGenerated);

                    _toDecompress = new BlockingCollection<Block>(_bufferSize);

                    Ectx.Assert(bufferSize > 0);

                    _toDecompressEnumerator = _toDecompress.GetConsumingEnumerable(ExMarshaller.Token).GetEnumerator();

                    _toRead = new BlockingCollection<Block>(bufferSize);
                    _toReadEnumerator = _toRead.GetConsumingEnumerable(ExMarshaller.Token).GetEnumerator();
                    _waiter = new OrderedWaiter();

                    _mapper = entry.GetValueMapper<T>();
                }

                public override void PrepAndSendCompressedBlock(long blockIndex, long blockSequence, int rowCount)
                {
                    long rowLim = blockIndex * Parent._rowsPerBlock;
                    var block = new Block(blockSequence, rowLim, rowLim + rowCount);
                    _toDecompress.Add(block, ExMarshaller.Token);
                }

                public override void SendSentinelBlock(long blockSequence)
                {
                    Block sentBlock = new Block(blockSequence);
                    _toDecompress.Add(sentBlock, ExMarshaller.Token);
                    _toDecompress.CompleteAdding();
                }

                public override bool DecompressOne()
                {
                    Block block;
                    lock (_toDecompressEnumerator)
                    {
                        if (!_toDecompressEnumerator.MoveNext())
                            return false;
                        block = _toDecompressEnumerator.Current;
                    }

                    Ectx.Assert(!_toRead.IsAddingCompleted);
                    if (block.IsSentinel)
                    {
                        _waiter.Wait(block.BlockSequence, ExMarshaller.Token);
                        _toRead.CompleteAdding();
                        _waiter.Increment();
                        return true;
                    }

                    if (Parent._disposed)
                    {
                        _waiter.Wait(block.BlockSequence, ExMarshaller.Token);
                        _waiter.Increment();
                        return true;
                    }

                    _waiter.Wait(block.BlockSequence, ExMarshaller.Token);
                    _toRead.Add(block, ExMarshaller.Token);
                    _waiter.Increment();
                    return true;
                    // This code mirrors that within the stream-based read pipe, except it has nothing to dispose.
                }

                public override bool MoveNext()
                {
                    Ectx.Assert(_remaining >= 0);
                    Ectx.Assert(_remaining == 0 || _curr != null);
                    if (_remaining == 0)
                    {
                        if (_curr != null)
                            _curr = null;
                        if (!_toReadEnumerator.MoveNext())
                            return false;
                        _curr = _toReadEnumerator.Current;
                        Ectx.AssertValue(_curr);
                        _remaining = _curr.Rows;
                    }
                    Ectx.Assert(_remaining > 0);
                    _remaining--;
                    return true;
                }

                public override bool MoveNextCleanup()
                {
                    // This is analogous to the _remaining == 0 part of
                    // MoveNext, except we don't actually do anything with
                    // the block we fetch.
                    if (_curr != null)
                        _curr = null;
                    if (!_toReadEnumerator.MoveNext())
                        return false;
                    _curr = _toReadEnumerator.Current;
                    return true;
                }

                private void Get(ref T value)
                {
                    Ectx.Check(_curr != null, _badCursorState);
                    long src = _curr.RowIndexLim - _remaining - 1;
                    _mapper(ref src, ref value);
                }

                public override Delegate GetGetter()
                {
                    ValueGetter<T> getter = Get;
                    return getter;
                }
            }

            private sealed class ReadPipe<T> : ReadPipe
            {
                private const int _bufferSize = 4;
                private readonly BlockLookup[] _lookup;
                private readonly Stream _stream;
                private readonly MemoryStreamPool _compPool;
                private readonly MemoryStreamPool _decompPool;
                /// <summary>
                /// Calls from the stream reader worker into <see cref="PrepAndSendCompressedBlock"/> will feed
                /// into this collection, and calls from the decompress worker into <see cref="DecompressOne"/>
                /// will consume this collection.
                /// </summary>
                private readonly BlockingCollection<CompressedBlock> _toDecompress;
                private readonly IEnumerator<CompressedBlock> _toDecompressEnumerator;
                private readonly BlockingCollection<ReaderContainer> _toRead;
                private readonly IEnumerator<ReaderContainer> _toReadEnumerator;
                private readonly IValueCodec<T> _codec;
                private readonly CompressionKind _compression;
                // The waiter on insertions to toRead. Any "add" or "complete adding" must depend on this waiter.
                private readonly OrderedWaiter _waiter;

                private ReaderContainer _curr;
                private int _remaining;

                private sealed class CompressedBlock
                {
                    public readonly MemoryStream Buffer;
                    public readonly int DecompressedLength;
                    public readonly long BlockIndex;
                    public readonly long BlockSequence;
                    public readonly int Rows;

                    /// <summary>
                    /// This indicates that this block does not contain any actual information, or
                    /// correspond to an actual block, but it will still contain the
                    /// <see cref="BlockSequence"/> index. Sentinel blocks are used to indicate that
                    /// there will be no more blocks to be decompressed along a particular pipe,
                    /// allowing the pipe worker to perform necessary cleanup.
                    /// </summary>
                    public bool IsSentinel { get { return BlockIndex == -1; } }

                    public CompressedBlock(MemoryStream buffer, int decompressedLength,
                        long blockIndex, long blockSequence, int rows)
                    {
                        Contracts.AssertValueOrNull(buffer);
                        Contracts.Assert(decompressedLength > 0);
                        Contracts.Assert(blockIndex >= 0);
                        Contracts.Assert(blockSequence >= 0);
                        Contracts.Assert(rows >= 0);

                        Buffer = buffer;
                        DecompressedLength = decompressedLength;
                        BlockIndex = blockIndex;
                        BlockSequence = blockSequence;
                        Rows = rows;
                        Contracts.Assert(!IsSentinel);
                    }

                    /// <summary>
                    /// Constructor for a sentinel compressed block. (For example,
                    /// the pipe's last block, which contains no valid data.)
                    /// </summary>
                    public CompressedBlock(long blockSequence)
                    {
                        Contracts.Assert(blockSequence >= 0);
                        BlockIndex = -1;
                        BlockSequence = blockSequence;
                        Contracts.Assert(IsSentinel);
                    }
                }

                private sealed class ReaderContainer
                {
                    public readonly IValueReader<T> Reader;
                    public readonly MemoryStream Stream;
                    public readonly int Rows;
                    public readonly long BlockSequence;

                    public ReaderContainer(IValueReader<T> reader, MemoryStream stream, int rows, long blockSequence)
                    {
                        Contracts.AssertValue(reader);
                        Contracts.AssertValue(stream);
                        Contracts.Assert(rows > 0);
                        Reader = reader;
                        Stream = stream;
                        Rows = rows;
                        BlockSequence = blockSequence;
                    }
                }

                /// <summary>
                /// This is called through reflection so it will appear to have no references.
                /// </summary>
                public ReadPipe(Cursor parent, int columnIndex, int bufferSize)
                    : base(parent, columnIndex)
                {
                    Ectx.Assert(bufferSize > 0);

                    TableOfContentsEntry entry = Parent._actives[ColumnIndex];
                    Ectx.AssertValue(entry);
                    Ectx.Assert(!entry.IsGenerated);
                    Ectx.AssertValue(entry.Codec);
                    Ectx.Assert(entry.Codec is IValueCodec<T>);
                    Ectx.Assert(Enum.IsDefined(typeof(CompressionKind), entry.Compression));

                    _codec = (IValueCodec<T>)entry.Codec;
                    _compression = entry.Compression;

                    int maxComp;
                    int maxDecomp;
                    entry.GetMaxBlockSizes(out maxComp, out maxDecomp);
                    _compPool = parent._parent._bufferCollection.Get(maxComp);
                    _decompPool = parent._parent._bufferCollection.Get(maxDecomp);
                    _lookup = entry.GetLookup();
                    _stream = parent._parent._stream;

                    Ectx.AssertValue(_compPool);
                    Ectx.AssertValue(_decompPool);
                    Ectx.AssertValue(_lookup);
                    Ectx.AssertValue(_stream);

                    _toDecompress = new BlockingCollection<CompressedBlock>(_bufferSize);
                    _toDecompressEnumerator = _toDecompress.GetConsumingEnumerable(ExMarshaller.Token).GetEnumerator();
                    _toRead = new BlockingCollection<ReaderContainer>(bufferSize);
                    _toReadEnumerator = _toRead.GetConsumingEnumerable(ExMarshaller.Token).GetEnumerator();
                    _waiter = new OrderedWaiter();
                }

                public override void PrepAndSendCompressedBlock(long blockIndex, long blockSequence, int rowCount)
                {
                    BlockLookup lookup = _lookup[(int)blockIndex];
                    var mem = _compPool.Get();
                    // Read the compressed buffer, then pass it off to the decompress worker threads.
                    EnsureCapacity(mem, lookup.BlockLength);
                    mem.SetLength(lookup.BlockLength);
                    ArraySegment<byte> buffer;
                    bool tmp = mem.TryGetBuffer(out buffer);
                    Contracts.Assert(tmp);
                    lock (_stream)
                    {
                        _stream.Seek(lookup.BlockOffset, SeekOrigin.Begin);
                        _stream.ReadBlock(buffer.Array, buffer.Offset, buffer.Count);
                        Contracts.Assert(lookup.BlockOffset + lookup.BlockLength == _stream.Position);
                    }
                    var block = new CompressedBlock(mem, lookup.DecompressedBlockLength, blockIndex, blockSequence, rowCount);
                    _toDecompress.Add(block, ExMarshaller.Token);
                }

                public override void SendSentinelBlock(long blockSequence)
                {
                    CompressedBlock sentBlock = new CompressedBlock(blockSequence);
                    _toDecompress.Add(sentBlock, ExMarshaller.Token);
                    _toDecompress.CompleteAdding();
                }

                private static void EnsureCapacity(MemoryStream stream, int value)
                {
                    // More or less copied and pasted from the memorystream's EnsureCapacity
                    // code... that sure would be useful to just have.
                    int cap = stream.Capacity;
                    if (cap >= value)
                        return;
                    const int arrayMaxLen = 0x7FFFFFC7;
                    int newCapacity = value;
                    if (newCapacity < 256)
                        newCapacity = 256;
                    if (newCapacity < cap * 2)
                        newCapacity = cap * 2;
                    if ((uint)(cap * 2) > arrayMaxLen)
                        newCapacity = value > arrayMaxLen ? value : arrayMaxLen;
                    stream.Capacity = newCapacity;
                }

                public override bool DecompressOne()
                {
                    CompressedBlock block;
                    // It is necessary to lock the decompress enumerator: the
                    // collection itself is thread safe, the enumerator is not.
                    // The MoveNext and Current must be an atomic operation.
                    lock (_toDecompressEnumerator)
                    {
                        if (!_toDecompressEnumerator.MoveNext())
                            return false;
                        block = _toDecompressEnumerator.Current;
                    }

                    Contracts.Assert(!_toRead.IsAddingCompleted);
                    if (block.IsSentinel)
                    {
                        // We can only complete adding after we are certain that all prior workers
                        // that may be working, have had the opportunity to interact with _toRead.
                        _waiter.Wait(block.BlockSequence, ExMarshaller.Token);
                        _toRead.CompleteAdding();
                        _waiter.Increment();
                        return true;
                    }

                    MemoryStream buffer = block.Buffer;
                    buffer.Position = 0;

                    if (Parent._disposed)
                    {
                        // We have disposed. Skip the decompression steps, while returning resources.
                        _compPool.Return(ref buffer);
                        // In this case we still need to increment to the next block sequence number,
                        // so that future workers along this pipe will have an opportunity to return.
                        _waiter.Wait(block.BlockSequence, ExMarshaller.Token);
                        _waiter.Increment();
                        return true;
                    }

                    MemoryStream decomp = _decompPool.Get();
                    EnsureCapacity(decomp, block.DecompressedLength);
                    decomp.SetLength(block.DecompressedLength);
                    using (Stream stream = _compression.DecompressStream(buffer))
                    {
                        ArraySegment<byte> buf;
                        bool tmp = decomp.TryGetBuffer(out buf);
                        Contracts.Assert(tmp);
                        stream.ReadBlock(buf.Array, buf.Offset, buf.Count);
                    }
                    _compPool.Return(ref buffer);
                    decomp.Seek(0, SeekOrigin.Begin);
                    IValueReader<T> reader = _codec.OpenReader(decomp, block.Rows);
                    _waiter.Wait(block.BlockSequence, ExMarshaller.Token);
                    // Enter exclusive section for this pipe.
                    _toRead.Add(new ReaderContainer(reader, decomp, block.Rows, block.BlockSequence), ExMarshaller.Token);
                    _waiter.Increment();
                    // Exit exclusive section for this pipe.
                    return true;
                }

                public override bool MoveNext()
                {
                    Contracts.Assert(_remaining >= 0);
                    Contracts.Assert(_remaining == 0 || _curr != null);
                    if (_remaining == 0)
                    {
                        if (_curr != null)
                        {
                            _curr.Reader.Dispose();
                            MemoryStream mem = _curr.Stream;
                            _curr = null;
                            _decompPool.Return(ref mem);
                        }
                        if (!_toReadEnumerator.MoveNext())
                            return false;
                        _curr = _toReadEnumerator.Current;
                        Contracts.AssertValue(_curr);
                        _remaining = _curr.Rows;
                    }
                    Contracts.Assert(_remaining > 0);
                    _curr.Reader.MoveNext();
                    _remaining--;
                    return true;
                }

                public override bool MoveNextCleanup()
                {
                    // This is analogous to the _remaining == 0 part of
                    // MoveNext, except we don't actually do anything with
                    // the block we fetch.
                    if (_curr != null)
                    {
                        _curr.Reader.Dispose();
                        MemoryStream mem = _curr.Stream;
                        _curr = null;
                        _decompPool.Return(ref mem);
                    }
                    if (!_toReadEnumerator.MoveNext())
                        return false;
                    _curr = _toReadEnumerator.Current;
                    return true;
                }

                private void Get(ref T value)
                {
                    Contracts.Check(_curr != null, _badCursorState);
                    _curr.Reader.Get(ref value);
                }

                public override Delegate GetGetter()
                {
                    ValueGetter<T> getter = Get;
                    return getter;
                }
            }

            public bool IsColumnActive(int col)
            {
                Ch.CheckParam(0 <= col && col < _colToActivesIndex.Length, nameof(col));
                return _colToActivesIndex[col] >= 0;
            }

            protected override bool MoveNextCore()
            {
                Ch.Assert(!_disposed);
                bool more = Position != _lastValidCounter;
                for (int c = 0; c < _pipes.Length; ++c)
                {
                    bool pipeMoved;
                    try
                    {
                        pipeMoved = _pipes[c].MoveNext();
                    }
                    catch (OperationCanceledException ex)
                    {
                        // Suppress the premature exit handing. We can be assured that all
                        // threads will exit if all potentially blocking operations are
                        // waiting on the same cancellation token that we catch here.
                        Contracts.Assert(ex.CancellationToken == _exMarshaller.Token);
                        _disposed = true;
                        // Unlike the early non-error dispose case, we do not make any
                        // effort to recycle buffers since it would be exceptionally difficult
                        // to do so. All threads are already unblocked, one of them with the
                        // source exception that kicked off this process, the remaining with
                        // other later exceptions or the operation cancelled exception. So we
                        // are free to join. Still, given the exceptional nature, we won't
                        // wait forever to do it.
                        const int timeOut = 100;
                        _pipeTask.Wait(timeOut);
                        _readerThread.Join(timeOut);
                        // Throw. Considering we are here, this must throw.
                        _exMarshaller.ThrowIfSet(Ch);
                        // This can't be, cause the cancellation token we were waiting
                        // on is private to the exception marshaller, and can't be
                        // set any other way than if set.
                        Contracts.Assert(false);
                        throw;
                    }

                    // They should all stop at the same time.
                    Ch.Assert(more == pipeMoved);
                }
                if (!more && _pipes.Length > 0)
                {
                    // Set the _disposed flag, so that when the Dispose
                    // method is called it does not trigger the "premature
                    // exit" handling.
                    _disposed = true;
                    // If we got to this point these threads must have already
                    // completed their work, but for the sake of hygiene join
                    // against them anyway.
                    _pipeTask.Wait();
                    _readerThread.Join();
                }
                return more;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.CheckParam(0 <= col && col < _colToActivesIndex.Length, nameof(col));
                Ch.CheckParam(_colToActivesIndex[col] >= 0, nameof(col), "requested column not active");
                var getter = _pipeGetters[_colToActivesIndex[col]] as ValueGetter<TValue>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }

            /// <summary>
            /// Even in the case with no rows, there still must be valid delegates. This will return
            /// a delegate that simply always throws.
            /// </summary>
            private Delegate GetNoRowGetter(ColumnType type)
            {
                return Utils.MarshalInvoke(NoRowGetter<int>, type.RawType);
            }

            private Delegate NoRowGetter<T>()
            {
                ValueGetter<T> del =
                    (ref T value) =>
                    {
                        throw Ch.Except(_badCursorState);
                    };
                return del;
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                if (_blockShuffleOrder == null)
                {
                    return
                        (ref UInt128 val) =>
                        {
                            Ch.Check(IsGood, "Cannot call ID getter in current state");
                            val = new UInt128((ulong)Position, 0);
                        };
                }
                // Find the index of the last block. Because the last block is unevenly sized,
                // but in the case of shuffling can occur anywhere, our calculations of the "true"
                // position of the row have to account for the presence of this strange block.
                int lastBlockIdx = 0;
                for (int i = 1; i < _blockShuffleOrder.Length; ++i)
                {
                    if (_blockShuffleOrder[i] > _blockShuffleOrder[lastBlockIdx])
                        lastBlockIdx = i;
                }
                int correction = _rowsPerBlock - _rowsInLastBlock;
                long firstPositionToCorrect = ((long)lastBlockIdx * _rowsPerBlock) + _rowsInLastBlock;

                return
                    (ref UInt128 val) =>
                    {
                        Ch.Check(IsGood, "Cannot call ID getter in current state");
                        long pos = Position;
                        if (pos >= firstPositionToCorrect)
                            pos += correction;
                        Ch.Assert(pos / _rowsPerBlock < _blockShuffleOrder.Length);
                        long blockPos = (long)_rowsPerBlock * _blockShuffleOrder[(int)(pos / _rowsPerBlock)];
                        blockPos += (pos % _rowsPerBlock);
                        Ch.Assert(0 <= blockPos && blockPos < _parent.RowCount);
                        val = new UInt128((ulong)blockPos, 0);
                    };
            }
        }

        public sealed class InfoCommand : ICommand
        {
            public const string LoadName = "IdvInfo";

            public sealed class Arguments
            {
                [DefaultArgument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The data file", SortOrder = 0)]
                public string DataFile;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Verbose?", ShortName = "v", Hide = true)]
                public bool? Verbose;
            }

            private readonly IHostEnvironment _env;
            private readonly string _dataFile;

            public InfoCommand(IHostEnvironment env, Arguments args)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(args, nameof(args));
                env.CheckNonWhiteSpace(args.DataFile, nameof(args.DataFile), "Data file must be specified");

                _dataFile = args.DataFile;
                _env = env.Register(LoadName, verbose: args.Verbose);
            }

            private string VersionToString(ulong ver)
            {
                return string.Format("{0}.{1}.{2}.{3}", ver >> 48,
                    (ver >> 32) & 0xffff, (ver >> 16) & 0xffff, ver & 0xffff);
            }

            public void Run()
            {
                var host = _env.Register(LoadName);
                var data = new MultiFileSource(_dataFile);
                // We will not be iterating through the data, so the defaults are fine.
                var args = new BinaryLoader.Arguments();

                using (var loader = new BinaryLoader(host, args, data))
                using (var ch = host.Start("Inspection"))
                {
                    RunCore(ch, loader);
                    ch.Done();
                }
            }

            private void RunCore(IChannel ch, BinaryLoader loader)
            {
                Contracts.AssertValue(ch);
                ch.AssertValue(loader);

                // Report on basic info from the header.
                Header header = loader._header;
                long idvSize = header.TailOffset + sizeof(ulong);
                if (loader._stream.Length != idvSize)
                {
                    ch.Warning("Stream is {0} bytes, IDV is {1} bytes. This is legal but unusual.",
                        loader._stream.Length, idvSize);
                }
                ch.Info("IDV {0} (compat {1}), {2} col, {3} row, {4} bytes", VersionToString(header.Version),
                    VersionToString(header.CompatibleVersion), header.ColumnCount, header.RowCount, idvSize);

                // Get all of the columns from the loader that are not generated.
                // We will want to report them in the order they appear in the file,
                // so order by column index.
                var cols = loader._aliveColumns.Select(t => new KeyValuePair<bool, TableOfContentsEntry>(true, t))
                    .Concat(loader._deadColumns.Select(t => new KeyValuePair<bool, TableOfContentsEntry>(false, t)))
                    .Where(t => !t.Value.IsGenerated).OrderBy(t => t.Value.ColumnIndex);

                long totalBlockSize = 0;
                long totalMetadataSize = 0;
                long totalMetadataTocSize = 0;
                long blockCount = 0;

                int colCount = 0;
                foreach (var isLiveAndCol in cols)
                {
                    var col = isLiveAndCol.Value;
                    ch.Assert(col.ColumnIndex == colCount); // *Every* column in the file should be here, even if dead.

                    // REVIEW: It is currently allowed in the format to point to the same block of data twice,
                    // even across columns. (Allowed in the sense that nothing prevents this from happening, but then
                    // there are no writers that take advantage of this to, say, resolve dependent blocks.) For our
                    // purposes here we will assume that all blocks are disjoint.

                    string typeDesc = col.Type == null ? "<?>" : col.Type.ToString();
                    long uncompressedSize = 0;
                    long compressedSize = 0;

                    var blockLookups = col.GetLookup();
                    ch.AssertValue(blockLookups); // Should be null iff this is generated, and we dropped those.
                    foreach (var lookup in blockLookups)
                    {
                        compressedSize += lookup.BlockLength;
                        uncompressedSize += lookup.DecompressedBlockLength;
                        blockCount++;
                    }
                    totalBlockSize += compressedSize;
                    ch.Info(MessageSensitivity.Schema, "Column {0} '{1}'{2} of {3} in {4} blocks of {5}", col.ColumnIndex, col.Name,
                        isLiveAndCol.Key ? "" : " (DEAD!)", typeDesc, blockLookups.Length, col.RowsPerBlock);
                    ch.Info("  {0} compressed from {1} with {2} ({3:0.00%})",
                        compressedSize, uncompressedSize, col.Compression, (compressedSize + 0.0) / uncompressedSize);

                    // Report on the metadata.
                    var mtoc = col.GetMetadataTocArray();
                    var deadMtoc = col.GetDeadMetadataTocArray();
                    if (mtoc == null)
                        ch.Assert(deadMtoc == null && col.MetadataTocOffset == 0);
                    else
                    {
                        long metadataSize = mtoc.Sum(t => t.BlockSize) + deadMtoc.Sum(t => t.BlockSize);
                        long metadataTocSize = col.GetMetadataTocEndOffset() - col.MetadataTocOffset;
                        string deadDisc = deadMtoc.Length > 0 ? string.Format(" ({0} dead)", deadMtoc.Length) : "";
                        ch.Info("  {0} pieces of metadata{1} has {2} byte table, {3} byte content",
                            mtoc.Length, deadDisc, metadataTocSize, metadataSize);
                        // REVIEW: Maybe with a switch, we could report on the individual pieces here as we do columns? Less important though.
                        totalMetadataSize += metadataSize;
                        totalMetadataTocSize += metadataTocSize;
                    }

                    colCount++;
                }
                ch.Assert(colCount == header.ColumnCount);

                // Report on the the size breakdown.
                long lookupTablesSize = blockCount * (sizeof(long) + sizeof(int) + sizeof(int));
                long totalTocSize = loader._tocEndLim - header.TableOfContentsOffset;
                const long headTailSize = Header.HeaderSize + sizeof(ulong);
                long accountedSize = 0;

                ch.Info(" "); // REVIEW: Ugh. Better way? Otherwise it's all smushed up.
                Action<string, long> report =
                    (desc, size) =>
                    {
                        ch.Info("{0,8:0.000%} for {1} ({2} bytes)", (size + 0.0) / idvSize, desc, size);
                        accountedSize += size;
                    };
                report("data blocks", totalBlockSize);
                report("block lookup", lookupTablesSize);
                report("table of contents", totalTocSize);
                report("metadata contents", totalMetadataSize);
                report("metadata table of contents", totalMetadataTocSize);
                report("header/tail", headTailSize);

                // The two could be unequal if there is an in-place modification done to the file.
                if (idvSize != accountedSize)
                    report("unknown", idvSize - accountedSize);
            }
        }
    }
}
