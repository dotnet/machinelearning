// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(TransposeLoader.Summary, typeof(TransposeLoader), typeof(TransposeLoader.Arguments), typeof(SignatureDataLoader),
    "Transpose Loader", TransposeLoader.LoadName, "Transpose", "trans")]

[assembly: LoadableClass(TransposeLoader.Summary, typeof(TransposeLoader), null, typeof(SignatureLoadDataLoader),
    "Transpose Data View Loader", TransposeLoader.LoadName)]

namespace Microsoft.ML.Runtime.Data.IO
{
    /// <summary>
    /// The transposed loader reads the transposed binary format. This binary format, at a high level, is nothing more
    /// than, for a dataview with "c" columns, "c+1" binary IDVs glued together. We call these sub-IDVs. The first of these,
    /// the master sub-IDV stores the overall schema, and optionally the data in row-wise format.
    /// </summary>
    /// <seealso cref="TransposeSaver"/>
    public sealed class TransposeLoader : IDataLoader, ITransposeDataView
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The number of worker decompressor threads to use", ShortName = "t")]
            public int? Threads;
        }

        [StructLayout(LayoutKind.Explicit, Size = HeaderSize)]
        public struct Header
        {
            /// <summary>
            /// The fixed header size. This should not be changed even in future versions of the format.
            /// </summary>
            public const int HeaderSize = 256;

            /// <summary>
            /// The header must start with this signature. This number will
            /// appear as the eight-byte sequence "XPOSEDDV" if encoded in
            /// little-endian. (XPOSEDDV is meant to suggest transposed DataView).
            /// </summary>
            public const ulong SignatureValue = 0x56444445534F5058;

            /// <summary>
            /// The file must end with this value. Is is simply the
            /// byte-order-reversed version of the head signature.
            /// </summary>
            public const ulong TailSignatureValue = 0x58504F5345444456;

            /// <summary>
            /// The current version of the format this software can write.
            /// </summary>
            public const ulong WriterVersion = 0x0001000100010001; // This first version of the format.
            public const ulong CanBeReadByVersion = 0x0001000100010001;

            internal static string VersionToString(ulong v)
            {
                return string.Format("{0}.{1}.{2}.{3}",
                    (v >> 48) & 0xffff, (v >> 32) & 0xffff,
                    (v >> 16) & 0xffff, v & 0xffff);
            }

            /// <summary>
            /// The magic number of this file.
            /// </summary>
            [FieldOffset(0)]
            public ulong Signature;

            /// <summary>
            /// Indicates the version of the data file.
            /// </summary>
            [FieldOffset(8)]
            public ulong Version;

            /// <summary>
            /// Indicates the minimum reader version that can interpret this file, possibly
            /// with some data loss.
            /// </summary>
            [FieldOffset(16)]
            public ulong CompatibleVersion;

            /// <summary>
            /// The offset to the list of the directory of the sub-IDV structures.
            /// </summary>
            [FieldOffset(24)]
            public long SubIdvTableOffset;

            /// <summary>
            /// The eight-byte tail signature starts at this offset. So, the entire dataset
            /// stream should be considered to have eight plus this value bytes.
            /// </summary>
            [FieldOffset(32)]
            public long TailOffset;

            /// <summary>
            /// The number of rows.
            /// </summary>
            [FieldOffset(40)]
            public long RowCount;

            /// <summary>
            /// The number of columns. There will be this + 1 entries in the sub-IDV table
            /// offset structure.
            /// </summary>
            [FieldOffset(48)]
            public int ColumnCount;

            // Lots of padding (up to size 256)....
        }

        /// <summary>
        /// A sub-IDV entry corresponds to an offset and length within the transposed file, that points
        /// either to a block binary-IDV formatted data if the offset is positive, or indicates that there
        /// is no corresponding IDV entry if the offset is zero.
        /// </summary>
        private abstract class SubIdvEntry
        {
            private readonly TransposeLoader _parent;
            // The start of the binary IDV stream in the file.
            private readonly long _offset;
            // The length of that binary IDV stream in the file.
            private readonly long _length;
            private IDataView _view;

            /// <summary>
            /// Is true when this sub-IDV appears to exist, without actually loading that sub-IDV.
            /// If this returns true, <see cref="GetViewOrNull"/> will either return a non-null
            /// value, or throw some sort of formatting error.
            /// </summary>
            public bool HasDataView { get { return _view != null || _offset > 0; } }

            private IHost Host { get { return _parent._host; } }

            /// <summary>
            /// Reads the table of contents entry from the file, advancing the binary loader stream.
            /// </summary>
            private SubIdvEntry(TransposeLoader parent, BinaryReader reader)
            {
                Contracts.AssertValue(parent);
                _parent = parent;
                Host.AssertValue(reader);

                _offset = reader.ReadInt64();
                Host.CheckDecode(_offset == 0 || (Header.HeaderSize <= _offset && _offset <= _parent._header.TailOffset));
                _length = reader.ReadInt64();
                // Want offset + length <= tail offset, structure to avoid overflow.
                Host.CheckDecode(0 <= _length && _offset <= _parent._header.TailOffset - _length);
            }

            /// <summary>
            /// Constructs an empty table of contents entry, with no offset.
            /// </summary>
            private SubIdvEntry(TransposeLoader parent)
            {
                Contracts.AssertValue(parent);
                _parent = parent;
            }

            /// <summary>
            /// Gets the dataview corresponding to this sub-IDV entry. This will
            /// lazily load the file, if it has not previously been requested. This
            /// will return <c>null</c> if the offset is 0.
            /// </summary>
            public IDataView GetViewOrNull()
            {
                if (_view == null && _offset > 0)
                {
                    Stream stream = _parent._file.Open(0);
                    stream.Seek(_offset, SeekOrigin.Begin);
                    Contracts.Check(stream.Position == _offset, "Unexpected position on substream");
                    SubsetStream ss = new SubsetStream(stream, _length);
                    var binArgs = new BinaryLoader.Arguments();
                    if (_parent._threads > 0)
                        binArgs.Threads = _parent._threads;
                    BinaryLoader loader = new BinaryLoader(Host,
                        binArgs, ss, leaveOpen: false);
                    var view = Interlocked.CompareExchange(ref _view, loader, null);
                    // If multiple threads have called this as it was being loaded,
                    // have ensure that this check only happens once.
                    if (view == loader)
                        VerifyView(view);
                }
                return _view;
            }

            /// <summary>
            /// Called once, to verify that the lazily read dataview is "correct." Called by
            /// <see cref="GetViewOrNull"/> once it has been read. Any problems with the data-view
            /// should be handle with <see cref="Contracts.CheckDecode(bool)"/> or by throwing
            /// <see cref="Contracts.ExceptDecode()"/>, as we consider the views not adhering to
            /// standards to be a file formatting issue. Note that this will never be called if
            /// the offset field is zero.
            /// </summary>
            protected abstract void VerifyView(IDataView view);

            /// <summary>
            /// This is the entry corresponding to the first IDV entry in the file, which will hold
            /// at least the schema information for all columns. There should be one of these per
            /// file. Optionally, this file can also hold the row-wise data stored as well, in case
            /// the user wanted to have the hybrid row/slotwise store. For this one, it is illegal
            /// for the offset to be zero.
            /// </summary>
            public sealed class SchemaSubIdv : SubIdvEntry
            {
                public IDataView GetView()
                {
                    // The schema sub-IDV is required to have actual content, so have a method
                    // that reflects this better than the GetViewOrNull.
                    var view = GetViewOrNull();
                    Contracts.Assert(view != null);
                    return view;
                }

                public SchemaSubIdv(TransposeLoader parent, BinaryReader reader)
                    : base(parent, reader)
                {
                    // REVIEW: Technically we could, I guess, support a relaxing of this
                    // in the case where there are no columns, but this seems really silly.
                    Host.CheckDecode(HasDataView);
                }

                public SchemaSubIdv(TransposeLoader parent, IDataView view)
                    : base(parent)
                {
                    _view = view;
                }

                protected override void VerifyView(IDataView view)
                {
                    Host.AssertValue(view);
                    var rowCountNull = view.GetRowCount();
                    // This came from a binary IDV, so it must have an actual row count.
                    Host.Assert(rowCountNull.HasValue);
                    long rowCount = rowCountNull.Value;
                    // Either we are holding only the schema information and have no rows,
                    // or we have the double-stored hybrid dataview with data stored both
                    // row-wise and column wise.
                    Host.CheckDecode(rowCount == 0 || _parent._header.RowCount == rowCount);

                    var schema = view.Schema;
                    Host.CheckDecode(schema.ColumnCount == _parent._header.ColumnCount);
                }
            }

            /// <summary>
            /// This is the entry corresponding to the transposed columns. There will be one of
            /// these per column, though some entries will not actually have a corresponding
            /// dataview (for example, they will have an offset of 0) if the column was not one selected
            /// for slot-wise transposition.
            /// </summary>
            public sealed class TransposedSubIdv : SubIdvEntry
            {
                private readonly int _col;

                public TransposedSubIdv(TransposeLoader parent, BinaryReader reader, int col)
                    : base(parent, reader)
                {
                    // The correctness of this relies upon the schema entry being read first.
                    Host.AssertValue(parent._schemaEntry);
                    Host.Assert(0 <= col && col < parent.Schema.ColumnCount);
                    _col = col;

                    // Either we have to have data, or the parent has to have explicit row data.
                    // If both of these are false, then we are advertising a column for which we
                    // have no data whatsoever, which is silly.
                    Host.CheckDecode(HasDataView || parent.HasRowData);
                }

                /// <summary>
                /// Returns an empty sub-IDV entry for the no-file case.
                /// </summary>
                public TransposedSubIdv(TransposeLoader parent, int col)
                    : base(parent)
                {
                    _col = col;
                }

                protected override void VerifyView(IDataView view)
                {
                    Host.AssertValue(view);
                    // This must have precisely one column, of type vector.
                    var schema = view.Schema;
                    Host.CheckDecode(schema.ColumnCount == 1);
                    var ttype = schema.GetColumnType(0);
                    Host.CheckDecode(ttype.IsVector);
                    // We have no way to encode a type of zero length vectors per se in the case
                    // when there are no rows in the original dataset, but accept that if the vector
                    // count is "unknown" then it's really a zero-row dataset.
                    Host.CheckDecode(ttype.ValueCount == _parent._header.RowCount);
                    // This came from a binary IDV, so it must have an actual "row" count,
                    // though this row count for this is more like a "slot" count.
                    var rowCountNull = view.GetRowCount();
                    Host.Assert(rowCountNull.HasValue);
                    long rowCount = rowCountNull.Value;
                    // There must be one "row" per "slot" on the column this is a transpose of.
                    // Check that.
                    var type = _parent.Schema.GetColumnType(_col);
                    Host.CheckDecode(type.ValueCount == rowCount);
                    // The item types should be the same.
                    Host.CheckDecode(type.ItemType.Equals(ttype.ItemType));
                }
            }
        }

        // Positive if explicit, otherwise let the sub-binary loader decide for themselves.
        private readonly int _threads;

        private readonly IMultiStreamSource _file;
        private readonly IHost _host;
        private readonly Header _header;
        private readonly SchemaImpl _schema;

        // This is a sub-IDV holding the schema, and optionally the data stored in row-wise format.
        private readonly SubIdvEntry.SchemaSubIdv _schemaEntry;
        // There will be _header.ColumnCount items here, holding the sub-IDVs.
        private readonly SubIdvEntry.TransposedSubIdv[] _entries;
        // There will a transposer per column's sub-IDV entry (lazily initialized, initially null)
        // if the master sub-IDV does not contain the actual data. This entire array will be null
        // iff the master sub-IDV contains the actual data.
        private readonly Transposer[] _colTransposers;
        // An object to lock on whenever one might be attempting to create one of the lazily initialized
        // transposers, since transposition is a tricky operation. This is null iff the above array is null.
        private readonly object _colTransposersLock;

        /// <summary>
        /// Lower inclusive bound of versions this reader can read.
        /// </summary>
        private const ulong ReaderFirstVersion = 0x0001000100010001;

        /// <summary>
        /// Upper inclusive bound of versions this reader can read.
        /// </summary>
        private const ulong ReaderVersion = ReaderFirstVersion;

        internal const string Summary = "Loads a binary transposed data file.";
        internal const string LoadName = "TransposeLoader";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "XPSLOADR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoadName);
        }

        // We return the schema view's schema, because we don't necessarily want
        // something that can be cast to a transpose schema, and also because the
        // transpose schema is defined after the entries have been read, which
        // inspect the schema. We also want to ensure that the useful property that
        // a cursor and view's schemas are the same, is preserved, which allows us
        // to use the cursors from the schema view if convenient to do so.
        public ISchema Schema { get { return _schemaEntry.GetView().Schema; } }

        public ITransposeSchema TransposeSchema { get { return _schema; } }

        /// <summary>
        /// Whether the master schema sub-IDV has the actual data.
        /// </summary>
        private bool HasRowData
        {
            get { return _header.RowCount == _schemaEntry.GetView().GetRowCount(); }
        }

        public bool CanShuffle
        {
            get
            {
                // If we have an internal view with the row-wise data actually in it,
                // then we can use that for shuffling. Otherwise we won't support it.
                var view = _schemaEntry.GetView();
                if (_header.RowCount == view.GetRowCount())
                    return view.CanShuffle;
                return false;
            }
        }

        public TransposeLoader(IHostEnvironment env, Arguments args, IMultiStreamSource file)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadName);
            _host.CheckValue(args, nameof(args));
            _host.CheckValue(file, nameof(file));
            _host.Check(file.Count == 1, "Transposed loader accepts a single file only");

            _threads = args.Threads ?? 0;
            if (_threads < 0)
                _threads = 0;

            _file = file;
            using (Stream stream = _file.Open(0))
            using (BinaryReader reader = new BinaryReader(stream))
            {
                _header = InitHeader(reader);
                reader.Seek(_header.SubIdvTableOffset);
                _schemaEntry = new SubIdvEntry.SchemaSubIdv(this, reader);
                _entries = new SubIdvEntry.TransposedSubIdv[_header.ColumnCount];
                for (int c = 0; c < _entries.Length; ++c)
                    _entries[c] = new SubIdvEntry.TransposedSubIdv(this, reader, c);
                _schema = new SchemaImpl(this);
                if (!HasRowData)
                {
                    _colTransposers = new Transposer[_header.ColumnCount];
                    _colTransposersLock = new object();
                }
            }
        }

        private TransposeLoader(IHost host, ModelLoadContext ctx, IMultiStreamSource file)
        {
            Contracts.CheckValue(host, nameof(host));
            _host = host;
            _host.CheckValue(file, nameof(file));
            _host.Check(file.Count == 1, "Transposed loader accepts a single file only");

            // *** Binary format **
            // int: Number of threads if explicitly defined, or 0 if the
            //      number of threads was automatically determined

            _threads = ctx.Reader.ReadInt32();
            _host.CheckDecode(_threads >= 0);

            // Dedupe code somehow?
            _file = file;
            using (Stream stream = _file.Open(0))
            using (BinaryReader reader = new BinaryReader(stream))
            {
                _header = InitHeader(reader);
                reader.Seek(_header.SubIdvTableOffset);
                _schemaEntry = new SubIdvEntry.SchemaSubIdv(this, reader);
                _entries = new SubIdvEntry.TransposedSubIdv[_header.ColumnCount];
                for (int c = 0; c < _entries.Length; ++c)
                    _entries[c] = new SubIdvEntry.TransposedSubIdv(this, reader, c);
                _schema = new SchemaImpl(this);
                if (!HasRowData)
                {
                    _colTransposers = new Transposer[_header.ColumnCount];
                    _colTransposersLock = new object();
                }
            }
        }

        private TransposeLoader(IHost host, ModelLoadContext ctx, IDataView schemaView)
        {
            Contracts.CheckValue(host, nameof(host));
            _host = host;
            _host.CheckValue(schemaView, nameof(schemaView));

            // *** Binary format **
            // int: Number of threads if explicitly defined, or 0 if the
            //      number of threads was automatically determined

            _threads = ctx.Reader.ReadInt32();
            _host.CheckDecode(_threads >= 0);

            _header = new Header()
            {
                ColumnCount = schemaView.Schema.ColumnCount
            };
            _schemaEntry = new SubIdvEntry.SchemaSubIdv(this, schemaView);
            _host.Assert(_schemaEntry.GetViewOrNull() == schemaView);
            _entries = new SubIdvEntry.TransposedSubIdv[_header.ColumnCount];
            for (int c = 0; c < _entries.Length; ++c)
            {
                _entries[c] = new SubIdvEntry.TransposedSubIdv(this, c);
                _host.Assert(_entries[c].GetViewOrNull() == null);
            }
            _host.Assert(HasRowData);

        }
        public static TransposeLoader Create(IHostEnvironment env, ModelLoadContext ctx, IMultiStreamSource files)
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
                        BinaryLoader schemaView = null;
                        // In the case where we have no input streams, but we have an input schema from
                        // the model repository, we still want to surface ourselves as being a binary loader
                        // with the existing schema. The loader "owns" this stream.
                        if (ctx.TryLoadBinaryStream("Schema.idv",
                            r => schemaView = new BinaryLoader(h, new BinaryLoader.Arguments(),
                                 HybridMemoryStream.CreateCache(r.BaseStream), leaveOpen: false)))
                        {
                            h.AssertValue(schemaView);
                            h.CheckDecode(schemaView.GetRowCount() == 0);
                            // REVIEW: Do we want to be a bit more restrictive around uninterpretable columns?
                            return new TransposeLoader(h, ctx, schemaView);
                        }
                        h.Assert(schemaView == null);
                        // Fall through, allow the failure to be on OpenStream.
                    }
                    return new TransposeLoader(h, ctx, files);
                });
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format **
            // int: Number of threads if explicitly defined, or 0 if the
            //      number of threads is automatically determined

            _host.Assert(_threads >= 0);
            ctx.Writer.Write(_threads);

            SaveSchema(_host, ctx, Schema);
        }

        /// <summary>
        /// Save a zero-row dataview that will be used to infer schema information, used in the case
        /// where the tranpsose loader is instantiated with no input streams.
        /// </summary>
        private static void SaveSchema(IHostEnvironment env, ModelSaveContext ctx, ISchema schema)
        {
            Contracts.AssertValue(env);

            env.AssertValue(ctx);
            env.AssertValue(schema);

            var noRows = new EmptyDataView(env, schema);
            env.Assert(noRows.GetRowCount() == 0);

            var saverArgs = new BinarySaver.Arguments();
            saverArgs.Silent = true;
            var saver = new BinarySaver(env, saverArgs);

            // We load our schema from what amounts to a binary loader, so all columns should likewise be savable.
            env.Assert(Enumerable.Range(0, schema.ColumnCount).All(c => saver.IsColumnSavable(schema.GetColumnType(c))));
            ctx.SaveBinaryStream("Schema.idv", w => saver.SaveData(w.BaseStream, noRows, Utils.GetIdentityPermutation(schema.ColumnCount)));
        }

        private unsafe Header InitHeader(BinaryReader reader)
        {
            byte[] headerBytes = new byte[Header.HeaderSize];
            int cb = reader.Read(headerBytes, 0, Header.HeaderSize);
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
                "This does not appear to be a transposed dataview file");

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
            if (header.CompatibleVersion > ReaderVersion)
            {
                throw _host.Except("Cannot read version {0} data, latest that can be handled is {1}",
                    Header.VersionToString(header.CompatibleVersion), Header.VersionToString(ReaderVersion));
            }

            _host.CheckDecode(header.RowCount >= 0, "Row count cannot be negative");
            _host.CheckDecode(header.ColumnCount >= 0, "Column count cannot be negative");
            // Check the table of contents offset, though we do not at this time have the contents themselves.
            if (header.ColumnCount != 0 && header.SubIdvTableOffset < Header.HeaderSize)
                throw _host.ExceptDecode("Table of contents offset {0} less than header size, impossible", header.SubIdvTableOffset);

            // Check the tail signature.
            if (header.TailOffset < Header.HeaderSize)
                throw _host.ExceptDecode("Tail offset {0} less than header size, impossible", header.TailOffset);
            reader.Seek(header.TailOffset);
            ulong tailSig = reader.ReadUInt64();
            _host.CheckDecode(tailSig == Header.TailSignatureValue, "Incorrect tail signature");
            return header;
        }

        private sealed class SchemaImpl : ITransposeSchema
        {
            private readonly TransposeLoader _parent;
            private ISchema Schema { get { return _parent.Schema; } }
            private IHost Host { get { return _parent._host; } }
            public int ColumnCount { get { return Schema.ColumnCount; } }

            public SchemaImpl(TransposeLoader parent)
            {
                Contracts.AssertValue(parent);
                _parent = parent;
                var view = parent._schemaEntry.GetView().Schema;
            }

            public string GetColumnName(int col)
            {
                return Schema.GetColumnName(col);
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                return Schema.TryGetColumnIndex(name, out col);
            }

            public ColumnType GetColumnType(int col)
            {
                return Schema.GetColumnType(col);
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                return Schema.GetMetadataTypeOrNull(kind, col);
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                return Schema.GetMetadataTypes(col);
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                Schema.GetMetadata<TValue>(kind, col, ref value);
            }

            public VectorType GetSlotType(int col)
            {
                Host.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                var view = _parent._entries[col].GetViewOrNull();
                if (view == null)
                    return null;
                return view.Schema.GetColumnType(0).AsVector;
            }
        }

        public long? GetRowCount(bool lazy = true)
        {
            return _header.RowCount;
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            _host.CheckValue(predicate, nameof(predicate));
            _host.CheckValueOrNull(rand);
            if (HasRowData)
                return _schemaEntry.GetView().GetRowCursor(predicate, rand);
            return new Cursor(this, predicate);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            _host.CheckValue(predicate, nameof(predicate));
            if (HasRowData)
                return _schemaEntry.GetView().GetRowCursorSet(out consolidator, predicate, n, rand);
            consolidator = null;
            return new IRowCursor[] { GetRowCursor(predicate, rand) };
        }

        public ISlotCursor GetSlotCursor(int col)
        {
            _host.CheckParam(0 <= col && col < _header.ColumnCount, nameof(col));
            var view = _entries[col].GetViewOrNull();
            if (view == null)
            {
                throw _host.ExceptParam(nameof(col), "Bad call to GetSlotCursor on untransposable column '{0}'",
                    Schema.GetColumnName(col));
            }
            _host.CheckParam(0 <= col && col < _header.ColumnCount, nameof(col));
            // We don't want the type error, if there is one, to be handled by the get-getter, because
            // at the point we've gotten the interior cursor, but not yet constructed the slot cursor.
            ColumnType cursorType = TransposeSchema.GetSlotType(col).ItemType;
            IRowCursor inputCursor = view.GetRowCursor(c => true);
            try
            {
                return Utils.MarshalInvoke(GetSlotCursorCore<int>, cursorType.RawType, inputCursor);
            }
            catch (Exception)
            {
                // We've already verified the types so we shouldn't throw here, in principle, but just
                // be extra careful so we're sure to dispose the input cursor.
                if (inputCursor != null)
                    inputCursor.Dispose();
                throw;
            }
        }

        private ISlotCursor GetSlotCursorCore<T>(IRowCursor inputCursor)
        {
            return new SlotCursor<T>(this, inputCursor);
        }

        private sealed class SlotCursor<T> : SynchronizedCursorBase<IRowCursor>, ISlotCursor
        {
            private readonly TransposeLoader _parent;
            private readonly ValueGetter<VBuffer<T>> _getter;

            private IHost Host { get { return _parent._host; } }

            public SlotCursor(TransposeLoader parent, IRowCursor cursor)
                : base(parent._host, cursor)
            {
                _parent = parent;
                Ch.Assert(cursor.Schema.ColumnCount == 1);
                Ch.Assert(cursor.Schema.GetColumnType(0).RawType == typeof(VBuffer<T>));
                _getter = Input.GetGetter<VBuffer<T>>(0);
            }

            public VectorType GetSlotType()
            {
                var type = Input.Schema.GetColumnType(0).AsVector;
                Ch.AssertValue(type);
                return type;
            }

            public ValueGetter<VBuffer<TValue>> GetGetter<TValue>()
            {
                ValueGetter<VBuffer<TValue>> getter = _getter as ValueGetter<VBuffer<TValue>>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }
        }

        private Transposer EnsureAndGetTransposer(int col)
        {
            _host.Assert(0 <= col & col < _header.ColumnCount);
            // Used to "fake" row data when we don't actually have it.
            _host.Assert(!HasRowData);

            if (_colTransposers[col] == null)
            {
                lock (_colTransposersLock)
                {
                    if (_colTransposers[col] == null)
                    {
                        var view = _entries[col].GetViewOrNull();
                        // Since we don't have row-wise data, this view must exist.
                        _host.AssertValue(view);
                        _host.Assert(view.Schema.ColumnCount == 1);
                        var trans = _colTransposers[col] = Transposer.Create(_host, view, false, new int[] { 0 });
                        _host.Assert(trans.TransposeSchema.ColumnCount == 1);
                        _host.Assert(trans.TransposeSchema.GetSlotType(0).ValueCount == Schema.GetColumnType(col).ValueCount);
                    }
                }
            }
            _host.AssertValue(_colTransposers[col]);
            return _colTransposers[col];
        }

        private sealed class Cursor : RootCursorBase, IRowCursor
        {
            private readonly TransposeLoader _parent;
            private readonly int[] _actives;
            private readonly int[] _colToActivesIndex;
            private readonly ICursor[] _transCursors;
            private readonly Delegate[] _getters;
            private bool _disposed;

            public ISchema Schema { get { return _parent.Schema; } }

            public override long Batch { get { return 0; } }

            public Cursor(TransposeLoader parent, Func<int, bool> pred)
                : base(parent._host)
            {
                _parent = parent;
                Ch.AssertValue(pred);
                // We should only have instantiated this cursor if we have that
                // col transposers array, and we don't have row data in the file.
                Ch.AssertValue(_parent._colTransposers);
                Ch.AssertValue(_parent._colTransposersLock);
                Ch.Assert(!_parent.HasRowData);

                Utils.BuildSubsetMaps(_parent._header.ColumnCount, pred, out _actives, out _colToActivesIndex);
                _transCursors = new ICursor[_actives.Length];
                _getters = new Delegate[_actives.Length];
                // The following will fill in both the _transCursors and _getters arrays.
                for (int i = 0; i < _actives.Length; ++i)
                    Init(_actives[i]);
            }

            public override void Dispose()
            {
                if (!_disposed)
                {
                    _disposed = true;
                    for (int i = 0; i < _transCursors.Length; ++i)
                        _transCursors[i].Dispose();
                    base.Dispose();
                }
            }

            /// <summary>
            /// Initializes the transpose cursors and getters for a column.
            /// </summary>
            private void Init(int col)
            {
                Ch.Assert(0 <= col && col < Schema.ColumnCount);
                Ch.Assert(_colToActivesIndex[col] >= 0);
                var type = Schema.GetColumnType(col);
                Ch.Assert(_parent.TransposeSchema.GetSlotType(col).ValueCount == _parent._header.RowCount);
                Action<int> func = InitOne<int>;
                if (type.IsVector)
                    func = InitVec<int>;
                var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.ItemType.RawType);
                meth.Invoke(this, new object[] { col });
            }

            private void InitOne<T>(int col)
            {
                var type = Schema.GetColumnType(col);
                Ch.Assert(typeof(T) == type.RawType);
                var trans = _parent.EnsureAndGetTransposer(col);
                ISlotCursor cursor = trans.GetSlotCursor(0);
                ValueGetter<VBuffer<T>> getter = cursor.GetGetter<T>();
                VBuffer<T> buff = default(VBuffer<T>);
                ValueGetter<T> oneGetter =
                    (ref T value) =>
                    {
                        getter(ref buff);
                        Ch.Assert(buff.Length == 1);
                        buff.GetItemOrDefault(0, ref value);
                    };
                int i = _colToActivesIndex[col];
                _getters[i] = oneGetter;
                _transCursors[i] = cursor;
            }

            private void InitVec<T>(int col)
            {
                var type = Schema.GetColumnType(col);
                Ch.Assert(type.IsVector);
                Ch.Assert(typeof(T) == type.ItemType.RawType);
                var trans = _parent.EnsureAndGetTransposer(col);
                ISlotCursor cursor = trans.GetSlotCursor(0);
                ValueGetter<VBuffer<T>> getter = cursor.GetGetter<T>();
                int i = _colToActivesIndex[col];
                _getters[i] = getter;
                _transCursors[i] = cursor;
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

            protected override bool MoveNextCore()
            {
                Ch.Assert(State != CursorState.Done);
                bool more = Position < _parent._header.RowCount - 1;
                for (int i = 0; i < _transCursors.Length; ++i)
                {
                    bool cMore = _transCursors[i].MoveNext();
                    // All subcursors should agree on whether we've finished or not.
                    Ch.Assert(cMore == more);
                }
                return more;
            }

            protected override bool MoveManyCore(long count)
            {
                Ch.Assert(State != CursorState.Done);
                bool more = Position < _parent._header.RowCount - count;
                for (int i = 0; i < _transCursors.Length; ++i)
                {
                    bool cMore = _transCursors[i].MoveMany(count);
                    Ch.Assert(cMore == more);
                }
                return more;
            }

            public bool IsColumnActive(int col)
            {
                Ch.CheckParam(0 <= col && col <= _colToActivesIndex.Length, nameof(col));
                return _colToActivesIndex[col] >= 0;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.CheckParam(0 <= col && col <= _colToActivesIndex.Length, nameof(col));
                Ch.CheckParam(IsColumnActive(col), nameof(col), "requested column not active");

                Ch.AssertValue(_getters[_colToActivesIndex[col]]);
                var getter = _getters[_colToActivesIndex[col]] as ValueGetter<TValue>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }
        }
    }
}
