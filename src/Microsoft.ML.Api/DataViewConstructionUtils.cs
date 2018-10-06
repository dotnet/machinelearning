// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace Microsoft.ML.Runtime.Api
{
    /// <summary>
    /// A helper class to create data views based on the user-provided types.
    /// </summary>
    internal static class DataViewConstructionUtils
    {
        public static IDataView CreateFromList<TRow>(IHostEnvironment env, IList<TRow> data,
            SchemaDefinition schemaDefinition = null)
            where TRow : class
        {
            Contracts.AssertValue(env);
            env.AssertValue(data);
            env.AssertValueOrNull(schemaDefinition);
            var internalSchemaDefn = schemaDefinition == null
                ? InternalSchemaDefinition.Create(typeof(TRow), SchemaDefinition.Direction.Read)
                : InternalSchemaDefinition.Create(typeof(TRow), schemaDefinition);
            return new ListDataView<TRow>(env, data, internalSchemaDefn);
        }

        public static StreamingDataView<TRow> CreateFromEnumerable<TRow>(IHostEnvironment env, IEnumerable<TRow> data,
            SchemaDefinition schemaDefinition = null)
            where TRow : class
        {
            Contracts.AssertValue(env);
            env.AssertValue(data);
            env.AssertValueOrNull(schemaDefinition);
            var internalSchemaDefn = schemaDefinition == null
                ? InternalSchemaDefinition.Create(typeof(TRow), SchemaDefinition.Direction.Read)
                : InternalSchemaDefinition.Create(typeof(TRow), schemaDefinition);
            return new StreamingDataView<TRow>(env, data, internalSchemaDefn);
        }

        public static InputRow<TRow> CreateInputRow<TRow>(IHostEnvironment env, SchemaDefinition schemaDefinition = null)
            where TRow : class
        {
            Contracts.AssertValue(env);
            env.AssertValueOrNull(schemaDefinition);
            var internalSchemaDefn = schemaDefinition == null
                ? InternalSchemaDefinition.Create(typeof(TRow), SchemaDefinition.Direction.Read)
                : InternalSchemaDefinition.Create(typeof(TRow), schemaDefinition);

            return new InputRow<TRow>(env, internalSchemaDefn);
        }

        public static IDataView LoadPipeWithPredictor(IHostEnvironment env, Stream modelStream, IDataView view)
        {
            // Load transforms.
            var pipe = env.LoadTransforms(modelStream, view);

            // Load predictor (if present) and apply default scorer.
            // REVIEW: distinguish the case of predictor / no predictor?
            var predictor = env.LoadPredictorOrNull(modelStream);
            if (predictor != null)
            {
                var roles = ModelFileUtils.LoadRoleMappingsOrNull(env, modelStream);
                pipe = roles != null
                    ? env.CreateDefaultScorer(new RoleMappedData(pipe, roles, opt: true), predictor)
                    : env.CreateDefaultScorer(new RoleMappedData(pipe, label: null, "Features"), predictor);
            }
            return pipe;
        }

        public sealed class InputRow<TRow> : InputRowBase<TRow>, IRowBackedBy<TRow>
            where TRow : class
        {
            private TRow _value;

            private long _position;
            public override long Position => _position;

            public InputRow(IHostEnvironment env, InternalSchemaDefinition schemaDef)
                : base(env, new Schema(GetSchemaColumns(schemaDef)), schemaDef, MakePeeks(schemaDef), c => true)
            {
                _position = -1;
            }

            private static Delegate[] MakePeeks(InternalSchemaDefinition schemaDef)
            {
                var peeks = new Delegate[schemaDef.Columns.Length];
                for (var i = 0; i < peeks.Length; i++)
                {
                    var currentColumn = schemaDef.Columns[i];
                    peeks[i] = currentColumn.IsComputed
                        ? currentColumn.Generator
                        : ApiUtils.GeneratePeek<InputRow<TRow>, TRow>(currentColumn);
                }
                return peeks;
            }

            public void ExtractValues(TRow row)
            {
                Host.CheckValue(row, nameof(row));
                _value = row;
                _position++;
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                return IdGetter;
            }

            private void IdGetter(ref UInt128 val) => val = new UInt128((ulong)Position, 0);

            protected override TRow GetCurrentRowObject()
            {
                Host.Check(Position >= 0, "Can't call a getter on an inactive cursor.");
                return _value;
            }
        }

        /// <summary>
        /// A row that consumes items of type <typeparamref name="TRow"/>, and provides an <see cref="IRow"/>. This
        /// is in contrast to <see cref="IRowReadableAs{TRow}"/> which consumes a data view row and publishes them as the output type.
        /// </summary>
        /// <typeparam name="TRow">The input data type.</typeparam>
        public abstract class InputRowBase<TRow> : IRow
            where TRow : class
        {
            private readonly int _colCount;
            private readonly Delegate[] _getters;
            protected readonly IHost Host;

            public long Batch => 0;

            public Schema Schema { get; }

            public abstract long Position { get; }

            public InputRowBase(IHostEnvironment env, Schema schema, InternalSchemaDefinition schemaDef, Delegate[] peeks, Func<int, bool> predicate)
            {
                Contracts.AssertValue(env);
                Host = env.Register("Row");
                Host.AssertValue(schema);
                Host.AssertValue(schemaDef);
                Host.AssertValue(peeks);
                Host.AssertValue(predicate);
                Host.Assert(schema.ColumnCount == schemaDef.Columns.Length);
                Host.Assert(schema.ColumnCount == peeks.Length);

                _colCount = schema.ColumnCount;
                Schema = schema;
                _getters = new Delegate[_colCount];
                for (int c = 0; c < _colCount; c++)
                    _getters[c] = predicate(c) ? CreateGetter(schema.GetColumnType(c), schemaDef.Columns[c], peeks[c]) : null;
            }

            //private Delegate CreateGetter(SchemaProxy schema, int index, Delegate peek)
            private Delegate CreateGetter(ColumnType colType, InternalSchemaDefinition.Column column, Delegate peek)
            {
                var outputType = column.OutputType;
                var genericType = outputType;
                Func<Delegate, Delegate> del;

                if (outputType.IsArray)
                {
                    Host.Assert(colType.IsVector);
                    // String[] -> ReadOnlyMemory<char>
                    if (outputType.GetElementType() == typeof(string))
                    {
                        Host.Assert(colType.ItemType.IsText);
                        return CreateConvertingArrayGetterDelegate<string, ReadOnlyMemory<char>>(peek, x => x != null ? x.AsMemory() : ReadOnlyMemory<char>.Empty);
                    }

                    // T[] -> VBuffer<T>
                    if (outputType.GetElementType().IsGenericType && outputType.GetElementType().GetGenericTypeDefinition() == typeof(Nullable<>))
                        Host.Assert(Nullable.GetUnderlyingType(outputType.GetElementType()) == colType.ItemType.RawType);
                    else
                        Host.Assert(outputType.GetElementType() == colType.ItemType.RawType);
                    del = CreateDirectArrayGetterDelegate<int>;
                    genericType = outputType.GetElementType();
                }
                else if (colType.IsVector)
                {
                    // VBuffer<T> -> VBuffer<T>
                    // REVIEW: Do we care about accomodating VBuffer<string> -> ReadOnlyMemory<char>?
                    Host.Assert(outputType.IsGenericType);
                    Host.Assert(outputType.GetGenericTypeDefinition() == typeof(VBuffer<>));
                    Host.Assert(outputType.GetGenericArguments()[0] == colType.ItemType.RawType);
                    del = CreateDirectVBufferGetterDelegate<int>;
                    genericType = colType.ItemType.RawType;
                }
                else if (colType.IsPrimitive)
                {
                    if (outputType == typeof(string))
                    {
                        // String -> ReadOnlyMemory<char>
                        Host.Assert(colType.IsText);
                        return CreateConvertingGetterDelegate<String, ReadOnlyMemory<char>>(peek, x => x != null ? x.AsMemory() : ReadOnlyMemory<char>.Empty);
                    }

                    // T -> T
                    if (outputType.IsGenericType && outputType.GetGenericTypeDefinition() == typeof(Nullable<>))
                        Host.Assert(colType.RawType == Nullable.GetUnderlyingType(outputType));
                    else
                        Host.Assert(colType.RawType == outputType);
                    del = CreateDirectGetterDelegate<int>;
                }
                else
                {
                    // REVIEW: Is this even possible?
                    throw Host.ExceptNotSupp("Type '{0}' is not yet supported.", outputType.FullName);
                }
                return Utils.MarshalInvoke(del, genericType, peek);
            }

            // REVIEW: The converting getter invokes a type conversion delegate on every call, so it's inherently slower
            // than the 'direct' getter. We don't have good indication of this to the user, and the selection
            // of affected types is pretty arbitrary (signed integers and bools, but not uints and floats).
            private Delegate CreateConvertingArrayGetterDelegate<TSrc, TDst>(Delegate peekDel, Func<TSrc, TDst> convert)
            {
                var peek = peekDel as Peek<TRow, TSrc[]>;
                Host.AssertValue(peek);
                TSrc[] buf = default;
                return (ValueGetter<VBuffer<TDst>>)((ref VBuffer<TDst> dst) =>
                {
                    peek(GetCurrentRowObject(), Position, ref buf);
                    var n = Utils.Size(buf);
                    dst = new VBuffer<TDst>(n, Utils.Size(dst.Values) < n
                        ? new TDst[n]
                        : dst.Values, dst.Indices);
                    for (int i = 0; i < n; i++)
                        dst.Values[i] = convert(buf[i]);
                });
            }

            private Delegate CreateConvertingGetterDelegate<TSrc, TDst>(Delegate peekDel, Func<TSrc, TDst> convert)
            {
                var peek = peekDel as Peek<TRow, TSrc>;
                Host.AssertValue(peek);
                TSrc buf = default;
                return (ValueGetter<TDst>)((ref TDst dst) =>
                {
                    peek(GetCurrentRowObject(), Position, ref buf);
                    dst = convert(buf);
                });
            }

            private Delegate CreateDirectArrayGetterDelegate<TDst>(Delegate peekDel)
            {
                var peek = peekDel as Peek<TRow, TDst[]>;
                Host.AssertValue(peek);
                TDst[] buf = null;
                return (ValueGetter<VBuffer<TDst>>)((ref VBuffer<TDst> dst) =>
                {
                    peek(GetCurrentRowObject(), Position, ref buf);
                    var n = Utils.Size(buf);
                    dst = new VBuffer<TDst>(n, Utils.Size(dst.Values) < n ? new TDst[n] : dst.Values,
                        dst.Indices);
                    if (buf != null)
                        Array.Copy(buf, dst.Values, n);
                });
            }

            private Delegate CreateDirectVBufferGetterDelegate<TDst>(Delegate peekDel)
            {
                var peek = peekDel as Peek<TRow, VBuffer<TDst>>;
                Host.AssertValue(peek);
                VBuffer<TDst> buf = default;
                return (ValueGetter<VBuffer<TDst>>)((ref VBuffer<TDst> dst) =>
                {
                    // The peek for a VBuffer is just a simple assignment, so there is
                    // no copy going on in the peek, so we must do that as a second
                    // step to the destination.
                    peek(GetCurrentRowObject(), Position, ref buf);
                    buf.CopyTo(ref dst);
                });
            }

            private Delegate CreateDirectGetterDelegate<TDst>(Delegate peekDel)
            {
                var peek = peekDel as Peek<TRow, TDst>;
                Host.AssertValue(peek);
                return (ValueGetter<TDst>)((ref TDst dst) =>
                    peek(GetCurrentRowObject(), Position, ref dst));
            }

            protected abstract TRow GetCurrentRowObject();

            public bool IsColumnActive(int col)
            {
                CheckColumnInRange(col);
                return _getters[col] != null;
            }

            private void CheckColumnInRange(int columnIndex)
            {
                if (columnIndex < 0 || columnIndex >= _colCount)
                    throw Host.Except("Column index must be between 0 and {0}", _colCount);
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                if (!IsColumnActive(col))
                    throw Host.Except("Column {0} is not active in the cursor", col);
                var getter = _getters[col];
                Contracts.AssertValue(getter);
                var fn = getter as ValueGetter<TValue>;
                if (fn == null)
                    throw Host.Except("Invalid TValue in GetGetter for column #{0}: '{1}'", col, typeof(TValue));
                return fn;
            }

            public abstract ValueGetter<UInt128> GetIdGetter();
        }

        /// <summary>
        /// The base class for the data view over items of user-defined type.
        /// </summary>
        /// <typeparam name="TRow">The user-defined data type.</typeparam>
        public abstract class DataViewBase<TRow> : IDataView
            where TRow : class
        {
            protected readonly IHost Host;

            private readonly Schema _schema;
            private readonly InternalSchemaDefinition _schemaDefn;

            // The array of generated methods that extract the fields of the current row object.
            private readonly Delegate[] _peeks;

            public abstract bool CanShuffle { get; }

            public Schema Schema => _schema;

            protected DataViewBase(IHostEnvironment env, string name, InternalSchemaDefinition schemaDefn)
            {
                Contracts.AssertValue(env);
                env.AssertNonWhiteSpace(name);
                Host = env.Register(name);
                Host.AssertValue(schemaDefn);

                _schemaDefn = schemaDefn;
                _schema = new Schema(GetSchemaColumns(schemaDefn));
                int n = schemaDefn.Columns.Length;
                _peeks = new Delegate[n];
                for (var i = 0; i < n; i++)
                {
                    var currentColumn = schemaDefn.Columns[i];
                    _peeks[i] = currentColumn.IsComputed
                        ? currentColumn.Generator
                        : ApiUtils.GeneratePeek<DataViewBase<TRow>, TRow>(currentColumn);
                }
            }

            public abstract long? GetRowCount(bool lazy = true);

            public abstract IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null);

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate,
                int n, IRandom rand = null)
            {
                consolidator = null;
                return new[] { GetRowCursor(predicate, rand) };
            }

            public abstract class DataViewCursorBase : InputRowBase<TRow>, IRowCursor
            {
                // There is no real concept of multiple inheritance and for various reasons it was better to
                // descend from the row class as opposed to wrapping it, so much of this class is regrettably
                // copied from RootCursorBase.

                protected readonly DataViewBase<TRow> DataView;
                protected readonly IChannel Ch;

                private long _position;
                /// <summary>
                /// Zero-based position of the cursor.
                /// </summary>
                public override long Position => _position;

                protected DataViewCursorBase(IHostEnvironment env, DataViewBase<TRow> dataView,
                    Func<int, bool> predicate)
                    : base(env, dataView.Schema, dataView._schemaDefn, dataView._peeks, predicate)
                {
                    Contracts.AssertValue(env);
                    Ch = env.Start("Cursor");
                    Ch.AssertValue(dataView);
                    Ch.AssertValue(predicate);

                    DataView = dataView;
                    _position = -1;
                    State = CursorState.NotStarted;
                }

                public CursorState State { get; private set; }

                /// <summary>
                /// Convenience property for checking whether the current state of the cursor is <see cref="CursorState.Good"/>.
                /// </summary>
                protected bool IsGood => State == CursorState.Good;

                public virtual void Dispose()
                {
                    if (State != CursorState.Done)
                    {
                        Ch.Done();
                        Ch.Dispose();
                        _position = -1;
                        State = CursorState.Done;
                    }
                }

                public bool MoveNext()
                {
                    if (State == CursorState.Done)
                        return false;

                    Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);
                    if (MoveNextCore())
                    {
                        Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);

                        _position++;
                        State = CursorState.Good;
                        return true;
                    }

                    Dispose();
                    return false;
                }

                public bool MoveMany(long count)
                {
                    // Note: If we decide to allow count == 0, then we need to special case
                    // that MoveNext() has never been called. It's not entirely clear what the return
                    // result would be in that case.
                    Ch.CheckParam(count > 0, nameof(count));

                    if (State == CursorState.Done)
                        return false;

                    Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);
                    if (MoveManyCore(count))
                    {
                        Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);

                        _position += count;
                        State = CursorState.Good;
                        return true;
                    }

                    Dispose();
                    return false;
                }

                /// <summary>
                /// Default implementation is to simply call MoveNextCore repeatedly. Derived classes should
                /// override if they can do better.
                /// </summary>
                /// <param name="count">The number of rows to move forward.</param>
                /// <returns>Whether the move forward is on a valid row</returns>
                protected virtual bool MoveManyCore(long count)
                {
                    Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);
                    Ch.Assert(count > 0);

                    while (MoveNextCore())
                    {
                        Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);
                        if (--count <= 0)
                            return true;
                    }

                    return false;
                }

                /// <summary>
                /// Core implementation of <see cref="MoveNext"/>, called if the cursor state is not
                /// <see cref="CursorState.Done"/>.
                /// </summary>
                protected abstract bool MoveNextCore();

                /// <summary>
                /// Returns a cursor that can be used for invoking <see cref="Position"/>, <see cref="State"/>,
                /// <see cref="MoveNext"/>, and <see cref="MoveMany(long)"/>, with results identical to calling
                /// those on this cursor. Generally, if the root cursor is not the same as this cursor, using
                /// the root cursor will be faster.
                /// </summary>
                public ICursor GetRootCursor() => this;
            }
        }

        /// <summary>
        /// An in-memory data view based on the IList of data.
        /// Supports shuffling.
        /// </summary>
        private sealed class ListDataView<TRow> : DataViewBase<TRow>
            where TRow : class
        {
            private readonly IList<TRow> _data;

            public ListDataView(IHostEnvironment env, IList<TRow> data, InternalSchemaDefinition schemaDefn)
                : base(env, "ListDataView", schemaDefn)
            {
                Host.CheckValue(data, nameof(data));
                _data = data;
            }

            public override bool CanShuffle
            {
                get { return true; }
            }

            public override long? GetRowCount(bool lazy = true)
            {
                return _data.Count;
            }

            public override IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                Host.CheckValue(predicate, nameof(predicate));
                return new Cursor(Host, "ListDataView", this, predicate, rand);
            }

            private sealed class Cursor : DataViewCursorBase
            {
                private readonly int[] _permutation;
                private readonly IList<TRow> _data;

                private int Index
                {
                    get { return _permutation == null ? (int)Position : _permutation[(int)Position]; }
                }

                public Cursor(IHostEnvironment env, string name, ListDataView<TRow> dataView,
                    Func<int, bool> predicate, IRandom rand)
                    : base(env, dataView, predicate)
                {
                    Ch.AssertValueOrNull(rand);
                    _data = dataView._data;
                    if (rand != null)
                        _permutation = Utils.GetRandomPermutation(rand, dataView._data.Count);
                }

                public override ValueGetter<UInt128> GetIdGetter()
                {
                    if (_permutation == null)
                    {
                        return
                            (ref UInt128 val) =>
                            {
                                Ch.Check(IsGood, "Cannot call ID getter in current state");
                                val = new UInt128((ulong)Position, 0);
                            };
                    }
                    else
                    {
                        return
                            (ref UInt128 val) =>
                            {
                                Ch.Check(IsGood, "Cannot call ID getter in current state");
                                val = new UInt128((ulong)Index, 0);
                            };
                    }
                }

                protected override TRow GetCurrentRowObject()
                {
                    Ch.Check(0 <= Position && Position < _data.Count, "Can't call a getter on an inactive cursor.");
                    return _data[Index];
                }

                protected override bool MoveNextCore()
                {
                    Ch.Assert(State != CursorState.Done);
                    Ch.Assert(Position < _data.Count);
                    return Position + 1 < _data.Count;
                }

                protected override bool MoveManyCore(long count)
                {
                    Ch.Assert(State != CursorState.Done);
                    Ch.Assert(Position < _data.Count);
                    return count < _data.Count - Position;
                }
            }
        }

        /// <summary>
        /// An in-memory data view based on the IEnumerable of data.
        /// Doesn't support shuffling.
        ///
        /// This class is public because prediction engine wants to call its <see cref="SetData"/>
        /// for performance reasons.
        /// </summary>
        public sealed class StreamingDataView<TRow> : DataViewBase<TRow>
            where TRow : class
        {
            private IEnumerable<TRow> _data;

            public StreamingDataView(IHostEnvironment env, IEnumerable<TRow> data, InternalSchemaDefinition schemaDefn)
                : base(env, "StreamingDataView", schemaDefn)
            {
                Contracts.CheckValue(data, nameof(data));
                _data = data;
            }

            public override bool CanShuffle
            {
                get { return false; }
            }

            public override long? GetRowCount(bool lazy = true)
            {
                return (_data as ICollection<TRow>)?.Count;
            }

            public override IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                return new Cursor(Host, this, predicate);
            }

            /// <summary>
            /// Since all the cursors only depend on an enumerator (rather than the data itself),
            /// it's safe to 'swap' the data inside the streaming data view. This doesn't affect
            /// the current 'live' cursors, only the ones that will be created later.
            /// This is used for efficiency in <see cref="BatchPredictionEngine{TSrc,TDst}"/>.
            /// </summary>
            public void SetData(IEnumerable<TRow> data)
            {
                Contracts.CheckValue(data, nameof(data));
                _data = data;
            }

            private class Cursor : DataViewCursorBase
            {
                private readonly IEnumerator<TRow> _enumerator;
                private TRow _currentRow;

                public Cursor(IHostEnvironment env, StreamingDataView<TRow> dataView, Func<int, bool> predicate)
                    : base(env, dataView, predicate)
                {
                    _enumerator = dataView._data.GetEnumerator();
                    _currentRow = null;
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

                protected override TRow GetCurrentRowObject()
                {
                    return _currentRow;
                }

                protected override bool MoveNextCore()
                {
                    Ch.Assert(State != CursorState.Done);
                    var result = _enumerator.MoveNext();
                    _currentRow = result ? _enumerator.Current : null;
                    if (result && _currentRow == null)
                        throw Ch.Except("Encountered null when iterating over data, this is not supported.");
                    return result;
                }
            }
        }

        /// <summary>
        /// This represents the 'infinite data view' over one (mutable) user-defined object.
        /// The 'current row' object can be updated at any time, this will affect all the
        /// newly created cursors, but not the ones already existing.
        /// </summary>
        public sealed class SingleRowLoopDataView<TRow> : DataViewBase<TRow>
            where TRow : class
        {
            private TRow _current;

            public SingleRowLoopDataView(IHostEnvironment env, InternalSchemaDefinition schemaDefn)
                : base(env, "SingleRowLoopDataView", schemaDefn)
            {
            }

            public override bool CanShuffle
            {
                get { return false; }
            }

            public override long? GetRowCount(bool lazy = true)
            {
                return null;
            }

            public void SetCurrentRowObject(TRow value)
            {
                Host.AssertValue(value);
                _current = value;
            }

            public override IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                Contracts.Assert(_current != null, "The current object must be set prior to cursoring");
                return new Cursor(Host, this, predicate);
            }

            private sealed class Cursor : DataViewCursorBase
            {
                private readonly TRow _currentRow;

                public Cursor(IHostEnvironment env, SingleRowLoopDataView<TRow> dataView, Func<int, bool> predicate)
                    : base(env, dataView, predicate)
                {
                    _currentRow = dataView._current;
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

                protected override TRow GetCurrentRowObject()
                {
                    return _currentRow;
                }

                protected override bool MoveNextCore()
                {
                    Ch.Assert(State != CursorState.Done);
                    return true;
                }

                protected override bool MoveManyCore(long count)
                {
                    Ch.Assert(State != CursorState.Done);
                    return true;
                }
            }
        }

        internal static Schema.Column[] GetSchemaColumns(InternalSchemaDefinition schemaDefn)
        {
            Contracts.AssertValue(schemaDefn);
            var columns = new Schema.Column[schemaDefn.Columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                var col = schemaDefn.Columns[i];
                var meta = new Schema.MetadataRow.Builder();
                foreach (var kvp in col.Metadata)
                    meta.Add(new Schema.Column(kvp.Value.Kind, kvp.Value.MetadataType, null), kvp.Value.GetGetterDelegate());
                columns[i] = new Schema.Column(col.ColumnName, col.ColumnType, meta.GetMetadataRow());
            }

            return columns;
        }
    }

    /// <summary>
    /// A single instance of metadata information, associated with a column.
    /// </summary>
    public abstract partial class MetadataInfo
    {
        /// <summary>
        /// The type of the metadata.
        /// </summary>
        public ColumnType MetadataType;
        /// <summary>
        /// The string identifier of the metadata. Some identifiers have special meaning,
        /// like "SlotNames", but any other identifiers can be used.
        /// </summary>
        public readonly string Kind;

        public abstract ValueGetter<TDst> GetGetter<TDst>();

        internal abstract Delegate GetGetterDelegate();

        protected MetadataInfo(string kind, ColumnType metadataType)
        {
            Contracts.AssertValueOrNull(metadataType);
            Contracts.AssertNonEmpty(kind);
            Kind = kind;
        }
    }

    /// <summary>
    /// Strongly-typed version of <see cref="MetadataInfo"/>, that contains the actual value of the metadata.
    /// </summary>
    /// <typeparam name="T">Type of the metadata value.</typeparam>
    public sealed class MetadataInfo<T> : MetadataInfo
    {
        public readonly T Value;

        /// <summary>
        /// Constructor for metadata of value type T.
        /// </summary>
        /// <param name="kind">The string identifier of the metadata. Some identifiers have special meaning,
        /// like "SlotNames", but any other identifiers can be used.</param>
        /// <param name="value">Metadata value.</param>
        /// <param name="metadataType">Type of the metadata.</param>
        public MetadataInfo(string kind, T value, ColumnType metadataType = null)
            : base(kind, metadataType)
        {
            Contracts.Assert(value != null);
            bool isVector;
            DataKind dataKind;
            InternalSchemaDefinition.GetVectorAndKind(typeof(T), "metadata value", out isVector, out dataKind);

            if (metadataType == null)
            {
                // Infer a type as best we can.
                var itemType = PrimitiveType.FromKind(dataKind);
                metadataType = isVector ? new VectorType(itemType) : (ColumnType)itemType;
            }
            else
            {
                // Make sure that the types are compatible with the declared type, including whether it is a vector type.
                if (isVector != metadataType.IsVector)
                {
                    throw Contracts.Except("Value inputted is supposed to be {0}, but type of Metadatainfo is {1}",
                        isVector ? "vector" : "scalar", metadataType.IsVector ? "vector" : "scalar");
                }
                if (dataKind != metadataType.ItemType.RawKind)
                {
                    throw Contracts.Except(
                        "Value inputted is supposed to have dataKind {0}, but type of Metadatainfo has {1}",
                        dataKind.ToString(), metadataType.ItemType.RawKind.ToString());
                }
            }
            MetadataType = metadataType;
            Value = value;
        }

        public override ValueGetter<TDst> GetGetter<TDst>()
        {
            var typeT = typeof(T);
            if (typeT.IsArray)
            {
                Contracts.Assert(MetadataType.IsVector);
                Contracts.Check(typeof(TDst).IsGenericType && typeof(TDst).GetGenericTypeDefinition() == typeof(VBuffer<>));
                var itemType = typeT.GetElementType();
                var dstItemType = typeof(TDst).GetGenericArguments()[0];

                // String[] -> VBuffer<ReadOnlyMemory<char>>
                if (itemType == typeof(string))
                {
                    Contracts.Check(dstItemType == typeof(ReadOnlyMemory<char>));

                    ValueGetter<VBuffer<ReadOnlyMemory<char>>> method = GetStringArray;
                    return method as ValueGetter<TDst>;
                }

                // T[] -> VBuffer<T>
                Contracts.Check(itemType == dstItemType);

                Func<ValueGetter<VBuffer<int>>> srcMethod = GetArrayGetter<int>;

                return srcMethod.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(dstItemType)
                    .Invoke(this, new object[] { }) as ValueGetter<TDst>;
            }
            if (MetadataType.IsVector)
            {
                // VBuffer<T> -> VBuffer<T>
                // REVIEW: Do we care about accomodating VBuffer<string> -> VBuffer<ReadOnlyMemory<char>>?

                Contracts.Assert(typeT.IsGenericType);
                Contracts.Check(typeof(TDst).IsGenericType);
                Contracts.Assert(typeT.GetGenericTypeDefinition() == typeof(VBuffer<>));
                Contracts.Check(typeof(TDst).GetGenericTypeDefinition() == typeof(VBuffer<>));
                var dstItemType = typeof(TDst).GetGenericArguments()[0];
                var itemType = typeT.GetGenericArguments()[0];
                Contracts.Assert(itemType == MetadataType.ItemType.RawType);
                Contracts.Check(itemType == dstItemType);

                Func<ValueGetter<VBuffer<int>>> srcMethod = GetVBufferGetter<int>;
                return srcMethod.GetMethodInfo().GetGenericMethodDefinition()
                    .MakeGenericMethod(MetadataType.ItemType.RawType)
                    .Invoke(this, new object[] { }) as ValueGetter<TDst>;
            }
            if (MetadataType.IsPrimitive)
            {
                if (typeT == typeof(string))
                {
                    // String -> ReadOnlyMemory<char>
                    Contracts.Assert(MetadataType.IsText);
                    ValueGetter<ReadOnlyMemory<char>> m = GetString;
                    return m as ValueGetter<TDst>;
                }
                // T -> T
                Contracts.Assert(MetadataType.RawType == typeT);
                return GetDirectValue;
            }
            throw Contracts.ExceptNotImpl("Type '{0}' is not yet supported.", typeT.FullName);
        }

        internal override Delegate GetGetterDelegate() => Utils.MarshalInvoke(GetGetter<int>, MetadataType.RawType);

        public class TElement
        {
        }

        private void GetStringArray(ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            var value = (string[])(object)Value;
            var n = Utils.Size(value);
            dst = new VBuffer<ReadOnlyMemory<char>>(n, Utils.Size(dst.Values) < n ? new ReadOnlyMemory<char>[n] : dst.Values, dst.Indices);

            for (int i = 0; i < n; i++)
                dst.Values[i] = value[i].AsMemory();

        }

        private ValueGetter<VBuffer<TDst>> GetArrayGetter<TDst>()
        {
            var value = (TDst[])(object)Value;
            var n = Utils.Size(value);
            return (ref VBuffer<TDst> dst) =>
            {
                dst = new VBuffer<TDst>(n, Utils.Size(dst.Values) < n ? new TDst[n] : dst.Values, dst.Indices);
                if (value != null)
                    Array.Copy(value, dst.Values, n);
            };
        }

        private ValueGetter<VBuffer<TDst>> GetVBufferGetter<TDst>()
        {
            var castValue = (VBuffer<TDst>)(object)Value;
            return (ref VBuffer<TDst> dst) => castValue.CopyTo(ref dst);
        }

        private void GetString(ref ReadOnlyMemory<char> dst)
        {
            dst = ((string)(object)Value).AsMemory();
        }

        private void GetDirectValue<TDst>(ref TDst dst)
        {
            dst = (TDst)(object)Value;
        }
    }
}