// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A helper class to create data views based on the user-provided types.
    /// </summary>
    [BestFriend]
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

        public static StreamingDataView<TRow> CreateFromEnumerable<TRow>(IHostEnvironment env, IEnumerable<TRow> data,
            DataViewSchema schema)
            where TRow : class
        {
            Contracts.AssertValue(env);
            env.AssertValue(data);
            env.AssertValueOrNull(schema);
            schema = schema ?? new DataViewSchema.Builder().ToSchema();
            return new StreamingDataView<TRow>(env, data, GetInternalSchemaDefinition<TRow>(env, schema));
        }

        internal static SchemaDefinition GetSchemaDefinition<TRow>(IHostEnvironment env, DataViewSchema schema)
        {
            Contracts.AssertValue(env);
            env.AssertValue(schema);

            var schemaDefinition = SchemaDefinition.Create(typeof(TRow), SchemaDefinition.Direction.Read);
            foreach (var col in schema)
            {
                var name = col.Name;
                var schemaDefinitionCol = schemaDefinition.FirstOrDefault(c => c.ColumnName == name);
                if (schemaDefinitionCol == null)
                    throw env.Except($"Type should contain a member named {name}");
                var annotations = col.Annotations;
                if (annotations != null)
                {
                    foreach (var annotation in annotations.Schema)
                    {
                        var info = Utils.MarshalInvoke(GetAnnotationInfo<int>, annotation.Type.RawType, annotation.Name, annotations);
                        schemaDefinitionCol.Annotations.Add(annotation.Name, info);
                    }
                }
            }
            return schemaDefinition;
        }

        private static InternalSchemaDefinition GetInternalSchemaDefinition<TRow>(IHostEnvironment env, DataViewSchema schema)
            where TRow : class
        {
            Contracts.AssertValue(env);
            env.AssertValue(schema);
            return InternalSchemaDefinition.Create(typeof(TRow), GetSchemaDefinition<TRow>(env, schema));
        }

        private static AnnotationInfo GetAnnotationInfo<T>(string kind, DataViewSchema.Annotations annotations)
        {
            T value = default;
            annotations.GetValue(kind, ref value);
            return new AnnotationInfo<T>(kind, value, annotations.Schema[kind].Type);
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

        public sealed class InputRow<TRow> : InputRowBase<TRow>
            where TRow : class
        {
            private TRow _value;

            private long _position;
            public override long Position => _position;

            public InputRow(IHostEnvironment env, InternalSchemaDefinition schemaDef)
                : base(env, SchemaExtensions.MakeSchema(GetSchemaColumns(schemaDef)), schemaDef, MakePeeks(schemaDef), c => true)
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

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return IdGetter;
            }

            private void IdGetter(ref DataViewRowId val) => val = new DataViewRowId((ulong)Position, 0);

            protected override TRow GetCurrentRowObject()
            {
                Host.Check(Position >= 0, RowCursorUtils.FetchValueStateError);
                return _value;
            }
        }

        /// <summary>
        /// A row that consumes items of type <typeparamref name="TRow"/>, and provides an <see cref="DataViewRow"/>. This
        /// is in contrast to <see cref="IRowReadableAs{TRow}"/> which consumes a data view row and publishes them as the output type.
        /// </summary>
        /// <typeparam name="TRow">The input data type.</typeparam>
        public abstract class InputRowBase<TRow> : DataViewRow
            where TRow : class
        {
            private readonly int _colCount;
            private readonly Delegate[] _getters;
            protected readonly IHost Host;

            public override long Batch => 0;

            public override DataViewSchema Schema { get; }

            public InputRowBase(IHostEnvironment env, DataViewSchema schema, InternalSchemaDefinition schemaDef, Delegate[] peeks, Func<int, bool> predicate)
            {
                Contracts.AssertValue(env);
                Host = env.Register("Row");
                Host.AssertValue(schema);
                Host.AssertValue(schemaDef);
                Host.AssertValue(peeks);
                Host.AssertValue(predicate);
                Host.Assert(schema.Count == schemaDef.Columns.Length);
                Host.Assert(schema.Count == peeks.Length);

                _colCount = schema.Count;
                Schema = schema;
                _getters = new Delegate[_colCount];
                for (int c = 0; c < _colCount; c++)
                    _getters[c] = predicate(c) ? CreateGetter(schema[c].Type, schemaDef.Columns[c], peeks[c]) : null;
            }

            //private Delegate CreateGetter(SchemaProxy schema, int index, Delegate peek)
            private Delegate CreateGetter(DataViewType colType, InternalSchemaDefinition.Column column, Delegate peek)
            {
                var outputType = column.OutputType;
                var genericType = outputType;
                Func<Delegate, Delegate> del;

                if (outputType.IsArray)
                {
                    VectorType vectorType = colType as VectorType;
                    Host.Assert(vectorType != null);

                    // String[] -> ReadOnlyMemory<char>
                    if (outputType.GetElementType() == typeof(string))
                    {
                        Host.Assert(vectorType.ItemType is TextDataViewType);
                        return CreateConvertingArrayGetterDelegate<string, ReadOnlyMemory<char>>(peek, x => x != null ? x.AsMemory() : ReadOnlyMemory<char>.Empty);
                    }

                    // T[] -> VBuffer<T>
                    if (outputType.GetElementType().IsGenericType && outputType.GetElementType().GetGenericTypeDefinition() == typeof(Nullable<>))
                        Host.Assert(Nullable.GetUnderlyingType(outputType.GetElementType()) == vectorType.ItemType.RawType);
                    else
                        Host.Assert(outputType.GetElementType() == vectorType.ItemType.RawType);
                    del = CreateDirectArrayGetterDelegate<int>;
                    genericType = outputType.GetElementType();
                }
                else if (colType is VectorType vectorType)
                {
                    // VBuffer<T> -> VBuffer<T>
                    // REVIEW: Do we care about accomodating VBuffer<string> -> ReadOnlyMemory<char>?
                    Host.Assert(outputType.IsGenericType);
                    Host.Assert(outputType.GetGenericTypeDefinition() == typeof(VBuffer<>));
                    Host.Assert(outputType.GetGenericArguments()[0] == vectorType.ItemType.RawType);
                    del = CreateDirectVBufferGetterDelegate<int>;
                    genericType = vectorType.ItemType.RawType;
                }
                else if (colType is PrimitiveDataViewType)
                {
                    if (outputType == typeof(string))
                    {
                        // String -> ReadOnlyMemory<char>
                        Host.Assert(colType is TextDataViewType);
                        return CreateConvertingGetterDelegate<String, ReadOnlyMemory<char>>(peek, x => x != null ? x.AsMemory() : ReadOnlyMemory<char>.Empty);
                    }

                    // T -> T
                    if (outputType.IsGenericType && outputType.GetGenericTypeDefinition() == typeof(Nullable<>))
                        Host.Assert(colType.RawType == Nullable.GetUnderlyingType(outputType));
                    else
                        Host.Assert(colType.RawType == outputType);

                    if (!(colType is KeyType keyType))
                        del = CreateDirectGetterDelegate<int>;
                    else
                    {
                        var keyRawType = colType.RawType;
                        Func<Delegate, DataViewType, Delegate> delForKey = CreateKeyGetterDelegate<uint>;
                        return Utils.MarshalInvoke(delForKey, keyRawType, peek, colType);
                    }
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
                    var dstEditor = VBufferEditor.Create(ref dst, n);
                    for (int i = 0; i < n; i++)
                        dstEditor.Values[i] = convert(buf[i]);
                    dst = dstEditor.Commit();
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
                    var dstEditor = VBufferEditor.Create(ref dst, n);
                    if (buf != null)
                        buf.AsSpan(0, n).CopyTo(dstEditor.Values);
                    dst = dstEditor.Commit();
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

            private Delegate CreateKeyGetterDelegate<TDst>(Delegate peekDel, DataViewType colType)
            {
                // Make sure the function is dealing with key.
                KeyType keyType = colType as KeyType;
                Host.Check(keyType != null);
                // Following equations work only with unsigned integers.
                Host.Check(typeof(TDst) == typeof(ulong) || typeof(TDst) == typeof(uint) ||
                    typeof(TDst) == typeof(byte) || typeof(TDst) == typeof(bool));

                // Convert delegate function to a function which can fetch the underlying value.
                var peek = peekDel as Peek<TRow, TDst>;
                Host.AssertValue(peek);

                TDst rawKeyValue = default;
                ulong key = 0; // the raw key value as ulong
                ulong max = keyType.Count - 1;
                ulong result = 0; // the result as ulong
                ValueGetter<TDst> getter = (ref TDst dst) =>
                {
                    peek(GetCurrentRowObject(), Position, ref rawKeyValue);
                    key = (ulong)Convert.ChangeType(rawKeyValue, typeof(ulong));
                    if (key <= max)
                        result = key + 1;
                    else
                        result = 0;
                    dst = (TDst)Convert.ChangeType(result, typeof(TDst));
                };
                return getter;
            }

            protected abstract TRow GetCurrentRowObject();

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                CheckColumnInRange(column.Index);
                return _getters[column.Index] != null;
            }

            private void CheckColumnInRange(int columnIndex)
            {
                if (columnIndex < 0 || columnIndex >= _colCount)
                    throw Host.Except("Column index must be between 0 and {0}", _colCount);
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Host.CheckParam(column.Index <= _getters.Length && IsColumnActive(column), nameof(column), "requested column not active");

                var getter = _getters[column.Index];
                Contracts.AssertValue(getter);
                var fn = getter as ValueGetter<TValue>;
                if (fn == null)
                    throw Host.Except("Invalid TValue in GetGetter for column #{0}: '{1}'", column, typeof(TValue));
                return fn;
            }
        }

        /// <summary>
        /// The base class for the data view over items of user-defined type.
        /// </summary>
        /// <typeparam name="TRow">The user-defined data type.</typeparam>
        public abstract class DataViewBase<TRow> : IDataView
            where TRow : class
        {
            protected readonly IHost Host;

            private readonly DataViewSchema _schema;
            private readonly InternalSchemaDefinition _schemaDefn;

            // The array of generated methods that extract the fields of the current row object.
            private readonly Delegate[] _peeks;

            public abstract bool CanShuffle { get; }

            public DataViewSchema Schema => _schema;

            protected DataViewBase(IHostEnvironment env, string name, InternalSchemaDefinition schemaDefn)
            {
                Contracts.AssertValue(env);
                env.AssertNonWhiteSpace(name);
                Host = env.Register(name);
                Host.AssertValue(schemaDefn);

                _schemaDefn = schemaDefn;
                _schema = SchemaExtensions.MakeSchema(GetSchemaColumns(schemaDefn));
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

            public abstract long? GetRowCount();

            public abstract DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null);

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                return new[] { GetRowCursor(columnsNeeded, rand) };
            }

            public sealed class WrappedCursor : DataViewRowCursor
            {
                private readonly DataViewCursorBase _toWrap;

                public WrappedCursor(DataViewCursorBase toWrap) => _toWrap = toWrap;

                public override long Position => _toWrap.Position;
                public override long Batch => _toWrap.Batch;
                public override DataViewSchema Schema => _toWrap.Schema;

                protected override void Dispose(bool disposing)
                {
                    if (disposing)
                        _toWrap.Dispose();
                }

                /// <summary>
                /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
                /// This throws if the column is not active in this row, or if the type
                /// <typeparamref name="TValue"/> differs from this column's type.
                /// </summary>
                /// <typeparam name="TValue"> is the column's content type.</typeparam>
                /// <param name="column"> is the output column whose getter should be returned.</param>
                public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
                    => _toWrap.GetGetter<TValue>(column);

                public override ValueGetter<DataViewRowId> GetIdGetter() => _toWrap.GetIdGetter();

                /// <summary>
                /// Returns whether the given column is active in this row.
                /// </summary>
                public override bool IsColumnActive(DataViewSchema.Column column) => _toWrap.IsColumnActive(column);
                public override bool MoveNext() => _toWrap.MoveNext();
            }

            public abstract class DataViewCursorBase : InputRowBase<TRow>
            {
                // There is no real concept of multiple inheritance and for various reasons it was better to
                // descend from the row class as opposed to wrapping it, so much of this class is regrettably
                // copied from RootCursorBase.

                protected readonly DataViewBase<TRow> DataView;
                protected readonly IChannel Ch;
                private long _position;
                private bool _disposed;

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
                }

                /// <summary>
                /// Convenience property for checking whether the cursor is in a good state where values
                /// can be retrieved, that is, whenever <see cref="Position"/> is non-negative.
                /// </summary>
                protected bool IsGood => Position >= 0;

                protected sealed override void Dispose(bool disposing)
                {
                    if (_disposed)
                        return;
                    if (disposing)
                    {
                        Ch.Dispose();
                        _position = -1;
                    }
                    _disposed = true;
                    base.Dispose(disposing);
                }

                public bool MoveNext()
                {
                    if (_disposed)
                        return false;

                    if (MoveNextCore())
                    {
                        _position++;
                        return true;
                    }

                    Dispose();
                    return false;
                }

                /// <summary>
                /// Core implementation of <see cref="MoveNext"/>, called if no prior call to this method
                /// has returned <see langword="false"/>.
                /// </summary>
                protected abstract bool MoveNextCore();
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

            public override bool CanShuffle => true;

            public override long? GetRowCount()
            {
                return _data.Count;
            }

            public override DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, Schema);
                return new WrappedCursor(new Cursor(Host, "ListDataView", this, predicate, rand));
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
                    Func<int, bool> predicate, Random rand)
                    : base(env, dataView, predicate)
                {
                    Ch.AssertValueOrNull(rand);
                    _data = dataView._data;
                    if (rand != null)
                        _permutation = Utils.GetRandomPermutation(rand, dataView._data.Count);
                }

                public override ValueGetter<DataViewRowId> GetIdGetter()
                {
                    if (_permutation == null)
                    {
                        return
                            (ref DataViewRowId val) =>
                            {
                                Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                                val = new DataViewRowId((ulong)Position, 0);
                            };
                    }
                    else
                    {
                        return
                            (ref DataViewRowId val) =>
                            {
                                Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                                val = new DataViewRowId((ulong)Index, 0);
                            };
                    }
                }

                protected override TRow GetCurrentRowObject()
                {
                    Ch.Check(0 <= Position && Position < _data.Count, RowCursorUtils.FetchValueStateError);
                    return _data[Index];
                }

                protected override bool MoveNextCore()
                {
                    Ch.Assert(Position < _data.Count);
                    return Position + 1 < _data.Count;
                }
            }
        }

        /// <summary>
        /// An in-memory data view based on the IEnumerable of data.
        /// Doesn't support shuffling.
        /// </summary>
        internal sealed class StreamingDataView<TRow> : DataViewBase<TRow>
            where TRow : class
        {
            private IEnumerable<TRow> _data;

            public StreamingDataView(IHostEnvironment env, IEnumerable<TRow> data, InternalSchemaDefinition schemaDefn)
                : base(env, "StreamingDataView", schemaDefn)
            {
                Contracts.CheckValue(data, nameof(data));
                _data = data;
            }

            public override bool CanShuffle => false;

            public override long? GetRowCount()
                => (_data as ICollection<TRow>)?.Count;

            public override DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, Schema);
                return new WrappedCursor(new Cursor(Host, this, predicate));
            }

            private sealed class Cursor : DataViewCursorBase
            {
                private readonly IEnumerator<TRow> _enumerator;
                private TRow _currentRow;

                public Cursor(IHostEnvironment env, StreamingDataView<TRow> dataView, Func<int, bool> predicate)
                    : base(env, dataView, predicate)
                {
                    _enumerator = dataView._data.GetEnumerator();
                    _currentRow = null;
                }

                public override ValueGetter<DataViewRowId> GetIdGetter()
                {
                    return
                        (ref DataViewRowId val) =>
                        {
                            Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                            val = new DataViewRowId((ulong)Position, 0);
                        };
                }

                protected override TRow GetCurrentRowObject()
                {
                    return _currentRow;
                }

                protected override bool MoveNextCore()
                {
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

            public override bool CanShuffle => false;

            public override long? GetRowCount() => null;

            public void SetCurrentRowObject(TRow value)
            {
                Host.AssertValue(value);
                _current = value;
            }

            public override DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                Contracts.Assert(_current != null, "The current object must be set prior to cursoring");
                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, Schema);
                return new WrappedCursor(new Cursor(Host, this, predicate));
            }

            private sealed class Cursor : DataViewCursorBase
            {
                private readonly TRow _currentRow;

                public Cursor(IHostEnvironment env, SingleRowLoopDataView<TRow> dataView, Func<int, bool> predicate)
                    : base(env, dataView, predicate)
                {
                    _currentRow = dataView._current;
                }

                public override ValueGetter<DataViewRowId> GetIdGetter()
                {
                    return
                        (ref DataViewRowId val) =>
                        {
                            Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                            val = new DataViewRowId((ulong)Position, 0);
                        };
                }

                protected override TRow GetCurrentRowObject() => _currentRow;

                protected override bool MoveNextCore() => true;
            }
        }

        [BestFriend]
        internal static DataViewSchema.DetachedColumn[] GetSchemaColumns(InternalSchemaDefinition schemaDefn)
        {
            Contracts.AssertValue(schemaDefn);
            var columns = new DataViewSchema.DetachedColumn[schemaDefn.Columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                var col = schemaDefn.Columns[i];
                var meta = new DataViewSchema.Annotations.Builder();
                foreach (var kvp in col.Annotations)
                    meta.Add(kvp.Value.Kind, kvp.Value.AnnotationType, kvp.Value.GetGetterDelegate());
                columns[i] = new DataViewSchema.DetachedColumn(col.ColumnName, col.ColumnType, meta.ToAnnotations());
            }

            return columns;
        }
    }

    /// <summary>
    /// A single instance of annotation information, associated with a column.
    /// </summary>
    public abstract partial class AnnotationInfo
    {
        /// <summary>
        /// The type of the annotation.
        /// </summary>
        public DataViewType AnnotationType;
        /// <summary>
        /// The string identifier of the annotation. Some identifiers have special meaning,
        /// like "SlotNames", but any other identifiers can be used.
        /// </summary>
        public readonly string Kind;

        public abstract ValueGetter<TDst> GetGetter<TDst>();

        internal abstract Delegate GetGetterDelegate();

        private protected AnnotationInfo(string kind, DataViewType annotationType)
        {
            Contracts.AssertValueOrNull(annotationType);
            Contracts.AssertNonEmpty(kind);
            AnnotationType = annotationType;
            Kind = kind;
        }
    }

    /// <summary>
    /// Strongly-typed version of <see cref="AnnotationInfo"/>, that contains the actual value of the annotation.
    /// </summary>
    /// <typeparam name="T">Type of the annotation value.</typeparam>
    public sealed class AnnotationInfo<T> : AnnotationInfo
    {
        public readonly T Value;

        /// <summary>
        /// Constructor for annotation of value type T.
        /// </summary>
        /// <param name="kind">The string identifier of the annotation. Some identifiers have special meaning,
        /// like "SlotNames", but any other identifiers can be used.</param>
        /// <param name="value">Annotation value.</param>
        /// <param name="annotationType">Type of the annotation.</param>
        public AnnotationInfo(string kind, T value, DataViewType annotationType = null)
            : base(kind, annotationType)
        {
            Contracts.Assert(value != null);
            bool isVector;
            Type itemType;
            InternalSchemaDefinition.GetVectorAndItemType(typeof(T), "annotation value", out isVector, out itemType);

            if (annotationType == null)
            {
                // Infer a type as best we can.
                var primitiveItemType = ColumnTypeExtensions.PrimitiveTypeFromType(itemType);
                annotationType = isVector ? new VectorType(primitiveItemType) : (DataViewType)primitiveItemType;
            }
            else
            {
                // Make sure that the types are compatible with the declared type, including whether it is a vector type.
                VectorType annotationVectorType = annotationType as VectorType;
                bool annotationIsVector = annotationVectorType != null;
                if (isVector != annotationIsVector)
                {
                    throw Contracts.Except("Value inputted is supposed to be {0}, but type of Annotationinfo is {1}",
                        isVector ? "vector" : "scalar", annotationIsVector ? "vector" : "scalar");
                }

                DataViewType annotationItemType = annotationVectorType?.ItemType ?? annotationType;
                if (itemType != annotationItemType.RawType)
                {
                    throw Contracts.Except(
                        "Value inputted is supposed to have Type {0}, but type of Annotationinfo has {1}",
                        itemType.ToString(), annotationItemType.RawType.ToString());
                }
            }
            AnnotationType = annotationType;
            Value = value;
        }

        public override ValueGetter<TDst> GetGetter<TDst>()
        {
            var typeT = typeof(T);
            if (typeT.IsArray)
            {
                Contracts.Assert(AnnotationType is VectorType);
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
            if (AnnotationType is VectorType annotationVectorType)
            {
                // VBuffer<T> -> VBuffer<T>
                // REVIEW: Do we care about accomodating VBuffer<string> -> VBuffer<ReadOnlyMemory<char>>?

                Contracts.Assert(typeT.IsGenericType);
                Contracts.Check(typeof(TDst).IsGenericType);
                Contracts.Assert(typeT.GetGenericTypeDefinition() == typeof(VBuffer<>));
                Contracts.Check(typeof(TDst).GetGenericTypeDefinition() == typeof(VBuffer<>));
                var dstItemType = typeof(TDst).GetGenericArguments()[0];
                var itemType = typeT.GetGenericArguments()[0];
                Contracts.Assert(itemType == annotationVectorType.ItemType.RawType);
                Contracts.Check(itemType == dstItemType);

                Func<ValueGetter<VBuffer<int>>> srcMethod = GetVBufferGetter<int>;
                return srcMethod.GetMethodInfo().GetGenericMethodDefinition()
                    .MakeGenericMethod(annotationVectorType.ItemType.RawType)
                    .Invoke(this, new object[] { }) as ValueGetter<TDst>;
            }
            if (AnnotationType is PrimitiveDataViewType)
            {
                if (typeT == typeof(string))
                {
                    // String -> ReadOnlyMemory<char>
                    Contracts.Assert(AnnotationType is TextDataViewType);
                    ValueGetter<ReadOnlyMemory<char>> m = GetString;
                    return m as ValueGetter<TDst>;
                }
                // T -> T
                Contracts.Assert(AnnotationType.RawType == typeT);
                return GetDirectValue;
            }
            throw Contracts.ExceptNotImpl("Type '{0}' is not yet supported.", typeT.FullName);
        }

        // We want to use MarshalInvoke instead of adding custom Reflection logic for calling GetGetter<TDst>
        private Delegate GetGetterCore<TDst>()
        {
            return GetGetter<TDst>();
        }

        internal override Delegate GetGetterDelegate()
        {
            return Utils.MarshalInvoke(GetGetterCore<int>, AnnotationType.RawType);
        }

        private void GetStringArray(ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            var value = (string[])(object)Value;
            var n = Utils.Size(value);
            var dstEditor = VBufferEditor.Create(ref dst, n);

            for (int i = 0; i < n; i++)
                dstEditor.Values[i] = value[i].AsMemory();

            dst = dstEditor.Commit();
        }

        private ValueGetter<VBuffer<TDst>> GetArrayGetter<TDst>()
        {
            var value = (TDst[])(object)Value;
            var n = Utils.Size(value);
            return (ref VBuffer<TDst> dst) =>
            {
                var dstEditor = VBufferEditor.Create(ref dst, n);
                if (value != null)
                    value.AsSpan(0, n).CopyTo(dstEditor.Values);
                dst = dstEditor.Commit();
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