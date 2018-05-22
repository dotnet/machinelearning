// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

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
                ? InternalSchemaDefinition.Create(typeof(TRow))
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
                ? InternalSchemaDefinition.Create(typeof(TRow))
                : InternalSchemaDefinition.Create(typeof(TRow), schemaDefinition);
            return new StreamingDataView<TRow>(env, data, internalSchemaDefn);
        }

        /// <summary>
        /// The base class for the data view over items of user-defined type.
        /// </summary>
        /// <typeparam name="TRow">The user-defined data type.</typeparam>
        public abstract class DataViewBase<TRow> : IDataView
            where TRow : class
        {
            protected readonly IHost Host;

            private readonly SchemaProxy _schema;

            // The array of generated methods that extract the fields of the current row object.
            private readonly Delegate[] _peeks;

            public abstract bool CanShuffle { get; }

            public ISchema Schema
            {
                get { return _schema; }
            }

            protected DataViewBase(IHostEnvironment env, string name, InternalSchemaDefinition schemaDefn)
            {
                Contracts.AssertValue(env);
                env.AssertNonWhiteSpace(name);
                Host = env.Register(name);
                Host.AssertValue(schemaDefn);
                _schema = new SchemaProxy(schemaDefn);
                int n = _schema.SchemaDefn.Columns.Length;
                _peeks = new Delegate[n];
                for (var i = 0; i < n; i++)
                {
                    var currentColumn = _schema.SchemaDefn.Columns[i];
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

            public abstract class DataViewCursorBase : RootCursorBase, IRowCursor
            {
                protected readonly DataViewBase<TRow> DataView;
                private readonly int _colCount;
                private readonly Delegate[] _getters;

                protected DataViewCursorBase(IChannelProvider provider, DataViewBase<TRow> dataView,
                    Func<int, bool> predicate)
                    : base(provider)
                {
                    Ch.AssertValue(dataView);
                    Ch.AssertValue(predicate);

                    DataView = dataView;
                    _colCount = DataView._schema.SchemaDefn.Columns.Length;

                    _getters = new Delegate[_colCount];
                    for (int i = 0; i < _colCount; i++)
                        _getters[i] = predicate(i) ? CreateGetter(i) : null;
                }

                private Delegate CreateGetter(int index)
                {
                    var colType = DataView.Schema.GetColumnType(index);

                    var column = DataView._schema.SchemaDefn.Columns[index];
                    var outputType = column.IsComputed ? column.ReturnType : column.FieldInfo.FieldType;

                    Func<int, Delegate> del;

                    if (outputType.IsArray)
                    {
                        Ch.Assert(colType.IsVector);
                        // String[] -> VBuffer<DvText>
                        if (outputType.GetElementType() == typeof(string))
                        {
                            Ch.Assert(colType.ItemType.IsText);
                            return CreateStringArrayToVBufferGetter(index);
                        }
                        // T[] -> VBuffer<T>
                        Ch.Assert(outputType.GetElementType() == colType.ItemType.RawType);
                        del = CreateArrayToVBufferGetter<int>;
                    }
                    else if (colType.IsVector)
                    {
                        // VBuffer<T> -> VBuffer<T>
                        // REVIEW: Do we care about accomodating VBuffer<string> -> VBuffer<DvText>?
                        Ch.Assert(outputType.IsGenericType);
                        Ch.Assert(outputType.GetGenericTypeDefinition() == typeof(VBuffer<>));
                        Ch.Assert(outputType.GetGenericArguments()[0] == colType.ItemType.RawType);
                        del = CreateVBufferToVBufferDelegate<int>;
                    }
                    else if (colType.IsPrimitive)
                    {
                        if (outputType == typeof(string))
                        {
                            // String -> DvText
                            Ch.Assert(colType.IsText);
                            return CreateStringToTextGetter(index);
                        }
                        else if (outputType == typeof(bool))
                        {
                            // Bool -> DvBool
                            Ch.Assert(colType.IsBool);
                            return CreateBooleanToDvBoolGetter(index);
                        }
                        else if (outputType == typeof(bool?))
                        {
                            // Bool? -> DvBool
                            Ch.Assert(colType.IsBool);
                            return CreateNullableBooleanToDvBoolGetter(index);
                        }

                        // T -> T
                        Ch.Assert(colType.RawType == outputType);
                        del = CreateDirectGetter<int>;
                    }
                    else
                    {
                        // REVIEW: Is this even possible?
                        throw Ch.ExceptNotImpl("Type '{0}' is not yet supported.", outputType.FullName);
                    }
                    MethodInfo meth =
                        del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(colType.ItemType.RawType);
                    return (Delegate)meth.Invoke(this, new object[] { index });
                }

                private Delegate CreateStringArrayToVBufferGetter(int index)
                {
                    var peek = DataView._peeks[index] as Peek<TRow, string[]>;
                    Ch.AssertValue(peek);

                    string[] buf = null;

                    return (ValueGetter<VBuffer<DvText>>)((ref VBuffer<DvText> dst) =>
                    {
                        peek(GetCurrentRowObject(), Position, ref buf);
                        var n = Utils.Size(buf);
                        dst = new VBuffer<DvText>(n, Utils.Size(dst.Values) < n
                            ? new DvText[n]
                            : dst.Values, dst.Indices);
                        for (int i = 0; i < n; i++)
                            dst.Values[i] = new DvText(buf[i]);
                    });
                }

                private Delegate CreateStringToTextGetter(int index)
                {
                    var peek = DataView._peeks[index] as Peek<TRow, string>;
                    Ch.AssertValue(peek);
                    string buf = null;
                    return (ValueGetter<DvText>)((ref DvText dst) =>
                        {
                            peek(GetCurrentRowObject(), Position, ref buf);
                            dst = new DvText(buf);
                        });
                }

                private Delegate CreateBooleanToDvBoolGetter(int index)
                {
                    var peek = DataView._peeks[index] as Peek<TRow, bool>;
                    Ch.AssertValue(peek);
                    bool buf = false;
                    return (ValueGetter<DvBool>)((ref DvBool dst) =>
                    {
                        peek(GetCurrentRowObject(), Position, ref buf);
                        dst =  (DvBool)buf;
                    });
                }

                private Delegate CreateNullableBooleanToDvBoolGetter(int index)
                {
                    var peek = DataView._peeks[index] as Peek<TRow, bool?>;
                    Ch.AssertValue(peek);
                    bool? buf = null;
                    return (ValueGetter<DvBool>)((ref DvBool dst) =>
                    {
                        peek(GetCurrentRowObject(), Position, ref buf);
                        dst = buf.HasValue ? (DvBool)buf.Value : DvBool.NA;
                    });
                }

                private Delegate CreateArrayToVBufferGetter<TDst>(int index)
                {
                    var peek = DataView._peeks[index] as Peek<TRow, TDst[]>;
                    Ch.AssertValue(peek);
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

                private Delegate CreateVBufferToVBufferDelegate<TDst>(int index)
                {
                    var peek = DataView._peeks[index] as Peek<TRow, VBuffer<TDst>>;
                    Ch.AssertValue(peek);
                    VBuffer<TDst> buf = default(VBuffer<TDst>);
                    return (ValueGetter<VBuffer<TDst>>)((ref VBuffer<TDst> dst) =>
                        {
                            // The peek for a VBuffer is just a simple assignment, so there is
                            // no copy going on in the peek, so we must do that as a second
                            // step to the destination.
                            peek(GetCurrentRowObject(), Position, ref buf);
                            buf.CopyTo(ref dst);
                        });
                }

                private Delegate CreateDirectGetter<TDst>(int index)
                {
                    var peek = DataView._peeks[index] as Peek<TRow, TDst>;
                    Ch.AssertValue(peek);
                    return (ValueGetter<TDst>)((ref TDst dst) => { peek(GetCurrentRowObject(), Position, ref dst); });
                }

                protected abstract TRow GetCurrentRowObject();

                public override long Batch
                {
                    get { return 0; }
                }

                public ISchema Schema
                {
                    get { return DataView._schema; }
                }

                public bool IsColumnActive(int col)
                {
                    CheckColumnInRange(col);
                    return _getters[col] != null;
                }

                public ValueGetter<TValue> GetGetter<TValue>(int col)
                {
                    if (!IsColumnActive(col))
                        throw Ch.Except("Column {0} is not active in the cursor", col);
                    var getter = _getters[col];
                    Contracts.AssertValue(getter);
                    var fn = getter as ValueGetter<TValue>;
                    if (fn == null)
                        throw Ch.Except("Invalid TValue in GetGetter for column #{0}: '{1}'", col, typeof(TValue));
                    return fn;
                }

                private void CheckColumnInRange(int columnIndex)
                {
                    if (columnIndex < 0 || columnIndex >= _colCount)
                        throw Ch.Except("Column index must be between 0 and {0}", _colCount);
                }
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

                public Cursor(IChannelProvider provider, string name, ListDataView<TRow> dataView,
                    Func<int, bool> predicate, IRandom rand)
                    : base(provider, dataView, predicate)
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
                if (_data is ICollection<TRow> collection)
                    return collection.Count;
                return null;
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

                public Cursor(IChannelProvider provider, StreamingDataView<TRow> dataView, Func<int, bool> predicate)
                    : base(provider, dataView, predicate)
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

                public Cursor(IChannelProvider provider, SingleRowLoopDataView<TRow> dataView, Func<int, bool> predicate)
                    : base(provider, dataView, predicate)
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

        private sealed class SchemaProxy : ISchema
        {
            public readonly InternalSchemaDefinition SchemaDefn;

            public SchemaProxy(InternalSchemaDefinition schemaDefn)
            {
                SchemaDefn = schemaDefn;
            }

            public int ColumnCount
            {
                get { return SchemaDefn.Columns.Length; }
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                col = Array.FindIndex(SchemaDefn.Columns, c => c.ColumnName == name);
                return col >= 0;
            }

            public string GetColumnName(int col)
            {
                CheckColumnInRange(col);
                return SchemaDefn.Columns[col].ColumnName;
            }

            public ColumnType GetColumnType(int col)
            {
                CheckColumnInRange(col);
                return SchemaDefn.Columns[col].ColumnType;
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                CheckColumnInRange(col);
                var columnMetadata = SchemaDefn.Columns[col].Metadata;
                if (columnMetadata == null)
                    yield break;
                foreach (var kvp in columnMetadata.Select(x => new KeyValuePair<string, ColumnType>(x.Key, x.Value.MetadataType)))
                    yield return kvp;
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                if (string.IsNullOrEmpty(kind))
                    throw MetadataUtils.ExceptGetMetadata();
                CheckColumnInRange(col);
                var column = SchemaDefn.Columns[col];
                return column.Metadata.ContainsKey(kind) ? column.Metadata[kind].MetadataType : null;
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                var metadataType = GetMetadataTypeOrNull(kind, col);
                if (metadataType == null)
                    throw MetadataUtils.ExceptGetMetadata();

                var metadata = SchemaDefn.Columns[col].Metadata[kind];
                metadata.GetGetter<TValue>()(ref value);
            }

            private void CheckColumnInRange(int columnIndex)
            {
                if (columnIndex < 0 || columnIndex >= SchemaDefn.Columns.Length)
                    throw Contracts.Except("Column index must be between 0 and {0}", SchemaDefn.Columns.Length);
            }
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

                // String[] -> VBuffer<DvText>
                if (itemType == typeof(string))
                {
                    Contracts.Check(dstItemType == typeof(DvText));

                    ValueGetter<VBuffer<DvText>> method = GetStringArray;
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
                // REVIEW: Do we care about accomodating VBuffer<string> -> VBuffer<DvText>?

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
                    // String -> DvText
                    Contracts.Assert(MetadataType.IsText);
                    ValueGetter<DvText> m = GetString;
                    return m as ValueGetter<TDst>;
                }
                // T -> T
                Contracts.Assert(MetadataType.RawType == typeT);
                return GetDirectValue;
            }
            throw Contracts.ExceptNotImpl("Type '{0}' is not yet supported.", typeT.FullName);
        }

        public class TElement
        {
        }

        private void GetStringArray(ref VBuffer<DvText> dst)
        {
            var value = (string[])(object)Value;
            var n = Utils.Size(value);
            dst = new VBuffer<DvText>(n, Utils.Size(dst.Values) < n ? new DvText[n] : dst.Values, dst.Indices);

            for (int i = 0; i < n; i++)
                dst.Values[i] = new DvText(value[i]);

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

        private void GetString(ref DvText dst)
        {
            dst = new DvText((string)(object)Value);
        }

        private void GetDirectValue<TDst>(ref TDst dst)
        {
            dst = (TDst)(object)Value;
        }
    }
}