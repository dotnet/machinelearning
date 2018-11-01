// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using System;
using System.IO;
using System.Text;

namespace Microsoft.ML.Transforms.Categorical
{
    public sealed partial class TermTransform
    {
        /// <summary>
        /// These are objects shared by both the scalar and vector implementations of <see cref="Trainer"/>
        /// to accumulate individual scalar objects, and facilitate the creation of a <see cref="TermMap"/>.
        /// </summary>
        private abstract class Builder
        {
            /// <summary>
            /// The item type we are building into a term map.
            /// </summary>
            public readonly PrimitiveType ItemType;

            /// <summary>
            /// The number of items that would be in the map if created right now.
            /// </summary>
            public abstract int Count { get; }

            protected Builder(PrimitiveType type)
            {
                Contracts.AssertValue(type);
                ItemType = type;
            }

            public static Builder Create(ColumnType type, SortOrder sortOrder)
            {
                Contracts.AssertValue(type);
                Contracts.Assert(type.IsVector || type.IsPrimitive);
                // Right now we have only two. This "public" interface externally looks like it might
                // accept any value, but currently the internal implementations of Builder are split
                // along this being a purely binary option, for now (though this can easily change
                // with mot implementations of Builder).
                Contracts.Assert(sortOrder == SortOrder.Occurrence || sortOrder == SortOrder.Value);
                bool sorted = sortOrder == SortOrder.Value;

                PrimitiveType itemType = type.ItemType.AsPrimitive;
                Contracts.AssertValue(itemType);
                if (itemType.IsText)
                    return new TextImpl(sorted);
                return Utils.MarshalInvoke(CreateCore<int>, itemType.RawType, itemType, sorted);
            }

            private static Builder CreateCore<T>(PrimitiveType type, bool sorted)
                where T : IEquatable<T>, IComparable<T>
            {
                Contracts.AssertValue(type);
                Contracts.Assert(type.RawType == typeof(T));

                // If this is a type with NA values, we should ignore those NA values for the purpose
                // of building our term dictionary. For the other types (practically, only the UX types),
                // we should ignore nothing.
                InPredicate<T> mapsToMissing;
                if (!Runtime.Data.Conversion.Conversions.Instance.TryGetIsNAPredicate(type, out mapsToMissing))
                    mapsToMissing = (in T val) => false;
                return new Impl<T>(type, mapsToMissing, sorted);
            }

            /// <summary>
            /// Called at the end of training, to get the final mapper object.
            /// </summary>
            public abstract TermMap Finish();

            /// <summary>
            /// Handling for the "terms" arg.
            /// </summary>
            /// <param name="terms">The input terms argument</param>
            /// <param name="ch">The channel against which to report errors and warnings</param>
            public abstract void ParseAddTermArg(ref ReadOnlyMemory<char> terms, IChannel ch);

            /// <summary>
            /// Handling for the "term" arg.
            /// </summary>
            /// <param name="terms">The input terms argument</param>
            /// <param name="ch">The channel against which to report errors and warnings</param>
            public abstract void ParseAddTermArg(string[] terms, IChannel ch);

            private sealed class TextImpl : Builder<ReadOnlyMemory<char>>
            {
                private readonly NormStr.Pool _pool;
                private readonly bool _sorted;

                public override int Count
                {
                    get { return _pool.Count; }
                }

                public TextImpl(bool sorted)
                    : base(TextType.Instance)
                {
                    _pool = new NormStr.Pool();
                    _sorted = sorted;
                }

                public override bool TryAdd(ref ReadOnlyMemory<char> val)
                {
                    if (val.IsEmpty)
                        return false;
                    int count = _pool.Count;
                    return ReadOnlyMemoryUtils.AddToPool(val, _pool).Id == count;
                }

                public override TermMap Finish()
                {
                    if (!_sorted || _pool.Count <= 1)
                        return new TermMap.TextImpl(_pool);
                    // REVIEW: Should write a Sort method in NormStr.Pool to make sorting more memory efficient.
                    var perm = Utils.GetIdentityPermutation(_pool.Count);
                    Comparison<int> comp = (i, j) => _pool.GetNormStrById(i).Value.Span.CompareTo(_pool.GetNormStrById(j).Value.Span, StringComparison.Ordinal);
                    Array.Sort(perm, comp);

                    var sortedPool = new NormStr.Pool();
                    for (int i = 0; i < perm.Length; ++i)
                    {
                        var nstr = sortedPool.Add(_pool.GetNormStrById(perm[i]).Value);
                        Contracts.Assert(nstr.Id == i);
                        Contracts.Assert(i == 0 || sortedPool.GetNormStrById(i - 1).Value.Span.CompareTo(sortedPool.GetNormStrById(i).Value.Span, StringComparison.Ordinal) < 0);
                    }
                    Contracts.Assert(sortedPool.Count == _pool.Count);
                    return new TermMap.TextImpl(sortedPool);
                }
            }

            /// <summary>
            /// The sorted builder outputs things so that the keys are in sorted order.
            /// </summary>
            private sealed class Impl<T> : Builder<T>
                where T : IEquatable<T>, IComparable<T>
            {
                // Because we can't know the actual mapping till we finish.
                private readonly HashArray<T> _values;
                private readonly InPredicate<T> _mapsToMissing;
                private readonly bool _sort;

                public override int Count
                {
                    get { return _values.Count; }
                }

                /// <summary>
                /// Instantiates.
                /// </summary>
                /// <param name="type">The type we are mapping</param>
                /// <param name="mapsToMissing">This indicates whether a given value will map
                /// to the missing value. If this returns true for a value then we do not attempt
                /// to store it in the map.</param>
                /// <param name="sort">Indicates whether to sort mapping IDs by input values.</param>
                public Impl(PrimitiveType type, InPredicate<T> mapsToMissing, bool sort)
                    : base(type)
                {
                    Contracts.Assert(type.RawType == typeof(T));
                    Contracts.AssertValue(mapsToMissing);

                    _values = new HashArray<T>();
                    _mapsToMissing = mapsToMissing;
                    _sort = sort;
                }

                public override bool TryAdd(ref T val)
                {
                    return !_mapsToMissing(in val) && _values.TryAdd(val);
                }

                public override TermMap Finish()
                {
                    if (_sort)
                        _values.Sort();
                    return new TermMap.HashArrayImpl<T>(ItemType, _values);
                }
            }
        }

        private abstract class Builder<T> : Builder
        {
            protected Builder(PrimitiveType type)
                : base(type)
            {
            }

            /// <summary>
            /// Ensures that the item is in the set. Returns true iff it added the item.
            /// </summary>
            /// <param name="val">The value to consider</param>
            public abstract bool TryAdd(ref T val);

            /// <summary>
            /// Handling for the "terms" arg.
            /// </summary>
            /// <param name="terms">The input terms argument</param>
            /// <param name="ch">The channel against which to report errors and warnings</param>
            public override void ParseAddTermArg(ref ReadOnlyMemory<char> terms, IChannel ch)
            {
                T val;
                var tryParse = Runtime.Data.Conversion.Conversions.Instance.GetParseConversion<T>(ItemType);
                for (bool more = true; more;)
                {
                    ReadOnlyMemory<char> term;
                    more = ReadOnlyMemoryUtils.SplitOne(terms, ',', out term, out terms);
                    term = ReadOnlyMemoryUtils.TrimSpaces(term);
                    if (term.IsEmpty)
                        ch.Warning("Empty strings ignored in 'terms' specification");
                    else if (!tryParse(in term, out val))
                        throw ch.Except($"Item '{term}' in 'terms' specification could not be parsed as '{ItemType}'");
                    else if (!TryAdd(ref val))
                        ch.Warning($"Duplicate item '{term}' ignored in 'terms' specification", term);
                }

                if (Count == 0)
                    throw ch.ExceptUserArg(nameof(Arguments.Terms), "Nothing parsed as '{0}'", ItemType);
            }

            /// <summary>
            /// Handling for the "term" arg.
            /// </summary>
            /// <param name="terms">The input terms argument</param>
            /// <param name="ch">The channel against which to report errors and warnings</param>
            public override void ParseAddTermArg(string[] terms, IChannel ch)
            {
                T val;
                var tryParse = Runtime.Data.Conversion.Conversions.Instance.GetParseConversion<T>(ItemType);
                foreach (var sterm in terms)
                {
                    ReadOnlyMemory<char> term = sterm.AsMemory();
                    term = ReadOnlyMemoryUtils.TrimSpaces(term);
                    if (term.IsEmpty)
                        ch.Warning("Empty strings ignored in 'term' specification");
                    else if (!tryParse(in term, out val))
                        ch.Warning("Item '{0}' ignored in 'term' specification since it could not be parsed as '{1}'", term, ItemType);
                    else if (!TryAdd(ref val))
                        ch.Warning("Duplicate item '{0}' ignored in 'term' specification", term);
                }

                if (Count == 0)
                    throw ch.ExceptUserArg(nameof(Arguments.Terms), "Nothing parsed as '{0}'", ItemType);
            }
        }

        /// <summary>
        /// The trainer is an object that given an <see cref="Builder"/> instance, maps a particular
        /// input, whether it be scalar or vector, into this and allows us to continue training on it.
        /// </summary>
        private abstract class Trainer
        {
            private readonly Builder _bldr;
            private int _remaining;

            public int Count { get { return _bldr.Count; } }

            private Trainer(Builder bldr, int max)
            {
                Contracts.AssertValue(bldr);
                Contracts.Assert(max >= 0);
                _bldr = bldr;
                _remaining = max;
            }

            /// <summary>
            /// Creates an instance of <see cref="Trainer"/> appropriate for the type at a given
            /// row and column.
            /// </summary>
            /// <param name="row">The row to fetch from</param>
            /// <param name="col">The column to get the getter from</param>
            /// <param name="count">The maximum count of items to map</param>
            /// <param name="autoConvert">Whether we attempt to automatically convert
            /// the input type to the desired type</param>
            /// <param name="bldr">The builder we add items to</param>
            /// <returns>An associated training pipe</returns>
            public static Trainer Create(IRow row, int col, bool autoConvert, int count, Builder bldr)
            {
                Contracts.AssertValue(row);
                var schema = row.Schema;
                Contracts.Assert(0 <= col && col < schema.ColumnCount);
                Contracts.Assert(count > 0);
                Contracts.AssertValue(bldr);

                var type = schema.GetColumnType(col);
                Contracts.Assert(autoConvert || bldr.ItemType == type.ItemType);
                // Auto conversion should only be possible when the type is text.
                Contracts.Assert(type.IsText || !autoConvert);
                if (type.IsVector)
                    return Utils.MarshalInvoke(CreateVec<int>, bldr.ItemType.RawType, row, col, count, bldr);
                return Utils.MarshalInvoke(CreateOne<int>, bldr.ItemType.RawType, row, col, autoConvert, count, bldr);
            }

            private static Trainer CreateOne<T>(IRow row, int col, bool autoConvert, int count, Builder bldr)
            {
                Contracts.AssertValue(row);
                Contracts.AssertValue(bldr);
                Contracts.Assert(bldr is Builder<T>);
                var bldrT = (Builder<T>)bldr;

                ValueGetter<T> inputGetter;
                if (autoConvert)
                    inputGetter = RowCursorUtils.GetGetterAs<T>(bldr.ItemType, row, col);
                else
                    inputGetter = row.GetGetter<T>(col);

                return new ImplOne<T>(inputGetter, count, bldrT);
            }

            private static Trainer CreateVec<T>(IRow row, int col, int count, Builder bldr)
            {
                Contracts.AssertValue(row);
                Contracts.AssertValue(bldr);
                Contracts.Assert(bldr is Builder<T>);
                var bldrT = (Builder<T>)bldr;

                var inputGetter = row.GetGetter<VBuffer<T>>(col);
                return new ImplVec<T>(inputGetter, count, bldrT);
            }

            /// <summary>
            /// Indicates to the <see cref="Trainer"/> that we have reached a new row and should consider
            /// what to do with these values. Returns false if we have determined that it is no longer necessary
            /// to call this train, because we've already accumulated the maximum number of values.
            /// </summary>
            public abstract bool ProcessRow();

            /// <summary>
            /// Returns a <see cref="TermMap"/> over the items in this column. Note that even if this
            /// was trained over a vector valued column, the particular implementation returned here
            /// should be a mapper over the item type.
            /// </summary>
            public TermMap Finish()
            {
                return _bldr.Finish();
            }

            private sealed class ImplOne<T> : Trainer
            {
                private readonly ValueGetter<T> _getter;
                private T _val;
                private new readonly Builder<T> _bldr;

                public ImplOne(ValueGetter<T> getter, int max, Builder<T> bldr)
                    : base(bldr, max)
                {
                    Contracts.AssertValue(getter);
                    Contracts.AssertValue(bldr);
                    _getter = getter;
                    _bldr = bldr;
                }

                public sealed override bool ProcessRow()
                {
                    Contracts.Assert(_remaining >= 0);
                    if (_remaining <= 0)
                        return false;
                    _getter(ref _val);
                    return !_bldr.TryAdd(ref _val) || --_remaining > 0;
                }
            }

            private sealed class ImplVec<T> : Trainer
            {
                private readonly ValueGetter<VBuffer<T>> _getter;
                private VBuffer<T> _val;
                private new readonly Builder<T> _bldr;
                private bool _addedDefaultFromSparse;

                public ImplVec(ValueGetter<VBuffer<T>> getter, int max, Builder<T> bldr)
                    : base(bldr, max)
                {
                    Contracts.AssertValue(getter);
                    Contracts.AssertValue(bldr);
                    _getter = getter;
                    _bldr = bldr;
                }

                private bool AccumAndDecrement(ref T val)
                {
                    Contracts.Assert(_remaining > 0);
                    return !_bldr.TryAdd(ref val) || --_remaining > 0;
                }

                public sealed override bool ProcessRow()
                {
                    Contracts.Assert(_remaining >= 0);
                    if (_remaining <= 0)
                        return false;
                    _getter(ref _val);
                    if (_val.IsDense || _addedDefaultFromSparse)
                    {
                        for (int i = 0; i < _val.Count; ++i)
                        {
                            if (!AccumAndDecrement(ref _val.Values[i]))
                                return false;
                        }
                        return true;
                    }
                    // The vector is sparse, and we have not yet tried adding the implicit default value.
                    // Because sparse vectors are supposed to be functionally the same as dense vectors
                    // and also because the order in which we see items matters for the final mapping, we
                    // must first add the first explicit entries where indices[i]==i, then add the default
                    // immediately following that, then continue with the remainder of the items. Note
                    // that the builder is taking care of the case where default maps to missing. Also
                    // note that this code is called on at most one row per column, so we aren't terribly
                    // excited about the slight inefficiency of that first if check.
                    Contracts.Assert(!_val.IsDense && !_addedDefaultFromSparse);
                    T def = default(T);
                    for (int i = 0; i < _val.Count; ++i)
                    {
                        if (!_addedDefaultFromSparse && _val.Indices[i] != i)
                        {
                            _addedDefaultFromSparse = true;
                            if (!AccumAndDecrement(ref def))
                                return false;
                        }
                        if (!AccumAndDecrement(ref _val.Values[i]))
                            return false;
                    }
                    if (!_addedDefaultFromSparse)
                    {
                        _addedDefaultFromSparse = true;
                        if (!AccumAndDecrement(ref def))
                            return false;
                    }
                    return true;
                }
            }
        }

        private enum MapType : byte
        {
            Text = 0,
            Codec = 1,
        }

        /// <summary>
        /// Given this instance, bind it to a particular input column. This allows us to service
        /// requests on the input dataset. This should throw an error if we attempt to bind this
        /// to the wrong type of item.
        /// </summary>
        private static BoundTermMap Bind(IHostEnvironment env, Schema schema, TermMap unbound, ColInfo[] infos, bool[] textMetadata, int iinfo)
        {
            env.Assert(0 <= iinfo && iinfo < infos.Length);

            var info = infos[iinfo];
            var inType = info.TypeSrc.ItemType;
            if (!inType.Equals(unbound.ItemType))
            {
                throw env.Except("Could not apply a map over type '{0}' to column '{1}' since it has type '{2}'",
                    unbound.ItemType, info.Name, inType);
            }
            return BoundTermMap.Create(env, schema, unbound, infos, textMetadata, iinfo);
        }

        /// <summary>
        /// A map is an object capable of creating the association from an input type, to an output
        /// type. The input type, whatever it is, must have <see cref="ItemType"/> as its input item
        /// type, and will produce either <see cref="OutputType"/>, or a vector type with that output
        /// type if the input was a vector.
        ///
        /// Note that instances of this class can be shared among multiple <see cref="TermTransform"/>
        /// instances. To associate this with a particular transform, use the <see cref="Bind"/> method.
        ///
        /// These are the immutable and serializable analogs to the <see cref="Builder"/> used in
        /// training.
        /// </summary>
        public abstract class TermMap
        {
            /// <summary>
            /// The item type of the input type, that is, either the input type or,
            /// if a vector, the item type of that type.
            /// </summary>
            public readonly PrimitiveType ItemType;

            /// <summary>
            /// The output item type. This will always be of known cardinality. Its count is always
            /// equal to <see cref="Count"/>, unless <see cref="Count"/> is 0 in which case this has
            /// key count of 1, since a count of 0 would indicate an unbound key. If we ever improve
            /// key types so they are capable of distinguishing between the set they index being
            /// empty vs. of unknown or unbound cardinality, this should change.
            /// </summary>
            public readonly KeyType OutputType;

            /// <summary>
            /// The number of items in the map.
            /// </summary>
            public readonly int Count;

            protected TermMap(PrimitiveType type, int count)
            {
                Contracts.AssertValue(type);
                Contracts.Assert(count >= 0);
                ItemType = type;
                Count = count;
                OutputType = new KeyType(DataKind.U4, 0, Count == 0 ? 1 : Count);
            }

            internal abstract void Save(ModelSaveContext ctx, IHostEnvironment host, CodecFactory codecFactory);

            internal static TermMap Load(ModelLoadContext ctx, IHostEnvironment ectx, CodecFactory codecFactory)
            {
                // *** Binary format ***
                // byte: map type code
                // <remainer> ...

                MapType mtype = (MapType)ctx.Reader.ReadByte();
                ectx.CheckDecode(Enum.IsDefined(typeof(MapType), mtype));
                switch (mtype)
                {
                    case MapType.Text:
                        // Binary format defined by this method.
                        return TextImpl.Create(ctx, ectx);
                    case MapType.Codec:
                        // *** Binary format ***
                        // codec parameterization: the codec
                        // int: number of terms
                        // value codec block: the terms written in the codec-defined binary format
                        IValueCodec codec;
                        if (!codecFactory.TryReadCodec(ctx.Reader.BaseStream, out codec))
                            throw ectx.Except("Unrecognized codec read");
                        ectx.CheckDecode(codec.Type.IsPrimitive);
                        int count = ctx.Reader.ReadInt32();
                        ectx.CheckDecode(count >= 0);
                        return Utils.MarshalInvoke(LoadCodecCore<int>, codec.Type.RawType, ctx, ectx, codec, count);
                    default:
                        ectx.Assert(false);
                        throw ectx.Except("Unrecognized type '{0}'", mtype);
                }
            }

            private static TermMap LoadCodecCore<T>(ModelLoadContext ctx, IExceptionContext ectx, IValueCodec codec, int count)
                where T : IEquatable<T>, IComparable<T>
            {
                Contracts.AssertValue(ectx);
                ectx.AssertValue(ctx);
                ectx.AssertValue(codec);
                ectx.Assert(codec is IValueCodec<T>);
                ectx.Assert(codec.Type.IsPrimitive);
                ectx.Assert(count >= 0);

                IValueCodec<T> codecT = (IValueCodec<T>)codec;

                var values = new HashArray<T>();
                if (count > 0)
                {
                    using (var reader = codecT.OpenReader(ctx.Reader.BaseStream, count))
                    {
                        T item = default(T);
                        for (int i = 0; i < count; i++)
                        {
                            reader.MoveNext();
                            reader.Get(ref item);
                            int index = values.Add(item);
                            ectx.Assert(0 <= index && index <= i);
                            if (index != i)
                                throw ectx.Except("Duplicate items at positions {0} and {1}", index, i);
                        }
                    }
                }

                return new HashArrayImpl<T>(codec.Type.AsPrimitive, values);
            }

            public abstract void WriteTextTerms(TextWriter writer);

            public sealed class TextImpl : TermMap<ReadOnlyMemory<char>>
            {
                private readonly NormStr.Pool _pool;

                /// <summary>
                /// A pool based text mapping implementation.
                /// </summary>
                /// <param name="pool">The string pool</param>
                public TextImpl(NormStr.Pool pool)
                    : base(TextType.Instance, pool.Count)
                {
                    Contracts.AssertValue(pool);
                    _pool = pool;
                }

                public static TextImpl Create(ModelLoadContext ctx, IExceptionContext ectx)
                {
                    // *** Binary format ***
                    // int: number of terms
                    // int[]: term string ids

                    // Note that this binary format as read here diverges from the save format
                    // insofar that the save format contains the "I am text" code, which by the
                    // time we reach here, we have already read.

                    var pool = new NormStr.Pool();
                    int cstr = ctx.Reader.ReadInt32();
                    ectx.CheckDecode(cstr >= 0);

                    for (int istr = 0; istr < cstr; istr++)
                    {
                        var nstr = pool.Add(ctx.LoadNonEmptyString());
                        ectx.CheckDecode(nstr.Id == istr);
                    }

                    // The way we "train" the termMap, they shouldn't contain the empty string.
                    ectx.CheckDecode(pool.Get("") == null);

                    return new TextImpl(pool);
                }

                internal override void Save(ModelSaveContext ctx, IHostEnvironment host, CodecFactory codecFactory)
                {
                    // *** Binary format ***
                    // byte: map type code, in this case 'Text' (0)
                    // int: number of terms
                    // int[]: term string ids

                    ctx.Writer.Write((byte)MapType.Text);
                    host.Assert(_pool.Count >= 0);
                    host.CheckDecode(_pool.Get("") == null);
                    ctx.Writer.Write(_pool.Count);

                    int id = 0;
                    foreach (var nstr in _pool)
                    {
                        host.Assert(nstr.Id == id);
                        ctx.SaveNonEmptyString(nstr.Value);
                        id++;
                    }
                }

                private void KeyMapper(in ReadOnlyMemory<char> src, ref uint dst)
                {
                    var nstr = ReadOnlyMemoryUtils.FindInPool(src, _pool);
                    if (nstr == null)
                        dst = 0;
                    else
                        dst = (uint)nstr.Id + 1;
                }

                public override ValueMapper<ReadOnlyMemory<char>, uint> GetKeyMapper()
                {
                    return KeyMapper;
                }

                public override void GetTerms(ref VBuffer<ReadOnlyMemory<char>> dst)
                {
                    ReadOnlyMemory<char>[] values = dst.Values;
                    if (Utils.Size(values) < _pool.Count)
                        values = new ReadOnlyMemory<char>[_pool.Count];
                    int slot = 0;
                    foreach (var nstr in _pool)
                    {
                        Contracts.Assert(0 <= nstr.Id & nstr.Id < values.Length);
                        Contracts.Assert(nstr.Id == slot);
                        values[nstr.Id] = nstr.Value;
                        slot++;
                    }

                    dst = new VBuffer<ReadOnlyMemory<char>>(_pool.Count, values, dst.Indices);
                }

                public override void WriteTextTerms(TextWriter writer)
                {
                    writer.WriteLine("# Number of terms = {0}", Count);
                    foreach (var nstr in _pool)
                        writer.WriteLine("{0}\t{1}", nstr.Id, nstr.Value);
                }
            }

            public sealed class HashArrayImpl<T> : TermMap<T>
                where T : IEquatable<T>, IComparable<T>
            {
                // One of the two must exist. If we need one we can initialize it
                // from the other.
                private readonly HashArray<T> _values;

                public HashArrayImpl(PrimitiveType itemType, HashArray<T> values)
                    // Note: The caller shouldn't pass a null HashArray.
                    : base(itemType, values.Count)
                {
                    Contracts.AssertValue(values);
                    _values = values;
                }

                internal override void Save(ModelSaveContext ctx, IHostEnvironment host, CodecFactory codecFactory)
                {
                    // *** Binary format ***
                    // byte: map type code, in this case 'Codec'
                    // codec parameterization: the codec
                    // int: number of terms
                    // value codec block: the terms written in the codec-defined binary format

                    IValueCodec codec;
                    if (!codecFactory.TryGetCodec(ItemType, out codec))
                        throw host.Except("We do not know how to serialize terms of type '{0}'", ItemType);
                    ctx.Writer.Write((byte)MapType.Codec);
                    host.Assert(codec.Type.Equals(ItemType));
                    host.Assert(codec.Type.IsPrimitive);
                    codecFactory.WriteCodec(ctx.Writer.BaseStream, codec);
                    IValueCodec<T> codecT = (IValueCodec<T>)codec;
                    ctx.Writer.Write(_values.Count);
                    using (var writer = codecT.OpenWriter(ctx.Writer.BaseStream))
                    {
                        for (int i = 0; i < _values.Count; ++i)
                        {
                            T val = _values.GetItem(i);
                            writer.Write(in val);
                        }
                        writer.Commit();
                    }
                }

                public override ValueMapper<T, uint> GetKeyMapper()
                {
                    return
                        (in T src, ref uint dst) =>
                        {
                            int val;
                            if (_values.TryGetIndex(src, out val))
                                dst = (uint)val + 1;
                            else
                                dst = 0;
                        };
                }

                public override void GetTerms(ref VBuffer<T> dst)
                {
                    if (Count == 0)
                    {
                        dst = new VBuffer<T>(0, dst.Values, dst.Indices);
                        return;
                    }
                    T[] values = dst.Values;
                    if (Utils.Size(values) < Count)
                        values = new T[Count];
                    Contracts.AssertValue(_values);
                    Contracts.Assert(_values.Count == Count);
                    _values.CopyTo(values);
                    dst = new VBuffer<T>(Count, values, dst.Indices);
                }

                public override void WriteTextTerms(TextWriter writer)
                {
                    writer.WriteLine("# Number of terms of type '{0}' = {1}", ItemType, Count);
                    StringBuilder sb = null;
                    var stringMapper = Runtime.Data.Conversion.Conversions.Instance.GetStringConversion<T>(ItemType);
                    for (int i = 0; i < _values.Count; ++i)
                    {
                        T val = _values.GetItem(i);
                        stringMapper(in val, ref sb);
                        writer.WriteLine("{0}\t{1}", i, sb.ToString());
                    }
                }
            }
        }

        public abstract class TermMap<T> : TermMap
        {
            protected TermMap(PrimitiveType type, int count)
                : base(type, count)
            {
                Contracts.Assert(ItemType.RawType == typeof(T));
            }

            public abstract ValueMapper<T, uint> GetKeyMapper();

            public abstract void GetTerms(ref VBuffer<T> dst);
        }

        private static void GetTextTerms<T>(in VBuffer<T> src, ValueMapper<T, StringBuilder> stringMapper, ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            // REVIEW: This convenience function is not optimized. For non-string
            // types, creating a whole bunch of string objects on the heap is one that is
            // fraught with risk. Ideally we'd have some sort of "copying" text buffer builder
            // but for now we'll see if this implementation suffices.

            // This utility function is not intended for use when we already have text!
            Contracts.Assert(typeof(T) != typeof(ReadOnlyMemory<char>));

            StringBuilder sb = null;
            ReadOnlyMemory<char>[] values = dst.Values;

            // We'd obviously have to adjust this a bit, if we ever had sparse metadata vectors.
            // The way the term map metadata getters are structured right now, this is impossible.
            Contracts.Assert(src.IsDense);

            if (Utils.Size(values) < src.Length)
                values = new ReadOnlyMemory<char>[src.Length];
            for (int i = 0; i < src.Length; ++i)
            {
                stringMapper(in src.Values[i], ref sb);
                values[i] = sb.ToString().AsMemory();
            }
            dst = new VBuffer<ReadOnlyMemory<char>>(src.Length, values, dst.Indices);
        }

        /// <summary>
        /// A mapper bound to a particular transform, and a particular column. These wrap
        /// a <see cref="TermMap"/>, and facilitate mapping that object to the inputs of
        /// a particular column, providing both values and metadata.
        /// </summary>
        private abstract class BoundTermMap
        {
            public readonly TermMap Map;

            private readonly int _iinfo;
            private readonly bool _inputIsVector;
            private readonly IHostEnvironment _host;
            private readonly bool[] _textMetadata;
            private readonly ColInfo[] _infos;
            private readonly Schema _schema;

            private bool IsTextMetadata { get { return _textMetadata[_iinfo]; } }

            private BoundTermMap(IHostEnvironment env, Schema schema, TermMap map, ColInfo[] infos, bool[] textMetadata, int iinfo)
            {
                _host = env;
                //assert me.
                _textMetadata = textMetadata;
                _infos = infos;
                _schema = schema;
                _host.AssertValue(map);
                _host.Assert(0 <= iinfo && iinfo < infos.Length);
                var info = infos[iinfo];
                _host.Assert(info.TypeSrc.ItemType.Equals(map.ItemType));

                Map = map;
                _iinfo = iinfo;
                _inputIsVector = info.TypeSrc.IsVector;
            }

            public static BoundTermMap Create(IHostEnvironment host, Schema schema, TermMap map, ColInfo[] infos, bool[] textMetadata, int iinfo)
            {
                host.AssertValue(map);
                host.Assert(0 <= iinfo && iinfo < infos.Length);
                var info = infos[iinfo];
                host.Assert(info.TypeSrc.ItemType.Equals(map.ItemType));

                return Utils.MarshalInvoke(CreateCore<int>, map.ItemType.RawType, host, schema, map, infos, textMetadata, iinfo);
            }

            public static BoundTermMap CreateCore<T>(IHostEnvironment env, Schema schema, TermMap map, ColInfo[] infos, bool[] textMetadata, int iinfo)
            {
                TermMap<T> mapT = (TermMap<T>)map;
                if (mapT.ItemType.IsKey)
                    return new KeyImpl<T>(env, schema, mapT, infos, textMetadata, iinfo);
                return new Impl<T>(env, schema, mapT, infos, textMetadata, iinfo);
            }

            public abstract Delegate GetMappingGetter(IRow row);

            /// <summary>
            /// Allows us to optionally register metadata. It is also perfectly legal for
            /// this to do nothing, which corresponds to there being no metadata.
            /// </summary>
            public abstract void AddMetadata(Schema.Metadata.Builder builder);

            /// <summary>
            /// Writes out all terms we map to a text writer, with one line per mapped term.
            /// The line should have the format mapped key value, then a tab, then the term
            /// that is mapped. The writer should not be closed, as it will be used to write
            /// all term maps. We should write <see cref="TermMap.Count"/> terms.
            /// </summary>
            /// <param name="writer">The writer to which we write terms</param>
            public virtual void WriteTextTerms(TextWriter writer)
            {
                Map.WriteTextTerms(writer);
            }

            private abstract class Base<T> : BoundTermMap
            {
                protected readonly TermMap<T> TypedMap;

                public Base(IHostEnvironment env, Schema schema, TermMap<T> map, ColInfo[] infos, bool[] textMetadata, int iinfo)
                    : base(env, schema, map, infos, textMetadata, iinfo)
                {
                    TypedMap = map;
                }

                /// <summary>
                /// Returns what the default value maps to.
                /// </summary>
                private static uint MapDefault(ValueMapper<T, uint> map)
                {
                    T src = default(T);
                    uint dst = 0;
                    map(in src, ref dst);
                    return dst;
                }

                public override Delegate GetMappingGetter(IRow input)
                {
                    // When constructing the getter, there are a few cases we have to consider:
                    // If scalar then it's just a straightforward mapping.
                    // If vector, then we have to detect whether the mapping happens to be
                    // sparsity preserving or not, that is, if the default value maps to the
                    // default (missing) key. For some types this will always be true, but it
                    // could also be true if we happened to never see the default value in
                    // training.

                    if (!_inputIsVector)
                    {
                        ValueMapper<T, uint> map = TypedMap.GetKeyMapper();
                        var info = _infos[_iinfo];
                        T src = default(T);
                        Contracts.Assert(!info.TypeSrc.IsVector);
                        input.Schema.TryGetColumnIndex(info.Source, out int colIndex);
                        _host.Assert(input.IsColumnActive(colIndex));
                        var getSrc = input.GetGetter<T>(colIndex);
                        ValueGetter<uint> retVal =
                            (ref uint dst) =>
                            {
                                getSrc(ref src);
                                map(in src, ref dst);
                            };
                        return retVal;
                    }
                    else
                    {
                        // It might be tempting to move "map" and "info" out of both blocks and into
                        // the main block of the function, but please don't do that. The implicit
                        // classes created by the compiler to hold the non-vector and vector lambdas
                        // will have an indirect wrapping class to hold "map" and "info". This is
                        // bad, especially since "map" is very frequently called.
                        ValueMapper<T, uint> map = TypedMap.GetKeyMapper();
                        var info = _infos[_iinfo];
                        // First test whether default maps to default. If so this is sparsity preserving.
                        input.Schema.TryGetColumnIndex(info.Source, out int colIndex);
                        _host.Assert(input.IsColumnActive(colIndex));
                        var getSrc = input.GetGetter<VBuffer<T>>(colIndex);
                        VBuffer<T> src = default(VBuffer<T>);
                        ValueGetter<VBuffer<uint>> retVal;
                        // REVIEW: Consider whether possible or reasonable to not use a builder here.
                        var bldr = new BufferBuilder<uint>(U4Adder.Instance);
                        int cv = info.TypeSrc.VectorSize;
                        uint defaultMapValue = MapDefault(map);
                        uint dstItem = default(uint);

                        if (defaultMapValue == 0)
                        {
                            // Sparsity preserving.
                            retVal =
                                (ref VBuffer<uint> dst) =>
                                {
                                    getSrc(ref src);
                                    int cval = src.Length;
                                    if (cv != 0 && cval != cv)
                                        throw _host.Except("Column '{0}': TermTransform expects {1} slots, but got {2}", info.Name, cv, cval);
                                    if (cval == 0)
                                    {
                                        // REVIEW: Should the VBufferBuilder be changed so that it can
                                        // build vectors of length zero?
                                        dst = new VBuffer<uint>(cval, dst.Values, dst.Indices);
                                        return;
                                    }

                                    bldr.Reset(cval, dense: false);

                                    var values = src.Values;
                                    var indices = !src.IsDense ? src.Indices : null;
                                    int count = src.Count;
                                    for (int islot = 0; islot < count; islot++)
                                    {
                                        map(in values[islot], ref dstItem);
                                        if (dstItem != 0)
                                        {
                                            int slot = indices != null ? indices[islot] : islot;
                                            bldr.AddFeature(slot, dstItem);
                                        }
                                    }

                                    bldr.GetResult(ref dst);
                                };
                        }
                        else
                        {
                            retVal =
                                (ref VBuffer<uint> dst) =>
                                {
                                    getSrc(ref src);
                                    int cval = src.Length;
                                    if (cv != 0 && cval != cv)
                                        throw _host.Except("Column '{0}': TermTransform expects {1} slots, but got {2}", info.Name, cv, cval);
                                    if (cval == 0)
                                    {
                                        // REVIEW: Should the VBufferBuilder be changed so that it can
                                        // build vectors of length zero?
                                        dst = new VBuffer<uint>(cval, dst.Values, dst.Indices);
                                        return;
                                    }

                                    // Despite default not mapping to default, it's very possible the result
                                    // might still be sparse, for example, the source vector could be full of
                                    // unrecognized items.
                                    bldr.Reset(cval, dense: false);

                                    var values = src.Values;
                                    if (src.IsDense)
                                    {
                                        for (int slot = 0; slot < src.Length; ++slot)
                                        {
                                            map(in values[slot], ref dstItem);
                                            if (dstItem != 0)
                                                bldr.AddFeature(slot, dstItem);
                                        }
                                    }
                                    else
                                    {
                                        var indices = src.Indices;
                                        int nextExplicitSlot = src.Count == 0 ? src.Length : indices[0];
                                        int islot = 0;
                                        for (int slot = 0; slot < src.Length; ++slot)
                                        {
                                            if (nextExplicitSlot == slot)
                                            {
                                                // This was an explicitly defined value.
                                                _host.Assert(islot < src.Count);
                                                map(in values[islot], ref dstItem);
                                                if (dstItem != 0)
                                                    bldr.AddFeature(slot, dstItem);
                                                nextExplicitSlot = ++islot == src.Count ? src.Length : indices[islot];
                                            }
                                            else
                                            {
                                                _host.Assert(slot < nextExplicitSlot);
                                                // This is a non-defined implicit default value. No need to attempt a remap
                                                // since we already have it.
                                                bldr.AddFeature(slot, defaultMapValue);
                                            }
                                        }
                                    }
                                    bldr.GetResult(ref dst);
                                };
                        }
                        return retVal;
                    }
                }

                public override void AddMetadata(Schema.Metadata.Builder builder)
                {
                    if (TypedMap.Count == 0)
                        return;
                    if (IsTextMetadata && !TypedMap.ItemType.IsText)
                    {
                        var conv = Runtime.Data.Conversion.Conversions.Instance;
                        var stringMapper = conv.GetStringConversion<T>(TypedMap.ItemType);

                        ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter =
                            (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                            {
                                // No buffer sharing convenient here.
                                VBuffer<T> dstT = default;
                                TypedMap.GetTerms(ref dstT);
                                GetTextTerms(in dstT, stringMapper, ref dst);
                            };
                        builder.AddKeyValues(TypedMap.OutputType.KeyCount, TextType.Instance, getter);
                    }
                    else
                    {
                        ValueGetter<VBuffer<T>> getter =
                            (ref VBuffer<T> dst) =>
                            {
                                TypedMap.GetTerms(ref dst);
                            };
                        builder.AddKeyValues(TypedMap.OutputType.KeyCount, TypedMap.ItemType, getter);
                    }
                }
            }

            /// <summary>
            /// The key-typed version is the same as <see cref="BoundTermMap.Impl{T}"/>, except the metadata
            /// is based off a subset of the key values metadata.
            /// </summary>
            private sealed class KeyImpl<T> : Base<T>
            {
                public KeyImpl(IHostEnvironment env, Schema schema, TermMap<T> map, ColInfo[] infos, bool[] textMetadata, int iinfo)
                    : base(env, schema, map, infos, textMetadata, iinfo)
                {
                    _host.Assert(TypedMap.ItemType.IsKey);
                }

                public override void AddMetadata(Schema.Metadata.Builder builder)
                {
                    if (TypedMap.Count == 0)
                        return;

                    _schema.TryGetColumnIndex(_infos[_iinfo].Source, out int srcCol);
                    ColumnType srcMetaType = _schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, srcCol);
                    if (srcMetaType == null || srcMetaType.VectorSize != TypedMap.ItemType.KeyCount ||
                        TypedMap.ItemType.KeyCount == 0 || !Utils.MarshalInvoke(AddMetadataCore<int>, srcMetaType.ItemType.RawType, srcMetaType.ItemType, builder))
                    {
                        // No valid input key-value metadata. Back off to the base implementation.
                        base.AddMetadata(builder);
                    }
                }

                private bool AddMetadataCore<TMeta>(ColumnType srcMetaType, Schema.Metadata.Builder builder)
                {
                    _host.AssertValue(srcMetaType);
                    _host.Assert(srcMetaType.RawType == typeof(TMeta));
                    _host.AssertValue(builder);
                    var srcType = TypedMap.ItemType.AsKey;
                    _host.AssertValue(srcType);
                    var dstType = new KeyType(DataKind.U4, srcType.Min, srcType.Count);
                    var convInst = Runtime.Data.Conversion.Conversions.Instance;
                    ValueMapper<T, uint> conv;
                    bool identity;
                    // If we can't convert this type to U4, don't try to pass along the metadata.
                    if (!convInst.TryGetStandardConversion<T, uint>(srcType, dstType, out conv, out identity))
                        return false;
                    _schema.TryGetColumnIndex(_infos[_iinfo].Source, out int srcCol);

                    ValueGetter<VBuffer<TMeta>> getter =
                        (ref VBuffer<TMeta> dst) =>
                        {
                            VBuffer<TMeta> srcMeta = default(VBuffer<TMeta>);
                            _schema.GetMetadata(MetadataUtils.Kinds.KeyValues, srcCol, ref srcMeta);
                            _host.Assert(srcMeta.Length == srcType.Count);

                            VBuffer<T> keyVals = default(VBuffer<T>);
                            TypedMap.GetTerms(ref keyVals);
                            TMeta[] values = dst.Values;
                            if (Utils.Size(values) < TypedMap.OutputType.KeyCount)
                                values = new TMeta[TypedMap.OutputType.KeyCount];
                            uint convKeyVal = 0;
                            foreach (var pair in keyVals.Items(all: true))
                            {
                                T keyVal = pair.Value;
                                conv(in keyVal, ref convKeyVal);
                                // The builder for the key values should not have any missings.
                                _host.Assert(0 < convKeyVal && convKeyVal <= srcMeta.Length);
                                srcMeta.GetItemOrDefault((int)(convKeyVal - 1), ref values[pair.Key]);
                            }
                            dst = new VBuffer<TMeta>(TypedMap.OutputType.KeyCount, values, dst.Indices);
                        };

                    if (IsTextMetadata && !srcMetaType.IsText)
                    {
                        var stringMapper = convInst.GetStringConversion<TMeta>(srcMetaType);
                        ValueGetter<VBuffer<ReadOnlyMemory<char>>> mgetter =
                            (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                            {
                                var tempMeta = default(VBuffer<TMeta>);
                                getter(ref tempMeta);
                                Contracts.Assert(tempMeta.IsDense);
                                GetTextTerms(in tempMeta, stringMapper, ref dst);
                                _host.Assert(dst.Length == TypedMap.OutputType.KeyCount);
                            };
                        builder.AddKeyValues(TypedMap.OutputType.KeyCount, TextType.Instance, mgetter);
                    }
                    else
                    {
                        ValueGetter<VBuffer<TMeta>> mgetter =
                            (ref VBuffer<TMeta> dst) =>
                            {
                                getter(ref dst);
                                _host.Assert(dst.Length == TypedMap.OutputType.KeyCount);
                            };
                        builder.AddKeyValues(TypedMap.OutputType.KeyCount, srcMetaType.ItemType.AsPrimitive, mgetter);
                    }
                    return true;
                }

                public override void WriteTextTerms(TextWriter writer)
                {
                    if (TypedMap.Count == 0)
                        return;

                    _schema.TryGetColumnIndex(_infos[_iinfo].Source, out int srcCol);
                    ColumnType srcMetaType = _schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, srcCol);
                    if (srcMetaType == null || srcMetaType.VectorSize != TypedMap.ItemType.KeyCount ||
                        TypedMap.ItemType.KeyCount == 0 || !Utils.MarshalInvoke(WriteTextTermsCore<int>, srcMetaType.ItemType.RawType, srcMetaType.AsVector.ItemType, writer))
                    {
                        // No valid input key-value metadata. Back off to the base implementation.
                        base.WriteTextTerms(writer);
                    }
                }

                private bool WriteTextTermsCore<TMeta>(PrimitiveType srcMetaType, TextWriter writer)
                {
                    _host.AssertValue(srcMetaType);
                    _host.Assert(srcMetaType.RawType == typeof(TMeta));
                    var srcType = TypedMap.ItemType.AsKey;
                    _host.AssertValue(srcType);
                    var dstType = new KeyType(DataKind.U4, srcType.Min, srcType.Count);
                    var convInst = Runtime.Data.Conversion.Conversions.Instance;
                    ValueMapper<T, uint> conv;
                    bool identity;
                    // If we can't convert this type to U4, don't try.
                    if (!convInst.TryGetStandardConversion<T, uint>(srcType, dstType, out conv, out identity))
                        return false;
                    _schema.TryGetColumnIndex(_infos[_iinfo].Source, out int srcCol);

                    VBuffer<TMeta> srcMeta = default(VBuffer<TMeta>);
                    _schema.GetMetadata(MetadataUtils.Kinds.KeyValues, srcCol, ref srcMeta);
                    if (srcMeta.Length != srcType.Count)
                        return false;

                    VBuffer<T> keyVals = default(VBuffer<T>);
                    TypedMap.GetTerms(ref keyVals);
                    TMeta metaVal = default(TMeta);
                    uint convKeyVal = 0;
                    StringBuilder sb = null;
                    var keyStringMapper = convInst.GetStringConversion<T>(TypedMap.ItemType);
                    var metaStringMapper = convInst.GetStringConversion<TMeta>(srcMetaType);

                    writer.WriteLine("# Number of terms of key '{0}' indexing '{1}' value = {2}",
                        TypedMap.ItemType, srcMetaType, TypedMap.Count);
                    foreach (var pair in keyVals.Items(all: true))
                    {
                        T keyVal = pair.Value;
                        conv(in keyVal, ref convKeyVal);
                        // The key mapping will not have admitted missing keys.
                        _host.Assert(0 < convKeyVal && convKeyVal <= srcMeta.Length);
                        srcMeta.GetItemOrDefault((int)(convKeyVal - 1), ref metaVal);
                        keyStringMapper(in keyVal, ref sb);
                        writer.Write("{0}\t{1}", pair.Key, sb.ToString());
                        metaStringMapper(in metaVal, ref sb);
                        writer.WriteLine("\t{0}", sb.ToString());
                    }
                    return true;
                }
            }

            private sealed class Impl<T> : Base<T>
            {
                public Impl(IHostEnvironment env, Schema schema, TermMap<T> map, ColInfo[] infos, bool[] textMetadata, int iinfo)
                    : base(env, schema, map, infos, textMetadata, iinfo)
                {
                }
            }
        }
    }
}
