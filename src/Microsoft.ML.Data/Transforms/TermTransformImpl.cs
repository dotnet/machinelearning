// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.IO;
using System.Text;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Data
{
    // Implementations of the helper objects for term transform.

    public sealed partial class TermTransform : OneToOneTransformBase, ITransformTemplate
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
                RefPredicate<T> mapsToMissing;
                if (!Conversion.Conversions.Instance.TryGetIsNAPredicate(type, out mapsToMissing))
                    mapsToMissing = (ref T val) => false;
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
            public abstract void ParseAddTermArg(ref DvText terms, IChannel ch);

            /// <summary>
            /// Handling for the "term" arg.
            /// </summary>
            /// <param name="terms">The input terms argument</param>
            /// <param name="ch">The channel against which to report errors and warnings</param>
            public abstract void ParseAddTermArg(string[] terms, IChannel ch);

            private sealed class TextImpl : Builder<DvText>
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

                public override bool TryAdd(ref DvText val)
                {
                    if (!val.HasChars)
                        return false;
                    int count = _pool.Count;
                    return val.AddToPool(_pool).Id == count;
                }

                public override TermMap Finish()
                {
                    if (!_sorted || _pool.Count <= 1)
                        return new TermMap.TextImpl(_pool);
                    // REVIEW: Should write a Sort method in NormStr.Pool to make sorting more memory efficient.
                    var perm = Utils.GetIdentityPermutation(_pool.Count);
                    Comparison<int> comp = (i, j) => _pool.GetNormStrById(i).Value.CompareTo(_pool.GetNormStrById(j).Value);
                    Array.Sort(perm, comp);

                    var sortedPool = new NormStr.Pool();
                    for (int i = 0; i < perm.Length; ++i)
                    {
                        var nstr = sortedPool.Add(_pool.GetNormStrById(perm[i]).Value);
                        Contracts.Assert(nstr.Id == i);
                        Contracts.Assert(i == 0 || sortedPool.GetNormStrById(i - 1).Value.CompareTo(sortedPool.GetNormStrById(i).Value) < 0);
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
                private readonly RefPredicate<T> _mapsToMissing;
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
                public Impl(PrimitiveType type, RefPredicate<T> mapsToMissing, bool sort)
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
                    return !_mapsToMissing(ref val) && _values.TryAdd(val);
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
            public override void ParseAddTermArg(ref DvText terms, IChannel ch)
            {
                T val;
                var tryParse = Conversion.Conversions.Instance.GetParseConversion<T>(ItemType);
                for (bool more = true; more; )
                {
                    DvText term;
                    more = terms.SplitOne(',', out term, out terms);
                    term = term.Trim();
                    if (!term.HasChars)
                        ch.Warning("Empty strings ignored in 'terms' specification");
                    else if (!tryParse(ref term, out val))
                        ch.Warning("Item '{0}' ignored in 'terms' specification since it could not be parsed as '{1}'", term, ItemType);
                    else if (!TryAdd(ref val))
                        ch.Warning("Duplicate item '{0}' ignored in 'terms' specification", term);
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
                var tryParse = Conversion.Conversions.Instance.GetParseConversion<T>(ItemType);
                foreach (var sterm in terms)
                {
                    DvText term = new DvText(sterm);
                    term = term.Trim();
                    if (!term.HasChars)
                        ch.Warning("Empty strings ignored in 'term' specification");
                    else if (!tryParse(ref term, out val))
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
                new private readonly Builder<T> _bldr;

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
                new private readonly Builder<T> _bldr;
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
        private abstract class TermMap
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

            public abstract void Save(ModelSaveContext ctx, TermTransform trans);

            public static TermMap Load(ModelLoadContext ctx, IExceptionContext ectx, TermTransform trans)
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
                    if (!trans.CodecFactory.TryReadCodec(ctx.Reader.BaseStream, out codec))
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

            /// <summary>
            /// Given this instance, bind it to a particular input column. This allows us to service
            /// requests on the input dataset. This should throw an error if we attempt to bind this
            /// to the wrong type of item.
            /// </summary>
            public BoundTermMap Bind(TermTransform trans, int iinfo)
            {
                Contracts.AssertValue(trans);
                trans.Host.Assert(0 <= iinfo && iinfo < trans.Infos.Length);

                var info = trans.Infos[iinfo];
                var inType = info.TypeSrc.ItemType;
                if (!inType.Equals(ItemType))
                {
                    throw trans.Host.Except("Could not apply a map over type '{0}' to column '{1}' since it has type '{2}'",
                        ItemType, info.Name, inType);
                }
                return BoundTermMap.Create(this, trans, iinfo);
            }

            public abstract void WriteTextTerms(TextWriter writer);

            public sealed class TextImpl : TermMap<DvText>
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

                public override void Save(ModelSaveContext ctx, TermTransform trans)
                {
                    // *** Binary format ***
                    // byte: map type code, in this case 'Text' (0)
                    // int: number of terms
                    // int[]: term string ids

                    ctx.Writer.Write((byte)MapType.Text);
                    trans.Host.Assert(_pool.Count >= 0);
                    trans.Host.CheckDecode(_pool.Get("") == null);
                    ctx.Writer.Write(_pool.Count);

                    int id = 0;
                    foreach (var nstr in _pool)
                    {
                        trans.Host.Assert(nstr.Id == id);
                        ctx.SaveNonEmptyString(nstr.Value);
                        id++;
                    }
                }

                private void KeyMapper(ref DvText src, ref uint dst)
                {
                    var nstr = src.FindInPool(_pool);
                    if (nstr == null)
                        dst = 0;
                    else
                        dst = (uint)nstr.Id + 1;
                }

                public override ValueMapper<DvText, uint> GetKeyMapper()
                {
                    return KeyMapper;
                }

                public override void GetTerms(ref VBuffer<DvText> dst)
                {
                    DvText[] values = dst.Values;
                    if (Utils.Size(values) < _pool.Count)
                        values = new DvText[_pool.Count];
                    int slot = 0;
                    foreach (var nstr in _pool)
                    {
                        Contracts.Assert(0 <= nstr.Id & nstr.Id < values.Length);
                        Contracts.Assert(nstr.Id == slot);
                        values[nstr.Id] = new DvText(nstr.Value);
                        slot++;
                    }

                    dst = new VBuffer<DvText>(_pool.Count, values, dst.Indices);
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

                public override void Save(ModelSaveContext ctx, TermTransform trans)
                {
                    // *** Binary format ***
                    // byte: map type code, in this case 'Codec'
                    // codec parameterization: the codec
                    // int: number of terms
                    // value codec block: the terms written in the codec-defined binary format

                    IValueCodec codec;
                    if (!trans.CodecFactory.TryGetCodec(ItemType, out codec))
                        throw trans.Host.Except("We do not know how to serialize terms of type '{0}'", ItemType);
                    ctx.Writer.Write((byte)MapType.Codec);
                    trans.Host.Assert(codec.Type.Equals(ItemType));
                    trans.Host.Assert(codec.Type.IsPrimitive);
                    trans.CodecFactory.WriteCodec(ctx.Writer.BaseStream, codec);
                    IValueCodec<T> codecT = (IValueCodec<T>)codec;
                    ctx.Writer.Write(_values.Count);
                    using (var writer = codecT.OpenWriter(ctx.Writer.BaseStream))
                    {
                        for (int i = 0; i < _values.Count; ++i)
                        {
                            T val = _values.GetItem(i);
                            writer.Write(ref val);
                        }
                        writer.Commit();
                    }
                }

                public override ValueMapper<T, uint> GetKeyMapper()
                {
                    return
                        (ref T src, ref uint dst) =>
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
                    var stringMapper = Conversion.Conversions.Instance.GetStringConversion<T>(ItemType);
                    for (int i = 0; i < _values.Count; ++i)
                    {
                        T val = _values.GetItem(i);
                        stringMapper(ref val, ref sb);
                        writer.WriteLine("{0}\t{1}", i, sb.ToString());
                    }
                }
            }
        }

        private abstract class TermMap<T> : TermMap
        {
            protected TermMap(PrimitiveType type, int count)
                : base(type, count)
            {
                Contracts.Assert(ItemType.RawType == typeof(T));
            }

            public abstract ValueMapper<T, uint> GetKeyMapper();

            public abstract void GetTerms(ref VBuffer<T> dst);
        }

        private static void GetTextTerms<T>(ref VBuffer<T> src, ValueMapper<T, StringBuilder> stringMapper, ref VBuffer<DvText> dst)
        {
            // REVIEW: This convenience function is not optimized. For non-string
            // types, creating a whole bunch of string objects on the heap is one that is
            // fraught with risk. Ideally we'd have some sort of "copying" text buffer builder
            // but for now we'll see if this implementation suffices.

            // This utility function is not intended for use when we already have text!
            Contracts.Assert(typeof(T) != typeof(DvText));

            StringBuilder sb = null;
            DvText[] values = dst.Values;

            // We'd obviously have to adjust this a bit, if we ever had sparse metadata vectors.
            // The way the term map metadata getters are structured right now, this is impossible.
            Contracts.Assert(src.IsDense);

            if (Utils.Size(values) < src.Length)
                values = new DvText[src.Length];
            for (int i = 0; i < src.Length; ++i)
            {
                stringMapper(ref src.Values[i], ref sb);
                values[i] = new DvText(sb.ToString());
            }
            dst = new VBuffer<DvText>(src.Length, values, dst.Indices);
        }

        /// <summary>
        /// A mapper bound to a particular transform, and a particular column. These wrap
        /// a <see cref="TermMap"/>, and facilitate mapping that object to the inputs of
        /// a particular column, providing both values and metadata.
        /// </summary>
        private abstract class BoundTermMap
        {
            public readonly TermMap Map;

            private readonly TermTransform _parent;
            private readonly int _iinfo;
            private readonly bool _inputIsVector;

            private IHost Host { get { return _parent.Host; } }

            private bool IsTextMetadata { get { return _parent._textMetadata[_iinfo]; } }

            private BoundTermMap(TermMap map, TermTransform trans, int iinfo)
            {
                Contracts.AssertValue(trans);
                _parent = trans;

                Host.AssertValue(map);
                Host.Assert(0 <= iinfo && iinfo < trans.Infos.Length);
                ColInfo info = trans.Infos[iinfo];
                Host.Assert(info.TypeSrc.ItemType.Equals(map.ItemType));

                Map = map;
                _iinfo = iinfo;
                _inputIsVector = info.TypeSrc.IsVector;
            }

            public static BoundTermMap Create(TermMap map, TermTransform trans, int iinfo)
            {
                Contracts.AssertValue(trans);
                var host = trans.Host;

                host.AssertValue(map);
                host.Assert(0 <= iinfo && iinfo < trans.Infos.Length);
                ColInfo info = trans.Infos[iinfo];
                host.Assert(info.TypeSrc.ItemType.Equals(map.ItemType));

                return Utils.MarshalInvoke(CreateCore<int>, map.ItemType.RawType, map, trans, iinfo);
            }

            public static BoundTermMap CreateCore<T>(TermMap map, TermTransform trans, int iinfo)
            {
                TermMap<T> mapT = (TermMap<T>)map;
                if (mapT.ItemType.IsKey)
                    return new KeyImpl<T>(mapT, trans, iinfo);
                return new Impl<T>(mapT, trans, iinfo);
            }

            public abstract Delegate GetMappingGetter(IRow row);

            /// <summary>
            /// Allows us to optionally register metadata. It is also perfectly legal for
            /// this to do nothing, which corresponds to there being no metadata.
            /// </summary>
            public abstract void AddMetadata(MetadataDispatcher.Builder bldr);

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

                public Base(TermMap<T> map, TermTransform trans, int iinfo)
                    : base(map, trans, iinfo)
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
                    map(ref src, ref dst);
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
                        var info = _parent.Infos[_iinfo];
                        T src = default(T);
                        Contracts.Assert(!info.TypeSrc.IsVector);
                        ValueGetter<T> getSrc = _parent.GetSrcGetter<T>(input, _iinfo);
                        ValueGetter<uint> retVal =
                            (ref uint dst) =>
                            {
                                getSrc(ref src);
                                map(ref src, ref dst);
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
                        var info = _parent.Infos[_iinfo];
                        // First test whether default maps to default. If so this is sparsity preserving.
                        ValueGetter<VBuffer<T>> getSrc = _parent.GetSrcGetter<VBuffer<T>>(input, _iinfo);
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
                                        throw Host.Except("Column '{0}': TermTransform expects {1} slots, but got {2}", info.Name, cv, cval);
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
                                        map(ref values[islot], ref dstItem);
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
                                        throw Host.Except("Column '{0}': TermTransform expects {1} slots, but got {2}", info.Name, cv, cval);
                                    if (cval == 0)
                                    {
                                        // REVIEW: Should the VBufferBuilder be changed so that it can
                                        // build vectors of length zero?
                                        dst = new VBuffer<uint>(cval, dst.Values, dst.Indices);
                                        return;
                                    }

                                    // Despite default not mapping to default, it's very possible the result
                                    // might still be sparse, e.g., the source vector could be full of
                                    // unrecognized items.
                                    bldr.Reset(cval, dense: false);

                                    var values = src.Values;
                                    if (src.IsDense)
                                    {
                                        for (int slot = 0; slot < src.Length; ++slot)
                                        {
                                            map(ref values[slot], ref dstItem);
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
                                                Host.Assert(islot < src.Count);
                                                map(ref values[islot], ref dstItem);
                                                if (dstItem != 0)
                                                    bldr.AddFeature(slot, dstItem);
                                                nextExplicitSlot = ++islot == src.Count ? src.Length : indices[islot];
                                            }
                                            else
                                            {
                                                Host.Assert(slot < nextExplicitSlot);
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

                public override void AddMetadata(MetadataDispatcher.Builder bldr)
                {
                    if (TypedMap.Count == 0)
                        return;

                    if (IsTextMetadata && !TypedMap.ItemType.IsText)
                    {
                        var conv = Conversion.Conversions.Instance;
                        var stringMapper = conv.GetStringConversion<T>(TypedMap.ItemType);

                        MetadataUtils.MetadataGetter<VBuffer<DvText>> getter =
                            (int iinfo, ref VBuffer<DvText> dst) =>
                            {
                                Host.Assert(iinfo == _iinfo);
                                // No buffer sharing convenient here.
                                VBuffer<T> dstT = default(VBuffer<T>);
                                TypedMap.GetTerms(ref dstT);
                                GetTextTerms(ref dstT, stringMapper, ref dst);
                            };
                        bldr.AddGetter<VBuffer<DvText>>(MetadataUtils.Kinds.KeyValues,
                            new VectorType(TextType.Instance, TypedMap.OutputType.KeyCount), getter);
                    }
                    else
                    {
                        MetadataUtils.MetadataGetter<VBuffer<T>> getter =
                            (int iinfo, ref VBuffer<T> dst) =>
                            {
                                Host.Assert(iinfo == _iinfo);
                                TypedMap.GetTerms(ref dst);
                            };
                        bldr.AddGetter<VBuffer<T>>(MetadataUtils.Kinds.KeyValues,
                            new VectorType(TypedMap.ItemType, TypedMap.OutputType.KeyCount), getter);
                    }
                }
            }

            /// <summary>
            /// The key-typed version is the same as <see cref="BoundTermMap.Impl{T}"/>, except the metadata
            /// is based off a subset of the key values metadata.
            /// </summary>
            private sealed class KeyImpl<T> : Base<T>
            {
                public KeyImpl(TermMap<T> map, TermTransform trans, int iinfo)
                    : base(map, trans, iinfo)
                {
                    Host.Assert(TypedMap.ItemType.IsKey);
                }

                public override void AddMetadata(MetadataDispatcher.Builder bldr)
                {
                    if (TypedMap.Count == 0)
                        return;

                    int srcCol = _parent.Infos[_iinfo].Source;
                    ColumnType srcMetaType = _parent.Source.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, srcCol);
                    if (srcMetaType == null || srcMetaType.VectorSize != TypedMap.ItemType.KeyCount ||
                        TypedMap.ItemType.KeyCount == 0 || !Utils.MarshalInvoke(AddMetadataCore<int>, srcMetaType.ItemType.RawType, srcMetaType.ItemType, bldr))
                    {
                        // No valid input key-value metadata. Back off to the base implementation.
                        base.AddMetadata(bldr);
                    }
                }

                private bool AddMetadataCore<TMeta>(ColumnType srcMetaType, MetadataDispatcher.Builder bldr)
                {
                    Host.AssertValue(srcMetaType);
                    Host.Assert(srcMetaType.RawType == typeof(TMeta));
                    Host.AssertValue(bldr);
                    var srcType = TypedMap.ItemType.AsKey;
                    Host.AssertValue(srcType);
                    var dstType = new KeyType(DataKind.U4, srcType.Min, srcType.Count);
                    var convInst = Conversion.Conversions.Instance;
                    ValueMapper<T, uint> conv;
                    bool identity;
                    // If we can't convert this type to U4, don't try to pass along the metadata.
                    if (!convInst.TryGetStandardConversion<T, uint>(srcType, dstType, out conv, out identity))
                        return false;
                    int srcCol = _parent.Infos[_iinfo].Source;

                    ValueGetter<VBuffer<TMeta>> getter =
                        (ref VBuffer<TMeta> dst) =>
                        {
                            VBuffer<TMeta> srcMeta = default(VBuffer<TMeta>);
                            _parent.Source.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, srcCol, ref srcMeta);
                            Host.Assert(srcMeta.Length == srcType.Count);

                            VBuffer<T> keyVals = default(VBuffer<T>);
                            TypedMap.GetTerms(ref keyVals);
                            TMeta[] values = dst.Values;
                            if (Utils.Size(values) < TypedMap.OutputType.KeyCount)
                                values = new TMeta[TypedMap.OutputType.KeyCount];
                            uint convKeyVal = 0;
                            foreach (var pair in keyVals.Items(all: true))
                            {
                                T keyVal = pair.Value;
                                conv(ref keyVal, ref convKeyVal);
                                // The builder for the key values should not have any missings.
                                Host.Assert(0 < convKeyVal && convKeyVal <= srcMeta.Length);
                                srcMeta.GetItemOrDefault((int)(convKeyVal - 1), ref values[pair.Key]);
                            }
                            dst = new VBuffer<TMeta>(TypedMap.OutputType.KeyCount, values, dst.Indices);
                        };

                    if (IsTextMetadata && !srcMetaType.IsText)
                    {
                        var stringMapper = convInst.GetStringConversion<TMeta>(srcMetaType);
                        MetadataUtils.MetadataGetter<VBuffer<DvText>> mgetter =
                            (int iinfo, ref VBuffer<DvText> dst) =>
                            {
                                Host.Assert(iinfo == _iinfo);
                                var tempMeta = default(VBuffer<TMeta>);
                                getter(ref tempMeta);
                                Contracts.Assert(tempMeta.IsDense);
                                GetTextTerms(ref tempMeta, stringMapper, ref dst);
                                Host.Assert(dst.Length == TypedMap.OutputType.KeyCount);
                            };

                        bldr.AddGetter<VBuffer<DvText>>(MetadataUtils.Kinds.KeyValues,
                            new VectorType(TextType.Instance, TypedMap.OutputType.KeyCount), mgetter);
                    }
                    else
                    {
                        MetadataUtils.MetadataGetter<VBuffer<TMeta>> mgetter =
                            (int iinfo, ref VBuffer<TMeta> dst) =>
                            {
                                Host.Assert(iinfo == _iinfo);
                                getter(ref dst);
                                Host.Assert(dst.Length == TypedMap.OutputType.KeyCount);
                            };

                        bldr.AddGetter<VBuffer<TMeta>>(MetadataUtils.Kinds.KeyValues,
                            new VectorType(srcMetaType.ItemType.AsPrimitive, TypedMap.OutputType.KeyCount), mgetter);
                    }
                    return true;
                }

                public override void WriteTextTerms(TextWriter writer)
                {
                    if (TypedMap.Count == 0)
                        return;

                    int srcCol = _parent.Infos[_iinfo].Source;
                    ColumnType srcMetaType = _parent.Source.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, srcCol);
                    if (srcMetaType == null || srcMetaType.VectorSize != TypedMap.ItemType.KeyCount ||
                        TypedMap.ItemType.KeyCount == 0 || !Utils.MarshalInvoke(WriteTextTermsCore<int>, srcMetaType.ItemType.RawType, srcMetaType.AsVector.ItemType, writer))
                    {
                        // No valid input key-value metadata. Back off to the base implementation.
                        base.WriteTextTerms(writer);
                    }
                }

                private bool WriteTextTermsCore<TMeta>(PrimitiveType srcMetaType, TextWriter writer)
                {
                    Host.AssertValue(srcMetaType);
                    Host.Assert(srcMetaType.RawType == typeof(TMeta));
                    var srcType = TypedMap.ItemType.AsKey;
                    Host.AssertValue(srcType);
                    var dstType = new KeyType(DataKind.U4, srcType.Min, srcType.Count);
                    var convInst = Conversion.Conversions.Instance;
                    ValueMapper<T, uint> conv;
                    bool identity;
                    // If we can't convert this type to U4, don't try.
                    if (!convInst.TryGetStandardConversion<T, uint>(srcType, dstType, out conv, out identity))
                        return false;
                    int srcCol = _parent.Infos[_iinfo].Source;

                    VBuffer<TMeta> srcMeta = default(VBuffer<TMeta>);
                    _parent.Source.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, srcCol, ref srcMeta);
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
                        conv(ref keyVal, ref convKeyVal);
                        // The key mapping will not have admitted missing keys.
                        Host.Assert(0 < convKeyVal && convKeyVal <= srcMeta.Length);
                        srcMeta.GetItemOrDefault((int)(convKeyVal - 1), ref metaVal);
                        keyStringMapper(ref keyVal, ref sb);
                        writer.Write("{0}\t{1}", pair.Key, sb.ToString());
                        metaStringMapper(ref metaVal, ref sb);
                        writer.WriteLine("\t{0}", sb.ToString());
                    }
                    return true;
                }
            }

            private sealed class Impl<T> : Base<T>
            {
                public Impl(TermMap<T> map, TermTransform trans, int iinfo)
                    : base(map, trans, iinfo)
                {
                }
            }
        }
    }
}
