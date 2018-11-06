// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Immutable;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This is the abstract base class for all types in the <see cref="IDataView"/> type system.
    /// </summary>
    public abstract class ColumnType : IEquatable<ColumnType>
    {
        // This private constructor sets all the _isXxx flags. It is invoked by other ctors.
        private ColumnType()
        {
            IsPrimitive = this is PrimitiveType;
            IsVector = this is VectorType;
            IsNumber = this is NumberType;
            IsKey = this is KeyType;
        }

        /// <summary>
        /// Constructor for extension types, which must be either <see cref="PrimitiveType"/> or <see cref="StructuredType"/>.
        /// </summary>
        private protected ColumnType(Type rawType)
            : this()
        {
            Contracts.CheckValue(rawType, nameof(rawType));
            RawType = rawType;
            RawType.TryGetDataKind(out var rawKind);
            RawKind = rawKind;
        }

        /// <summary>
        /// Internal sub types can pass both the <paramref name="rawType"/> and <paramref name="rawKind"/> values.
        /// This asserts that they are consistent.
        /// </summary>
        private protected ColumnType(Type rawType, DataKind rawKind)
            : this()
        {
            Contracts.AssertValue(rawType);
#if DEBUG
            DataKind tmp;
            rawType.TryGetDataKind(out tmp);
            Contracts.Assert(tmp == rawKind);
#endif
            RawType = rawType;
            RawKind = rawKind;
        }

        /// <summary>
        /// The raw <see cref="Type"/> for this <see cref="ColumnType"/>. Note that this is the raw representation type
        /// and not the complete information content of the <see cref="ColumnType"/>. Code should not assume that
        /// a <see cref="RawType"/> uniquely identifiers a <see cref="ColumnType"/>. For example, most practical instances of
        /// <see cref="KeyType"/> and <see cref="NumberType.U4"/> will have a <see cref="RawType"/> of <see cref="uint"/>,
        /// but both are very different in the types of information conveyed in that number.
        /// </summary>
        public Type RawType { get; }

        /// <summary>
        /// The DataKind corresponding to RawType, if there is one (zero otherwise). It is equivalent
        /// to the result produced by DataKindExtensions.TryGetDataKind(RawType, out kind).
        /// </summary>
        [BestFriend]
        internal DataKind RawKind { get; }

        /// <summary>
        /// Whether this is a primitive type. External code should use <c>is <see cref="PrimitiveType"/></c>.
        /// </summary>
        [BestFriend]
        internal bool IsPrimitive { get; }

        /// <summary>
        /// Equivalent to "this as PrimitiveType".
        /// </summary>
        [BestFriend]
        internal PrimitiveType AsPrimitive => IsPrimitive ? (PrimitiveType)this : null;

        /// <summary>
        /// Whether this type is a standard numeric type.
        /// </summary>
        [BestFriend]
        internal bool IsNumber { get; }

        /// <summary>
        /// Whether this type is the standard text type.
        /// </summary>
        [BestFriend]
        internal bool IsText
        {
            get
            {
                if (!(this is TextType))
                    return false;
                // TextType is a singleton.
                Contracts.Assert(this == TextType.Instance);
                return true;
            }
        }

        /// <summary>
        /// Whether this type is the standard boolean type.
        /// </summary>
        [BestFriend]
        internal bool IsBool
        {
            get
            {
                if (!(this is BoolType))
                    return false;
                // BoolType is a singleton.
                Contracts.Assert(this == BoolType.Instance);
                return true;
            }
        }

        /// <summary>
        /// Whether this type is a standard scalar type completely determined by its <see cref="RawType"/>
        /// (not a <see cref="KeyType"/> or <see cref="StructuredType"/>, etc).
        /// </summary>
        [BestFriend]
        internal bool IsStandardScalar => IsNumber || IsText || IsBool ||
            (this is TimeSpanType) || (this is DateTimeType) || (this is DateTimeOffsetType);

        /// <summary>
        /// Whether this type is a key type, which implies that the order of values is not significant,
        /// and arithmetic is non-sensical. A key type can define a cardinality.
        /// </summary>
        [BestFriend]
        internal bool IsKey { get; }

        /// <summary>
        /// Equivalent to "this as KeyType".
        /// </summary>
        [BestFriend]
        internal KeyType AsKey => IsKey ? (KeyType)this : null;

        /// <summary>
        /// Zero return means either it's not a key type or the cardinality is unknown. External code should first
        /// test whether this is of type <see cref="KeyType"/>, then if so get the <see cref="KeyType.Count"/> property
        /// from that.
        /// </summary>
        [BestFriend]
        internal int KeyCount => KeyCountCore;

        /// <summary>
        /// The only sub-class that should override this is <see cref="KeyType"/>.
        /// </summary>
        private protected virtual int KeyCountCore => 0;

        /// <summary>
        /// Whether this is a vector type. External code should just check directly against whether this type
        /// is <see cref="VectorType"/>.
        /// </summary>
        [BestFriend]
        internal bool IsVector { get; }

        /// <summary>
        /// Equivalent to "this as VectorType".
        /// </summary>
        [BestFriend]
        internal VectorType AsVector => IsVector ? (VectorType)this : null;

        /// <summary>
        /// For non-vector types, this returns the column type itself (i.e., return this).
        /// </summary>
        [BestFriend]
        internal ColumnType ItemType => ItemTypeCore;

        /// <summary>
        /// Whether this is a vector type with known size. Returns false for non-vector types.
        /// Equivalent to VectorSize > 0.
        /// </summary>
        [BestFriend]
        internal bool IsKnownSizeVector => VectorSize > 0;

        /// <summary>
        /// Zero return means either it's not a vector or the size is unknown. Equivalent to
        /// IsVector ? ValueCount : 0 and to IsKnownSizeVector ? ValueCount : 0.
        /// </summary>
        [BestFriend]
        internal int VectorSize => VectorSizeCore;

        /// <summary>
        /// For non-vectors, this returns one. For unknown size vectors, it returns zero.
        /// Equivalent to IsVector ? VectorSize : 1.
        /// </summary>
        [BestFriend]
        internal int ValueCount => ValueCountCore;

        /// <summary>
        /// The only sub-class that should override this is VectorType!
        /// </summary>
        private protected virtual ColumnType ItemTypeCore => this;

        /// <summary>
        /// The only sub-class that should override this is <see cref="VectorType"/>!
        /// </summary>
        private protected virtual int VectorSizeCore => 0;

        /// <summary>
        /// The only sub-class that should override this is VectorType!
        /// </summary>
        private protected virtual int ValueCountCore => 1;

        // IEquatable<T> interface recommends also to override base class implementations of
        // Object.Equals(Object) and GetHashCode. In classes below where Equals(ColumnType other)
        // is effectively a referencial comparison, there is no need to override base class implementations
        // of Object.Equals(Object) (and GetHashCode) since its also a referencial comparison.
        public abstract bool Equals(ColumnType other);

        /// <summary>
        /// Equivalent to calling Equals(ColumnType) for non-vector types. For vector type,
        /// returns true if current and other vector types have the same size and item type.
        /// </summary>
        [BestFriend]
        internal bool SameSizeAndItemType(ColumnType other)
        {
            if (other == null)
                return false;

            if (Equals(other))
                return true;

            // For vector types, we don't care about the factoring of the dimensions.
            if (!IsVector || !other.IsVector)
                return false;
            if (!ItemType.Equals(other.ItemType))
                return false;
            return VectorSize == other.VectorSize;
        }
    }

    /// <summary>
    /// The abstract base class for all non-primitive types.
    /// </summary>
    public abstract class StructuredType : ColumnType
    {
        protected StructuredType(Type rawType)
            : base(rawType)
        {
            Contracts.Assert(!IsPrimitive);
        }

        private protected StructuredType(Type rawType, DataKind rawKind)
            : base(rawType, rawKind)
        {
            Contracts.Assert(!IsPrimitive);
        }
    }

    /// <summary>
    /// The abstract base class for all primitive types. Values of these types can be freely copied
    /// without concern for ownership, mutation, or disposing.
    /// </summary>
    public abstract class PrimitiveType : ColumnType
    {
        protected PrimitiveType(Type rawType)
            : base(rawType)
        {
            Contracts.Assert(IsPrimitive);
            Contracts.CheckParam(!typeof(IDisposable).IsAssignableFrom(RawType), nameof(rawType),
                "A " + nameof(PrimitiveType) + " cannot have a disposable " + nameof(RawType));
        }

        private protected PrimitiveType(Type rawType, DataKind rawKind)
            : base(rawType, rawKind)
        {
            Contracts.Assert(IsPrimitive);
            Contracts.Assert(!typeof(IDisposable).IsAssignableFrom(RawType));
        }

        [BestFriend]
        internal static PrimitiveType FromKind(DataKind kind)
        {
            if (kind == DataKind.TX)
                return TextType.Instance;
            if (kind == DataKind.BL)
                return BoolType.Instance;
            if (kind == DataKind.TS)
                return TimeSpanType.Instance;
            if (kind == DataKind.DT)
                return DateTimeType.Instance;
            if (kind == DataKind.DZ)
                return DateTimeOffsetType.Instance;
            return NumberType.FromKind(kind);
        }
    }

    /// <summary>
    /// The standard text type.
    /// </summary>
    public sealed class TextType : PrimitiveType
    {
        private static volatile TextType _instance;
        public static TextType Instance
        {
            get
            {
                if (_instance == null)
                    Interlocked.CompareExchange(ref _instance, new TextType(), null);
                return _instance;
            }
        }

        private TextType()
            : base(typeof(ReadOnlyMemory<char>), DataKind.TX)
        {
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Contracts.Assert(!(other is TextType));
            return false;
        }

        public override string ToString() => "Text";
    }

    /// <summary>
    /// The standard number types.
    /// </summary>
    public sealed class NumberType : PrimitiveType
    {
        private readonly string _name;

        private NumberType(DataKind kind, string name)
            : base(kind.ToType(), kind)
        {
            Contracts.AssertNonEmpty(name);
            _name = name;
            Contracts.Assert(IsNumber);
        }

        private static volatile NumberType _instI1;
        public static NumberType I1
        {
            get
            {
                if (_instI1 == null)
                    Interlocked.CompareExchange(ref _instI1, new NumberType(DataKind.I1, "I1"), null);
                return _instI1;
            }
        }

        private static volatile NumberType _instU1;
        public static NumberType U1
        {
            get
            {
                if (_instU1 == null)
                    Interlocked.CompareExchange(ref _instU1, new NumberType(DataKind.U1, "U1"), null);
                return _instU1;
            }
        }

        private static volatile NumberType _instI2;
        public static NumberType I2
        {
            get
            {
                if (_instI2 == null)
                    Interlocked.CompareExchange(ref _instI2, new NumberType(DataKind.I2, "I2"), null);
                return _instI2;
            }
        }

        private static volatile NumberType _instU2;
        public static NumberType U2
        {
            get
            {
                if (_instU2 == null)
                    Interlocked.CompareExchange(ref _instU2, new NumberType(DataKind.U2, "U2"), null);
                return _instU2;
            }
        }

        private static volatile NumberType _instI4;
        public static NumberType I4
        {
            get
            {
                if (_instI4 == null)
                    Interlocked.CompareExchange(ref _instI4, new NumberType(DataKind.I4, "I4"), null);
                return _instI4;
            }
        }

        private static volatile NumberType _instU4;
        public static NumberType U4
        {
            get
            {
                if (_instU4 == null)
                    Interlocked.CompareExchange(ref _instU4, new NumberType(DataKind.U4, "U4"), null);
                return _instU4;
            }
        }

        private static volatile NumberType _instI8;
        public static NumberType I8
        {
            get
            {
                if (_instI8 == null)
                    Interlocked.CompareExchange(ref _instI8, new NumberType(DataKind.I8, "I8"), null);
                return _instI8;
            }
        }

        private static volatile NumberType _instU8;
        public static NumberType U8
        {
            get
            {
                if (_instU8 == null)
                    Interlocked.CompareExchange(ref _instU8, new NumberType(DataKind.U8, "U8"), null);
                return _instU8;
            }
        }

        private static volatile NumberType _instUG;
        public static NumberType UG
        {
            get
            {
                if (_instUG == null)
                    Interlocked.CompareExchange(ref _instUG, new NumberType(DataKind.UG, "UG"), null);
                return _instUG;
            }
        }

        private static volatile NumberType _instR4;
        public static NumberType R4
        {
            get
            {
                if (_instR4 == null)
                    Interlocked.CompareExchange(ref _instR4, new NumberType(DataKind.R4, "R4"), null);
                return _instR4;
            }
        }

        private static volatile NumberType _instR8;
        public static NumberType R8
        {
            get
            {
                if (_instR8 == null)
                    Interlocked.CompareExchange(ref _instR8, new NumberType(DataKind.R8, "R8"), null);
                return _instR8;
            }
        }

        public static NumberType Float => R4;

        [BestFriend]
        internal static new NumberType FromKind(DataKind kind)
        {
            switch (kind)
            {
                case DataKind.I1:
                    return I1;
                case DataKind.U1:
                    return U1;
                case DataKind.I2:
                    return I2;
                case DataKind.U2:
                    return U2;
                case DataKind.I4:
                    return I4;
                case DataKind.U4:
                    return U4;
                case DataKind.I8:
                    return I8;
                case DataKind.U8:
                    return U8;
                case DataKind.R4:
                    return R4;
                case DataKind.R8:
                    return R8;
                case DataKind.UG:
                    return UG;
            }

            Contracts.Assert(false);
            throw Contracts.Except($"Bad data kind in {nameof(NumberType)}.{nameof(FromKind)}: {kind}");
        }

        [BestFriend]
        internal static NumberType FromType(Type type)
        {
            DataKind kind;
            if (type.TryGetDataKind(out kind))
                return FromKind(kind);

            Contracts.Assert(false);
            throw Contracts.Except($"Bad data kind in {nameof(NumberType)}.{nameof(FromType)}: {kind}", kind);
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Contracts.Assert(other == null || !other.IsNumber || other.RawKind != RawKind);
            return false;
        }

        public override string ToString() => _name;
    }

    /// <summary>
    /// The standard boolean type.
    /// </summary>
    public sealed class BoolType : PrimitiveType
    {
        private static volatile BoolType _instance;
        public static BoolType Instance
        {
            get
            {
                if (_instance == null)
                    Interlocked.CompareExchange(ref _instance, new BoolType(), null);
                return _instance;
            }
        }

        private BoolType()
            : base(typeof(bool), DataKind.BL)
        {
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Contracts.Assert(!(other is BoolType));
            return false;
        }

        public override string ToString()
        {
            return "Bool";
        }
    }

    public sealed class DateTimeType : PrimitiveType
    {
        private static volatile DateTimeType _instance;
        public static DateTimeType Instance
        {
            get
            {
                if (_instance == null)
                    Interlocked.CompareExchange(ref _instance, new DateTimeType(), null);
                return _instance;
            }
        }

        private DateTimeType()
            : base(typeof(DateTime), DataKind.DT)
        {
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Contracts.Assert(!(other is DateTimeType));
            return false;
        }

        public override string ToString() => "DateTime";
    }

    public sealed class DateTimeOffsetType : PrimitiveType
    {
        private static volatile DateTimeOffsetType _instance;
        public static DateTimeOffsetType Instance
        {
            get
            {
                if (_instance == null)
                    Interlocked.CompareExchange(ref _instance, new DateTimeOffsetType(), null);
                return _instance;
            }
        }

        private DateTimeOffsetType()
            : base(typeof(DateTimeOffset), DataKind.DZ)
        {
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Contracts.Assert(!(other is DateTimeOffsetType));
            return false;
        }

        public override string ToString() => "DateTimeZone";
    }

    /// <summary>
    /// The standard timespan type.
    /// </summary>
    public sealed class TimeSpanType : PrimitiveType
    {
        private static volatile TimeSpanType _instance;
        public static TimeSpanType Instance
        {
            get
            {
                if (_instance == null)
                    Interlocked.CompareExchange(ref _instance, new TimeSpanType(), null);
                return _instance;
            }
        }

        private TimeSpanType()
            : base(typeof(TimeSpan), DataKind.TS)
        {
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Contracts.Assert(!(other is TimeSpanType));
            return false;
        }

        public override string ToString() => "TimeSpan";
    }

    /// <summary>
    /// KeyTypes are for "id"-like data. The information happens to be stored in an unsigned integer
    /// type, but the information is not inherently numeric, so, typically, arithmetic is not
    /// meaningful. Examples are SSNs, phone numbers, auto-generated/incremented key values,
    /// class numbers, etc. For example, in multi-class classification, the label is typically
    /// a class number which is naturally a KeyType.
    ///
    /// KeyTypes can be contiguous (the class number example), in which case they can have
    /// a cardinality/Count. For non-contiguous KeyTypes the Count property returns zero.
    /// Any KeyType (contiguous or not) can have a Min value. The Min value is always >= 0.
    ///
    /// Note that the representation value does not necessarily match the logical value.
    /// For example, if a KeyType has range 1000-5000, then it has a Min of 1000, Count
    /// of 4001, but the representational values are 1-4001. The representation value zero
    /// is reserved to mean none/invalid.
    /// </summary>
    public sealed class KeyType : PrimitiveType
    {
        private KeyType(Type type, DataKind kind, ulong min, int count, bool contiguous)
            : base(type, kind)
        {
            Contracts.AssertValue(type);
            Contracts.Assert(kind.ToType() == type);

            Contracts.CheckParam(min >= 0, nameof(min));
            Contracts.CheckParam(count >= 0, nameof(count), "Must be non-negative.");
            Contracts.CheckParam((ulong)count <= ulong.MaxValue - min, nameof(count));
            Contracts.CheckParam((ulong)count <= kind.ToMaxInt(), nameof(count));
            Contracts.CheckParam(contiguous || count == 0, nameof(count), "Must be 0 for non-contiguous");

            Contiguous = contiguous;
            Min = min;
            Count = count;
            Contracts.Assert(IsKey);
        }

        public KeyType(Type type, ulong min, int count, bool contiguous = true)
            : this(type, CheckRefRawType(type), min, count, contiguous)
        {
        }

        [BestFriend]
        internal KeyType(DataKind kind, ulong min, int count, bool contiguous = true)
            : this(ToRawType(kind), kind, min, count, contiguous)
        {
        }

        private static DataKind CheckRefRawType(Type type)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(IsValidDataType(type), nameof(type));
            var result = type.TryGetDataKind(out var kind);
            Contracts.Assert(result);
            return kind;

        }

        private static Type ToRawType(DataKind kind)
        {
            Contracts.CheckParam(IsValidDataKind(kind), nameof(kind));
            return kind.ToType();
        }

        /// <summary>
        /// Returns true iff the given DataKind is valid for a <see cref="KeyType"/>. The valid ones are
        /// <see cref="DataKind.U1"/>, <see cref="DataKind.U2"/>, <see cref="DataKind.U4"/>, and <see cref="DataKind.U8"/>,
        /// that is, the unsigned integer kinds.
        /// </summary>
        [BestFriend]
        internal static bool IsValidDataKind(DataKind kind)
        {
            switch (kind)
            {
                case DataKind.U1:
                case DataKind.U2:
                case DataKind.U4:
                case DataKind.U8:
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        /// Returns true iff the given type is valid for a <see cref="KeyType"/>. The valid ones are
        /// <see cref="byte"/>, <see cref="ushort"/>, <see cref="uint"/>, and <see cref="ulong"/>, that is, the unsigned integer types.
        /// </summary>
        public static bool IsValidDataType(Type type)
        {
            Contracts.CheckValue(type, nameof(type));
            return type == typeof(byte) || type == typeof(ushort) || type == typeof(uint) || type == typeof(ulong);
        }

        private protected override int KeyCountCore => Count;

        /// <summary>
        /// This is the Min of the key type for display purposes and conversion to/from text. The values
        /// actually stored always start at 1 (for the smallest legal value), with zero being reserved
        /// for "not there"/"none". Typical Min values are 0 or 1, but can be any value >= 0.
        /// </summary>
        public ulong Min { get; }

        /// <summary>
        /// If this key type has contiguous values and a known cardinality, Count is that cardinality.
        /// Otherwise, this returns zero. Note that such a key type can be converted to a bit vector
        /// representation by mapping to a vector of length Count, with "id" mapped to a vector with
        /// 1 in slot (id - 1) and 0 in all other slots. This is the standard "indicator"
        /// representation. Note that an id of 0 is used to represent the notion "none", which is
        /// typically mapped to a vector of all zeros (of length Count).
        /// </summary>
        public int Count { get; }

        public bool Contiguous { get; }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;

            var tmp = other as KeyType;
            if (tmp == null)
                return false;
            if (RawKind != tmp.RawKind)
                return false;
            Contracts.Assert(RawType == tmp.RawType);
            if (Contiguous != tmp.Contiguous)
                return false;
            if (Min != tmp.Min)
                return false;
            if (Count != tmp.Count)
                return false;
            return true;
        }

        public override bool Equals(object other)
        {
            return other is ColumnType tmp && Equals(tmp);
        }

        public override int GetHashCode()
        {
            return Hashing.CombinedHash(RawKind.GetHashCode(), Contiguous, Min, Count);
        }

        public override string ToString()
        {
            if (Count > 0)
                return string.Format("Key<{0}, {1}-{2}>", RawKind.GetString(), Min, Min + (ulong)Count - 1);
            if (Contiguous)
                return string.Format("Key<{0}, {1}-*>", RawKind.GetString(), Min);
            // This is the non-contiguous case - simply show the Min.
            return string.Format("Key<{0}, Min:{1}>", RawKind.GetString(), Min);
        }
    }

    /// <summary>
    /// The standard vector type.
    /// </summary>
    public sealed class VectorType : StructuredType
    {
        /// <summary>b
        /// The dimensions. This will always have at least one item. All values will be non-negative.
        /// As with <see cref="Size"/>, a zero value indicates that the vector type is considered to have
        /// unknown length along that dimension.
        /// </summary>
        public ImmutableArray<int> Dimensions { get; }

        /// <summary>
        /// Constructs a new single-dimensional vector type.
        /// </summary>
        /// <param name="itemType">The type of the items contained in the vector.</param>
        /// <param name="size">The size of the single dimension.</param>
        public VectorType(PrimitiveType itemType, int size = 0)
            : base(GetRawType(itemType), 0)
        {
            Contracts.CheckParam(size >= 0, nameof(size));

            ItemType = itemType;
            Size = size;
            Dimensions = ImmutableArray.Create(Size);
        }

        /// <summary>
        /// Constructs a potentially multi-dimensional vector type.
        /// </summary>
        /// <param name="itemType">The type of the items contained in the vector.</param>
        /// <param name="dimensions">The dimensions. Note that, like <see cref="Dimensions"/>, must be non-empty, with all
        /// non-negative values. Also, because <see cref="Size"/> is the product of <see cref="Dimensions"/>, the result of
        /// multiplying all these values together must not overflow <see cref="int"/>.</param>
        public VectorType(PrimitiveType itemType, params int[] dimensions)
            : base(GetRawType(itemType), default)
        {
            Contracts.CheckParam(Utils.Size(dimensions) > 0, nameof(dimensions));
            Contracts.CheckParam(dimensions.All(d => d >= 0), nameof(dimensions));

            ItemType = itemType;
            Dimensions = dimensions.ToImmutableArray();
            Size = ComputeSize(Dimensions);
        }

        /// <summary>
        /// Creates a <see cref="VectorType"/> whose dimensionality information is the given <paramref name="template"/>'s information.
        /// </summary>
        [BestFriend]
        internal VectorType(PrimitiveType itemType, VectorType template)
            : base(GetRawType(itemType), default)
        {
            Contracts.CheckValue(template, nameof(template));

            ItemType = itemType;
            Dimensions = template.Dimensions;
            Size = template.Size;
        }

        /// <summary>
        /// Creates a <see cref="VectorType"/> whose dimensionality information is the given <paramref name="template"/>'s information,
        /// concatenated with the specified <paramref name="dims"/>.
        /// </summary>
        [BestFriend]
        internal VectorType(PrimitiveType itemType, VectorType template, params int[] dims)
            : base(GetRawType(itemType), default)
        {
            Contracts.CheckValue(template, nameof(template));
            Contracts.CheckParam(Utils.Size(dims) > 0, nameof(dims));
            Contracts.CheckParam(dims.All(d => d >= 0), nameof(dims));

            ItemType = itemType;
            Dimensions = template.Dimensions.AddRange(dims);
            Size = ComputeSize(Dimensions);
        }

        private static Type GetRawType(PrimitiveType itemType)
        {
            Contracts.CheckValue(itemType, nameof(itemType));
            return typeof(VBuffer<>).MakeGenericType(itemType.RawType);
        }

        private static int ComputeSize(ImmutableArray<int> dims)
        {
            int size = 1;
            for (int i = 0; i < dims.Length; ++i)
                size = checked(size * dims[i]);
            return size;
        }

        /// <summary>
        /// The type of the items stored as values in vectors of this type.
        /// </summary>
        public new PrimitiveType ItemType { get; }

        /// <summary>
        /// The size of the vector. A value of zero means it is a vector whose size is unknown.
        /// A vector whose size is known should correspond to values that always have the same <see cref="VBuffer{T}.Length"/>,
        /// whereas one whose size is known may have values whose <see cref="VBuffer{T}.Length"/> varies from record to record.
        /// Note that this is always the product of the elements in <see cref="Dimensions"/>.
        /// </summary>
        public int Size { get; }

        private protected override ColumnType ItemTypeCore => ItemType;

        private protected override int VectorSizeCore => Size;

        private protected override int ValueCountCore => Size;

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            if (!(other is VectorType tmp))
                return false;
            if (!ItemType.Equals(tmp.ItemType))
                return false;
            if (Size != tmp.Size)
                return false;
            if (Dimensions.Length != tmp.Dimensions.Length)
                return false;
            for (int i = 0; i < Dimensions.Length; i++)
            {
                if (Dimensions[i] != tmp.Dimensions[i])
                    return false;
            }
            return true;
        }

        public override bool Equals(object other)
        {
            return other is ColumnType tmp && Equals(tmp);
        }

        public override int GetHashCode()
        {
            int hash = Hashing.CombinedHash(ItemType.GetHashCode(), Size);
            hash = Hashing.CombineHash(hash, Dimensions.Length);
            for (int i = 0; i < Dimensions.Length; i++)
                hash = Hashing.CombineHash(hash, Dimensions[i].GetHashCode());
            return hash;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append("Vec<").Append(ItemType);

            if (Dimensions.Length == 1)
            {
                if (Size > 0)
                    sb.Append(", ").Append(Size);
            }
            else
            {
                foreach (var dim in Dimensions)
                {
                    sb.Append(", ");
                    if (dim > 0)
                        sb.Append(dim);
                    else
                        sb.Append('*');
                }
            }
            sb.Append(">");

            return sb.ToString();
        }
    }
}