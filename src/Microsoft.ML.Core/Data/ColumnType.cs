// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This is the abstract base class for all types in the <see cref="IDataView"/> type system.
    /// </summary>
    public abstract class ColumnType : IEquatable<ColumnType>
    {
        /// <summary>
        /// Constructor for extension types, which must be either <see cref="PrimitiveType"/> or <see cref="StructuredType"/>.
        /// </summary>
        private protected ColumnType(Type rawType)
        {
            Contracts.CheckValue(rawType, nameof(rawType));
            RawType = rawType;
        }

        /// <summary>
        /// The raw <see cref="Type"/> for this <see cref="ColumnType"/>. Note that this is the raw representation type
        /// and not the complete information content of the <see cref="ColumnType"/>. Code should not assume that
        /// a <see cref="RawType"/> uniquely identifiers a <see cref="ColumnType"/>. For example, most practical instances of
        /// <see cref="KeyType"/> and <see cref="NumberType.U4"/> will have a <see cref="RawType"/> of <see cref="uint"/>,
        /// but both are very different in the types of information conveyed in that number.
        /// </summary>
        public Type RawType { get; }

        // IEquatable<T> interface recommends also to override base class implementations of
        // Object.Equals(Object) and GetHashCode. In classes below where Equals(ColumnType other)
        // is effectively a referencial comparison, there is no need to override base class implementations
        // of Object.Equals(Object) (and GetHashCode) since its also a referencial comparison.
        public abstract bool Equals(ColumnType other);
    }

    /// <summary>
    /// The abstract base class for all non-primitive types.
    /// </summary>
    public abstract class StructuredType : ColumnType
    {
        protected StructuredType(Type rawType)
            : base(rawType)
        {
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
            Contracts.CheckParam(!typeof(IDisposable).IsAssignableFrom(RawType), nameof(rawType),
                "A " + nameof(PrimitiveType) + " cannot have a disposable " + nameof(RawType));
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

        [BestFriend]
        internal static PrimitiveType FromType(Type type)
        {
            if (type == typeof(ReadOnlyMemory<char>))
                return TextType.Instance;
            if (type == typeof(bool))
                return BoolType.Instance;
            if (type == typeof(TimeSpan))
                return TimeSpanType.Instance;
            if (type == typeof(DateTime))
                return DateTimeType.Instance;
            if (type == typeof(DateTimeOffset))
                return DateTimeOffsetType.Instance;
            return NumberType.FromType(type);
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
            : base(typeof(ReadOnlyMemory<char>))
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

        private NumberType(Type rawType, string name)
            : base(rawType)
        {
            Contracts.AssertNonEmpty(name);
            _name = name;
        }

        private static volatile NumberType _instI1;
        public static NumberType I1
        {
            get
            {
                if (_instI1 == null)
                    Interlocked.CompareExchange(ref _instI1, new NumberType(typeof(sbyte), "I1"), null);
                return _instI1;
            }
        }

        private static volatile NumberType _instU1;
        public static NumberType U1
        {
            get
            {
                if (_instU1 == null)
                    Interlocked.CompareExchange(ref _instU1, new NumberType(typeof(byte), "U1"), null);
                return _instU1;
            }
        }

        private static volatile NumberType _instI2;
        public static NumberType I2
        {
            get
            {
                if (_instI2 == null)
                    Interlocked.CompareExchange(ref _instI2, new NumberType(typeof(short), "I2"), null);
                return _instI2;
            }
        }

        private static volatile NumberType _instU2;
        public static NumberType U2
        {
            get
            {
                if (_instU2 == null)
                    Interlocked.CompareExchange(ref _instU2, new NumberType(typeof(ushort), "U2"), null);
                return _instU2;
            }
        }

        private static volatile NumberType _instI4;
        public static NumberType I4
        {
            get
            {
                if (_instI4 == null)
                    Interlocked.CompareExchange(ref _instI4, new NumberType(typeof(int), "I4"), null);
                return _instI4;
            }
        }

        private static volatile NumberType _instU4;
        public static NumberType U4
        {
            get
            {
                if (_instU4 == null)
                    Interlocked.CompareExchange(ref _instU4, new NumberType(typeof(uint), "U4"), null);
                return _instU4;
            }
        }

        private static volatile NumberType _instI8;
        public static NumberType I8
        {
            get
            {
                if (_instI8 == null)
                    Interlocked.CompareExchange(ref _instI8, new NumberType(typeof(long), "I8"), null);
                return _instI8;
            }
        }

        private static volatile NumberType _instU8;
        public static NumberType U8
        {
            get
            {
                if (_instU8 == null)
                    Interlocked.CompareExchange(ref _instU8, new NumberType(typeof(ulong), "U8"), null);
                return _instU8;
            }
        }

        private static volatile NumberType _instUG;
        public static NumberType UG
        {
            get
            {
                if (_instUG == null)
                    Interlocked.CompareExchange(ref _instUG, new NumberType(typeof(RowId), "UG"), null);
                return _instUG;
            }
        }

        private static volatile NumberType _instR4;
        public static NumberType R4
        {
            get
            {
                if (_instR4 == null)
                    Interlocked.CompareExchange(ref _instR4, new NumberType(typeof(float), "R4"), null);
                return _instR4;
            }
        }

        private static volatile NumberType _instR8;
        public static NumberType R8
        {
            get
            {
                if (_instR8 == null)
                    Interlocked.CompareExchange(ref _instR8, new NumberType(typeof(double), "R8"), null);
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
        internal static new NumberType FromType(Type type)
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
            Contracts.Assert(other == null || !(other is NumberType) || other.RawType != RawType);
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
            : base(typeof(bool))
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
            : base(typeof(DateTime))
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
            : base(typeof(DateTimeOffset))
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
            : base(typeof(TimeSpan))
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
    /// KeyTypes have a cardinality (i.e., Count) that is strictly positive.
    ///
    /// Note that the underlying representation value does not necessarily match the logical value.
    /// For example, if a KeyType has range 0-5000, then it has a Count of 5001, but
    /// the representational values are 1-5001. The representation value zero is reserved
    /// to mean a missing value (similar to NaN).
    /// </summary>
    public sealed class KeyType : PrimitiveType
    {
        public KeyType(Type type, ulong count)
            : base(type)
        {
            Contracts.AssertValue(type);
            if (count == 0 || type.ToMaxInt() < count)
                throw Contracts.ExceptParam(nameof(count), "The cardinality of a {0} must not exceed {1}.MaxValue" +
                    " and must be strictly positive but got {2}.", nameof(KeyType), type.Name, count);
            Count = count;
        }

        public KeyType(Type type, int count)
            : this(type, (ulong)count)
        {
            Contracts.CheckParam(0 < count, nameof(count), "The cardinality of a " + nameof(KeyType) + " must be strictly positive.");
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

        /// <summary>
        /// <see cref="Count"/> is the cardinality of the <see cref="KeyType"/>. Note that such a key type can be converted to a
        /// bit vector representation by mapping to a vector of length Count, with "id" mapped to a
        /// vector with 1 in slot (id - 1) and 0 in all other slots. This is the standard "indicator"
        /// representation. Note that an id of 0 is used to represent the notion "none", which is
        /// typically mapped, by for example, one-hot encoding, to a vector of all zeros (of length Count).
        /// </summary>
        public ulong Count { get; }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;

            if (!(other is KeyType tmp))
                return false;
            if (RawType != tmp.RawType)
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
            return Hashing.CombinedHash(RawType.GetHashCode(), Count);
        }

        public override string ToString()
        {
            DataKind rawKind = this.GetRawKind();
            return string.Format("Key<{0}, {1}-{2}>", rawKind.GetString(), 0, Count - 1);
        }
    }

    /// <summary>
    /// The standard vector type.
    /// </summary>
    public sealed class VectorType : StructuredType
    {
        /// <summary>
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
            : base(GetRawType(itemType))
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
            : base(GetRawType(itemType))
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
            : base(GetRawType(itemType))
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
            : base(GetRawType(itemType))
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
        /// Whether this is a vector type with known size.
        /// Equivalent to <c><see cref="Size"/> &gt; 0</c>.
        /// </summary>
        public bool IsKnownSize => Size > 0;

        /// <summary>
        /// The type of the items stored as values in vectors of this type.
        /// </summary>
        public PrimitiveType ItemType { get; }

        /// <summary>
        /// The size of the vector. A value of zero means it is a vector whose size is unknown.
        /// A vector whose size is known should correspond to values that always have the same <see cref="VBuffer{T}.Length"/>,
        /// whereas one whose size is unknown may have values whose <see cref="VBuffer{T}.Length"/> varies from record to record.
        /// Note that this is always the product of the elements in <see cref="Dimensions"/>.
        /// </summary>
        public int Size { get; }

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
