// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// ColumnType is the abstract base class for all types in the IDataView type system.
    /// </summary>
    public abstract class ColumnType : IEquatable<ColumnType>
    {
        private readonly Type _rawType;
        private readonly DataKind _rawKind;

        // We cache these for speed and code size.
        private readonly bool _isPrimitive;
        private readonly bool _isVector;
        private readonly bool _isNumber;
        private readonly bool _isKey;

        // This private constructor sets all the _isXxx flags. It is invoked by other ctors.
        private ColumnType()
        {
            _isPrimitive = this is PrimitiveType;
            _isVector = this is VectorType;
            _isNumber = this is NumberType;
            _isKey = this is KeyType;
        }

        protected ColumnType(Type rawType)
            : this()
        {
            Contracts.CheckValue(rawType, nameof(rawType));
            _rawType = rawType;
            _rawType.TryGetDataKind(out _rawKind);
        }

        /// <summary>
        /// Internal sub types can pass both the rawType and rawKind values. This asserts that they
        /// are consistent.
        /// </summary>
        internal ColumnType(Type rawType, DataKind rawKind)
            : this()
        {
            Contracts.AssertValue(rawType);
#if DEBUG
            DataKind tmp;
            rawType.TryGetDataKind(out tmp);
            Contracts.Assert(tmp == rawKind);
#endif
            _rawType = rawType;
            _rawKind = rawKind;
        }

        /// <summary>
        /// The raw System.Type for this ColumnType. Note that this is the raw representation type
        /// and NOT the complete information content of the ColumnType. Code should not assume that
        /// a RawType uniquely identifiers a ColumnType.
        /// </summary>
        public Type RawType { get { return _rawType; } }

        /// <summary>
        /// The DataKind corresponding to RawType, if there is one (zero otherwise). It is equivalent
        /// to the result produced by DataKindExtensions.TryGetDataKind(RawType, out kind).
        /// </summary>
        public DataKind RawKind { get { return _rawKind; } }

        /// <summary>
        /// Whether this is a primitive type.
        /// </summary>
        public bool IsPrimitive { get { return _isPrimitive; } }

        /// <summary>
        /// Equivalent to "this as PrimitiveType".
        /// </summary>
        public PrimitiveType AsPrimitive { get { return _isPrimitive ? (PrimitiveType)this : null; } }

        /// <summary>
        /// Whether this type is a standard numeric type.
        /// </summary>
        public bool IsNumber { get { return _isNumber; } }

        /// <summary>
        /// Whether this type is the standard text type.
        /// </summary>
        public bool IsText
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
        public bool IsBool
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
        /// Whether this type is the standard timespan type.
        /// </summary>
        public bool IsTimeSpan
        {
            get
            {
                if (!(this is TimeSpanType))
                    return false;
                // TimeSpanType is a singleton.
                Contracts.Assert(this == TimeSpanType.Instance);
                return true;
            }
        }

        /// <summary>
        /// Whether this type is a DvDateTime.
        /// </summary>
        public bool IsDateTime
        {
            get
            {
                if (!(this is DateTimeType))
                    return false;
                // DateTimeType is a singleton.
                Contracts.Assert(this == DateTimeType.Instance);
                return true;
            }
        }

        /// <summary>
        /// Whether this type is a DvDateTimeZone.
        /// </summary>
        public bool IsDateTimeZone
        {
            get
            {
                if (!(this is DateTimeZoneType))
                    return false;
                // DateTimeZoneType is a singleton.
                Contracts.Assert(this == DateTimeZoneType.Instance);
                return true;
            }
        }

        /// <summary>
        /// Whether this type is a standard scalar type completely determined by its RawType
        /// (not a KeyType or StructureType, etc).
        /// </summary>
        public bool IsStandardScalar
        {
            get { return IsNumber || IsText || IsBool || IsTimeSpan || IsDateTime || IsDateTimeZone; }
        }

        /// <summary>
        /// Whether this type is a key type, which implies that the order of values is not significant,
        /// and arithmetic is non-sensical. A key type can define a cardinality.
        /// </summary>
        public bool IsKey { get { return _isKey; } }

        /// <summary>
        /// Equivalent to "this as KeyType".
        /// </summary>
        public KeyType AsKey { get { return _isKey ? (KeyType)this : null; } }

        /// <summary>
        /// Zero return means either it's not a key type or the cardinality is unknown.
        /// </summary>
        public int KeyCount { get { return KeyCountCore; } }

        /// <summary>
        /// The only sub-class that should override this is KeyType!
        /// </summary>
        internal virtual int KeyCountCore { get { return 0; } }

        /// <summary>
        /// Whether this is a vector type.
        /// </summary>
        public bool IsVector { get { return _isVector; } }

        /// <summary>
        /// Equivalent to "this as VectorType".
        /// </summary>
        public VectorType AsVector { get { return _isVector ? (VectorType)this : null; } }

        /// <summary>
        /// For non-vector types, this returns the column type itself (ie, return this).
        /// </summary>
        public ColumnType ItemType { get { return ItemTypeCore; } }

        /// <summary>
        /// Whether this is a vector type with known size. Returns false for non-vector types.
        /// Equivalent to VectorSize > 0.
        /// </summary>
        public bool IsKnownSizeVector { get { return VectorSize > 0; } }

        /// <summary>
        /// Zero return means either it's not a vector or the size is unknown. Equivalent to
        /// IsVector ? ValueCount : 0 and to IsKnownSizeVector ? ValueCount : 0.
        /// </summary>
        public int VectorSize { get { return VectorSizeCore; } }

        /// <summary>
        /// For non-vectors, this returns one. For unknown size vectors, it returns zero.
        /// Equivalent to IsVector ? VectorSize : 1.
        /// </summary>
        public int ValueCount { get { return ValueCountCore; } }

        /// <summary>
        /// The only sub-class that should override this is VectorType!
        /// </summary>
        internal virtual ColumnType ItemTypeCore { get { return this; } }

        /// <summary>
        /// The only sub-class that should override this is VectorType!
        /// </summary>
        internal virtual int VectorSizeCore { get { return 0; } }

        /// <summary>
        /// The only sub-class that should override this is VectorType!
        /// </summary>
        internal virtual int ValueCountCore { get { return 1; } }

        // IEquatable<T> interface recommends also to override base class implementations of
        // Object.Equals(Object) and GetHashCode. In classes below where Equals(ColumnType other)
        // is effectively a referencial comparison, there is no need to override base class implementations
        // of Object.Equals(Object) (and GetHashCode) since its also a referencial comparison.
        public abstract bool Equals(ColumnType other);

        /// <summary>
        /// Equivalent to calling Equals(ColumnType) for non-vector types. For vector type,
        /// returns true if current and other vector types have the same size and item type.
        /// </summary>
        public bool SameSizeAndItemType(ColumnType other)
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

        internal StructuredType(Type rawType, DataKind rawKind)
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
                "A PrimitiveType cannot have a disposable RawType");
        }

        internal PrimitiveType(Type rawType, DataKind rawKind)
            : base(rawType, rawKind)
        {
            Contracts.Assert(IsPrimitive);
            Contracts.Assert(!typeof(IDisposable).IsAssignableFrom(RawType));
        }

        public static PrimitiveType FromKind(DataKind kind)
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
                return DateTimeZoneType.Instance;
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
            : base(typeof(DvText), DataKind.TX)
        {
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Contracts.Assert(!(other is TextType));
            return false;
        }

        public override string ToString()
        {
            return "Text";
        }
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

        private static volatile NumberType _instNI1;
        public static NumberType NI1
        {
            get
            {
                if (_instNI1 == null)
                    Interlocked.CompareExchange(ref _instNI1, new NumberType(DataKind.NI1, "NI1"), null);
                return _instNI1;
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

        private static volatile NumberType _instNI2;
        public static NumberType NI2
        {
            get
            {
                if (_instNI2 == null)
                    Interlocked.CompareExchange(ref _instNI2, new NumberType(DataKind.NI2, "NI2"), null);
                return _instNI2;
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

        private static volatile NumberType _instNI4;
        public static NumberType NI4
        {
            get
            {
                if (_instNI4 == null)
                    Interlocked.CompareExchange(ref _instNI4, new NumberType(DataKind.NI4, "NI4"), null);
                return _instNI4;
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

        private static volatile NumberType _instNI8;
        public static NumberType NI8
        {
            get
            {
                if (_instNI8 == null)
                    Interlocked.CompareExchange(ref _instNI8, new NumberType(DataKind.NI8, "NI8"), null);
                return _instNI8;
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

        public static NumberType Float
        {
            get { return R4; }
        }

        public static new NumberType FromKind(DataKind kind)
        {
            switch (kind)
            {
            case DataKind.I1:
                return I1;
            case DataKind.NI1:
                return NI1;
            case DataKind.U1:
                return U1;
            case DataKind.I2:
                return I2;
            case DataKind.NI2:
                return NI2;
            case DataKind.U2:
                return U2;
            case DataKind.I4:
                return I4;
            case DataKind.NI4:
                return NI4;
            case DataKind.U4:
                return U4;
            case DataKind.I8:
                return I8;
            case DataKind.NI8:
                return NI8;
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
            throw Contracts.Except("Bad data kind in NumericType.FromKind: {0}", kind);
        }

        public static NumberType FromType(Type type)
        {
            DataKind kind;
            if (type.TryGetDataKind(out kind))
                return FromKind(kind);

            Contracts.Assert(false);
            throw Contracts.Except("Bad data kind in NumericType.FromKind: {0}", kind);
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Contracts.Assert(other == null || !other.IsNumber || other.RawKind != RawKind);
            return false;
        }

        public override string ToString()
        {
            return _name;
        }
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
                    Interlocked.CompareExchange(ref _instance, new BoolType(DataKind.BL, "Bool"), null);
                return _instance;
            }
        }

        private readonly string _name;

        private BoolType(DataKind kind, string name)
            : base(kind.ToType(), kind)
        {
            Contracts.AssertNonEmpty(name);
            _name = name;
            Contracts.Assert(IsNumber);
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
            return _name;
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
            : base(typeof(DvDateTime), DataKind.DT)
        {
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Contracts.Assert(!(other is DateTimeType));
            return false;
        }

        public override string ToString()
        {
            return "DateTime";
        }
    }

    public sealed class DateTimeZoneType : PrimitiveType
    {
        private static volatile DateTimeZoneType _instance;
        public static DateTimeZoneType Instance
        {
            get
            {
                if (_instance == null)
                    Interlocked.CompareExchange(ref _instance, new DateTimeZoneType(), null);
                return _instance;
            }
        }

        private DateTimeZoneType()
            : base(typeof(DvDateTimeZone), DataKind.DZ)
        {
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Contracts.Assert(!(other is DateTimeZoneType));
            return false;
        }

        public override string ToString()
        {
            return "DateTimeZone";
        }
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
            : base(typeof(DvTimeSpan), DataKind.TS)
        {
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Contracts.Assert(!(other is TimeSpanType));
            return false;
        }

        public override string ToString()
        {
            return "TimeSpan";
        }
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
        private readonly bool _contiguous;
        private readonly ulong _min;
        // _count is only valid if _contiguous is true. Zero means unknown.
        private readonly int _count;

        public KeyType(DataKind kind, ulong min, int count, bool contiguous = true)
            : base(ToRawType(kind), kind)
        {
            Contracts.CheckParam(min >= 0, nameof(min));
            Contracts.CheckParam(count >= 0, nameof(count), "count for key type must be non-negative");
            Contracts.CheckParam((ulong)count <= ulong.MaxValue - min, nameof(count));
            Contracts.CheckParam((ulong)count <= kind.ToMaxInt(), nameof(count));
            Contracts.CheckParam(contiguous || count == 0, nameof(count), "count must be 0 for non-contiguous");

            _contiguous = contiguous;
            _min = min;
            _count = count;
            Contracts.Assert(IsKey);
        }

        private static Type ToRawType(DataKind kind)
        {
            Contracts.CheckParam(IsValidDataKind(kind), nameof(kind));
            return kind.ToType();
        }

        /// <summary>
        /// Returns true iff the given DataKind is valid for a KeyType. The valid ones are
        /// U1, U2, U4, and U8, that is, the unsigned integer kinds.
        /// </summary>
        public static bool IsValidDataKind(DataKind kind)
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

        internal override int KeyCountCore { get { return _count; } }

        /// <summary>
        /// This is the Min of the key type for display purposes and conversion to/from text. The values
        /// actually stored always start at 1 (for the smallest legal value), with zero being reserved
        /// for "not there"/"none". Typical Min values are 0 or 1, but can be any value >= 0.
        /// </summary>
        public ulong Min { get { return _min; } }

        /// <summary>
        /// If this key type has contiguous values and a known cardinality, Count is that cardinality.
        /// Otherwise, this returns zero. Note that such a key type can be converted to a bit vector
        /// representation by mapping to a vector of length Count, with "id" mapped to a vector with
        /// 1 in slot (id - 1) and 0 in all other slots. This is the standard "indicator"
        /// representation. Note that an id of 0 is used to represent the notion "none", which is
        /// typically mapped to a vector of all zeros (of length Count).
        /// </summary>
        public int Count { get { return _count; } }

        public bool Contiguous { get { return _contiguous; } }

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
            if (_contiguous != tmp._contiguous)
                return false;
            if (_min != tmp._min)
                return false;
            if (_count != tmp._count)
                return false;
            return true;
        }

        public override bool Equals(object other)
        {
            return other is ColumnType tmp && Equals(tmp);
        }

        public override int GetHashCode()
        {
            return Hashing.CombinedHash(RawKind.GetHashCode(), _contiguous, _min, _count);
        }

        public override string ToString()
        {
            if (_count > 0)
                return string.Format("Key<{0}, {1}-{2}>", RawKind.GetString(), _min, _min + (ulong)_count - 1);
            if (_contiguous)
                return string.Format("Key<{0}, {1}-*>", RawKind.GetString(), _min);
            // This is the non-contiguous case - simply show the Min.
            return string.Format("Key<{0}, Min:{1}>", RawKind.GetString(), _min);
        }
    }

    /// <summary>
    /// The standard vector type.
    /// </summary>
    public sealed class VectorType : StructuredType
    {
        private readonly PrimitiveType _itemType;
        private readonly int _size;

        // The _sizes are the cumulative products of the _dims. These may be null, meaning that
        // the information is naturally one dimensional.
        private readonly int[] _sizes;
        private readonly int[] _dims;

        public VectorType(PrimitiveType itemType, int size = 0)
            : base(GetRawType(itemType), 0)
        {
            Contracts.CheckParam(size >= 0, nameof(size));

            _itemType = itemType;
            _size = size;
        }

        public VectorType(PrimitiveType itemType, params int[] dims)
            : base(GetRawType(itemType), default(DataKind))
        {
            Contracts.CheckParam(Utils.Size(dims) > 0, nameof(dims));
            Contracts.CheckParam(dims.All(d => d >= 0), nameof(dims));

            _itemType = itemType;

            if (dims.Length == 1)
                _size = dims[0];
            else
            {
                _dims = new int[dims.Length];
                Array.Copy(dims, _dims, _dims.Length);
                _size = ComputeSizes(_dims, out _sizes);
            }
        }

        /// <summary>
        /// Creates a VectorType whose dimensionality information is the given template's information.
        /// </summary>
        public VectorType(PrimitiveType itemType, VectorType template)
            : base(GetRawType(itemType), default(DataKind))
        {
            Contracts.CheckValue(template, nameof(template));

            _itemType = itemType;
            _size = template._size;
            _sizes = template._sizes;
            _dims = template._dims;
        }

        /// <summary>
        /// Creates a VectorType whose dimensionality information is the given template's information
        /// concatenated with the specified dims.
        /// </summary>
        public VectorType(PrimitiveType itemType, VectorType template, params int[] dims)
            : base(GetRawType(itemType), default(DataKind))
        {
            Contracts.CheckValue(template, nameof(template));

            _itemType = itemType;

            if (template._dims == null)
                _dims = Utils.Concat(new int[] { template._size }, dims);
            else
            {
                Contracts.Assert(template._dims.Length >= 2);
                _dims = Utils.Concat(template._dims, dims);
            }
            _size = ComputeSizes(_dims, out _sizes);
        }

        private static Type GetRawType(PrimitiveType itemType)
        {
            Contracts.CheckValue(itemType, nameof(itemType));
            return typeof(VBuffer<>).MakeGenericType(itemType.RawType);
        }

        private static int ComputeSizes(int[] dims, out int[] sizes)
        {
            sizes = new int[dims.Length];
            int size = 1;
            for (int i = dims.Length; --i >= 0; )
                size = sizes[i] = checked(size * dims[i]);
            return size;
        }

        public int DimCount { get { return _dims != null ? _dims.Length : _size > 0 ? 1 : 0; } }

        public int GetDim(int idim)
        {
            if (_dims == null)
            {
                // That that if _size is zero, DimCount is zero, so this method is illegal
                // to call. That case is caught by Check(_size > 0).
                Contracts.Check(_size > 0);
                Contracts.Assert(DimCount == 1);
                Contracts.CheckParam(idim == 0, nameof(idim));
                return _size;
            }

            Contracts.CheckParam(0 <= idim && idim < _dims.Length, nameof(idim));
            return _dims[idim];
        }

        public new PrimitiveType ItemType { get { return _itemType; } }

        internal override ColumnType ItemTypeCore { get { return _itemType; } }

        internal override int VectorSizeCore { get { return _size; } }

        internal override int ValueCountCore { get { return _size; } }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            var tmp = other.AsVector;
            if (tmp == null)
                return false;
            if (!_itemType.Equals(tmp._itemType))
                return false;
            if (_size != tmp._size)
                return false;
            int count = Utils.Size(_dims);
            if (count != Utils.Size(tmp._dims))
                return false;
            if (count == 0)
                return true;
            for (int i = 0; i < count; i++)
            {
                if (_dims[i] != tmp._dims[i])
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
            int hash = Hashing.CombinedHash(_itemType.GetHashCode(), _size);
            int count = Utils.Size(_dims);
            hash = Hashing.CombineHash(hash, count.GetHashCode());
            for (int i = 0; i < count; i++)
                hash = Hashing.CombineHash(hash, _dims[i].GetHashCode());
            return hash;
        }

        /// <summary>
        /// Returns true if current has the same item type of other, and the size
        /// of other is unknown or the current size is equal to the size of other.
        /// </summary>
        public bool IsSubtypeOf(VectorType other)
        {
            if (other == this)
                return true;
            if (other == null)
                return false;

            // REVIEW: Perhaps we should allow the case when _itemType is
            // a sub-type of other._itemType (in particular for key types)
            if (!_itemType.Equals(other._itemType))
                return false;
            if (other._size == 0 || _size == other._size)
                return true;
            return false;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append("Vec<").Append(_itemType);

            if (_dims == null)
            {
                if (_size > 0)
                    sb.Append(", ").Append(_size);
            }
            else
            {
                foreach (var dim in _dims)
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