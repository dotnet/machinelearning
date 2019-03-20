// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.Reflection;
using System.Threading;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This is the abstract base class for all types in the <see cref="IDataView"/> type system.
    /// </summary>
    public abstract class DataViewType : IEquatable<DataViewType>
    {
        /// <summary>
        /// Constructor for extension types, which must be either <see cref="PrimitiveDataViewType"/> or <see cref="StructuredDataViewType"/>.
        /// </summary>
        private protected DataViewType(Type rawType)
        {
            RawType = rawType ?? throw new ArgumentNullException(nameof(rawType));
        }

        /// <summary>
        /// The raw <see cref="Type"/> for this <see cref="DataViewType"/>. Note that this is the raw representation type
        /// and not the complete information content of the <see cref="DataViewType"/>.
        /// </summary>
        /// <remarks>
        /// Code should not assume that a <see cref="RawType"/> uniquely identifiers a <see cref="DataViewType"/>.
        /// For example, most practical instances of ML.NET's KeyType and <see cref="NumberDataViewType.UInt32"/> will have a
        /// <see cref="RawType"/> of <see cref="uint"/>, but both are very different in the types of information conveyed in that number.
        /// </remarks>
        public Type RawType { get; }

        // IEquatable<T> interface recommends also to override base class implementations of
        // Object.Equals(Object) and GetHashCode. In classes below where Equals(ColumnType other)
        // is effectively a referencial comparison, there is no need to override base class implementations
        // of Object.Equals(Object) (and GetHashCode) since its also a referencial comparison.
        public abstract bool Equals(DataViewType other);
    }

    /// <summary>
    /// The abstract base class for all non-primitive types.
    /// </summary>
    public abstract class StructuredDataViewType : DataViewType
    {
        protected StructuredDataViewType(Type rawType)
            : base(rawType)
        {
        }
    }

    /// <summary>
    /// The abstract base class for all primitive types. Values of these types can be freely copied
    /// without concern for ownership, mutation, or disposing.
    /// </summary>
    public abstract class PrimitiveDataViewType : DataViewType
    {
        protected PrimitiveDataViewType(Type rawType)
            : base(rawType)
        {
            if (typeof(IDisposable).GetTypeInfo().IsAssignableFrom(RawType.GetTypeInfo()))
                throw new ArgumentException("A " + nameof(PrimitiveDataViewType) + " cannot have a disposable " + nameof(RawType), nameof(rawType));
        }
    }

    /// <summary>
    /// The standard text type.
    /// </summary>
    public sealed class TextDataViewType : PrimitiveDataViewType
    {
        private static volatile TextDataViewType _instance;
        public static TextDataViewType Instance
        {
            get
            {
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new TextDataViewType(), null) ??
                    _instance;
            }
        }

        private TextDataViewType()
            : base(typeof(ReadOnlyMemory<char>))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this)
                return true;
            Debug.Assert(!(other is TextDataViewType));
            return false;
        }

        public override string ToString() => "Text";
    }

    /// <summary>
    /// The standard number types.
    /// </summary>
    public sealed class NumberDataViewType : PrimitiveDataViewType
    {
        private readonly string _name;

        private NumberDataViewType(Type rawType, string name)
            : base(rawType)
        {
            Debug.Assert(!string.IsNullOrEmpty(name));
            _name = name;
        }

        private static volatile NumberDataViewType _instSByte;
        public static NumberDataViewType SByte
        {
            get
            {
                return _instSByte ??
                    Interlocked.CompareExchange(ref _instSByte, new NumberDataViewType(typeof(sbyte), "I1"), null) ??
                    _instSByte;
            }
        }

        private static volatile NumberDataViewType _instByte;
        public static NumberDataViewType Byte
        {
            get
            {
                return _instByte ??
                    Interlocked.CompareExchange(ref _instByte, new NumberDataViewType(typeof(byte), "U1"), null) ??
                    _instByte;
            }
        }

        private static volatile NumberDataViewType _instInt16;
        public static NumberDataViewType Int16
        {
            get
            {
                return _instInt16 ??
                    Interlocked.CompareExchange(ref _instInt16, new NumberDataViewType(typeof(short), "I2"), null) ??
                    _instInt16;
            }
        }

        private static volatile NumberDataViewType _instUInt16;
        public static NumberDataViewType UInt16
        {
            get
            {
                return _instUInt16 ??
                    Interlocked.CompareExchange(ref _instUInt16, new NumberDataViewType(typeof(ushort), "U2"), null) ??
                    _instUInt16;
            }
        }

        private static volatile NumberDataViewType _instInt32;
        public static NumberDataViewType Int32
        {
            get
            {
                return _instInt32 ??
                    Interlocked.CompareExchange(ref _instInt32, new NumberDataViewType(typeof(int), "I4"), null) ??
                    _instInt32;
            }
        }

        private static volatile NumberDataViewType _instUInt32;
        public static NumberDataViewType UInt32
        {
            get
            {
                return _instUInt32 ??
                    Interlocked.CompareExchange(ref _instUInt32, new NumberDataViewType(typeof(uint), "U4"), null) ??
                    _instUInt32;
            }
        }

        private static volatile NumberDataViewType _instInt64;
        public static NumberDataViewType Int64
        {
            get
            {
                return _instInt64 ??
                    Interlocked.CompareExchange(ref _instInt64, new NumberDataViewType(typeof(long), "I8"), null) ??
                    _instInt64;
            }
        }

        private static volatile NumberDataViewType _instUInt64;
        public static NumberDataViewType UInt64
        {
            get
            {
                return _instUInt64 ??
                    Interlocked.CompareExchange(ref _instUInt64, new NumberDataViewType(typeof(ulong), "U8"), null) ??
                    _instUInt64;
            }
        }

        private static volatile NumberDataViewType _instSingle;
        public static NumberDataViewType Single
        {
            get
            {
                return _instSingle ??
                    Interlocked.CompareExchange(ref _instSingle, new NumberDataViewType(typeof(float), "R4"), null) ??
                    _instSingle;
            }
        }

        private static volatile NumberDataViewType _instDouble;
        public static NumberDataViewType Double
        {
            get
            {
                return _instDouble ??
                    Interlocked.CompareExchange(ref _instDouble, new NumberDataViewType(typeof(double), "R8"), null) ??
                    _instDouble;
            }
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this)
                return true;
            Debug.Assert(other == null || !(other is NumberDataViewType) || other.RawType != RawType);
            return false;
        }

        public override string ToString() => _name;
    }

    /// <summary>
    /// The DataViewRowId type.
    /// </summary>
    public sealed class RowIdDataViewType : PrimitiveDataViewType
    {
        private static volatile RowIdDataViewType _instance;
        public static RowIdDataViewType Instance
        {
            get
            {
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new RowIdDataViewType(), null) ??
                    _instance;
            }
        }

        private RowIdDataViewType()
            : base(typeof(DataViewRowId))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this)
                return true;
            Debug.Assert(!(other is RowIdDataViewType));
            return false;
        }

        public override string ToString()
        {
            return "DataViewRowId";
        }
    }

    /// <summary>
    /// The standard boolean type.
    /// </summary>
    public sealed class BooleanDataViewType : PrimitiveDataViewType
    {
        private static volatile BooleanDataViewType _instance;
        public static BooleanDataViewType Instance
        {
            get
            {
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new BooleanDataViewType(), null) ??
                    _instance;
            }
        }

        private BooleanDataViewType()
            : base(typeof(bool))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this)
                return true;
            Debug.Assert(!(other is BooleanDataViewType));
            return false;
        }

        public override string ToString()
        {
            return "Bool";
        }
    }

    public sealed class DateTimeDataViewType : PrimitiveDataViewType
    {
        private static volatile DateTimeDataViewType _instance;
        public static DateTimeDataViewType Instance
        {
            get
            {
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new DateTimeDataViewType(), null) ??
                    _instance;
            }
        }

        private DateTimeDataViewType()
            : base(typeof(DateTime))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this)
                return true;
            Debug.Assert(!(other is DateTimeDataViewType));
            return false;
        }

        public override string ToString() => "DateTime";
    }

    public sealed class DateTimeOffsetDataViewType : PrimitiveDataViewType
    {
        private static volatile DateTimeOffsetDataViewType _instance;
        public static DateTimeOffsetDataViewType Instance
        {
            get
            {
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new DateTimeOffsetDataViewType(), null) ??
                    _instance;
            }
        }

        private DateTimeOffsetDataViewType()
            : base(typeof(DateTimeOffset))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this)
                return true;
            Debug.Assert(!(other is DateTimeOffsetDataViewType));
            return false;
        }

        public override string ToString() => "DateTimeZone";
    }

    /// <summary>
    /// The standard timespan type.
    /// </summary>
    public sealed class TimeSpanDataViewType : PrimitiveDataViewType
    {
        private static volatile TimeSpanDataViewType _instance;
        public static TimeSpanDataViewType Instance
        {
            get
            {
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new TimeSpanDataViewType(), null) ??
                    _instance;
            }
        }

        private TimeSpanDataViewType()
            : base(typeof(TimeSpan))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this)
                return true;
            Debug.Assert(!(other is TimeSpanDataViewType));
            return false;
        }

        public override string ToString() => "TimeSpan";
    }
}