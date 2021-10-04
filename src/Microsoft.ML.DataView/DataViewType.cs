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
    /// <remarks>
    /// Those that wish to extend the <see cref="IDataView"/> type system should derive from one of
    /// the more specific abstract classes <see cref="StructuredDataViewType"/> or <see cref="PrimitiveDataViewType"/>.
    /// </remarks>
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
        /// <summary>
        /// Return <see langword="true"/> if <see langword="this"/> is equivalent to <paramref name="other"/> and <see langword="false"/> otherwise.
        /// </summary>
        /// <param name="other">Another <see cref="DataViewType"/> to be compared with <see langword="this"/>.</param>
        public abstract bool Equals(DataViewType other);
    }

    /// <summary>
    /// The abstract base class for all non-primitive types.
    /// </summary>
    /// <remarks>
    /// This class stands in constrast to <see cref="PrimitiveDataViewType"/>. As that class is defined
    /// to encapsulate cases where instances of the representation type can be freely copied without concerns
    /// about ownership, mutability, or dispoal, this is defined for those types where these factors become concerns.
    ///
    /// To take the most conspicuous example, <see cref="VectorDataViewType"/> is a structure type,
    /// which through the buffer sharing mechanisms of its <see cref="VBuffer{T}"/> representation type,
    /// does not have assignment as sufficient to create an independent copy.
    /// </remarks>
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
    /// The standard text type. This has representation type of <see cref="ReadOnlyMemory{T}"/> with type parameter <see cref="char"/>.
    /// Note this can have only one possible value, accessible by the singleton static property <see cref="Instance"/>.
    /// </summary>
    public sealed class TextDataViewType : PrimitiveDataViewType
    {
        private static volatile TextDataViewType _instance;
        /// <summary>
        /// The singleton instance of this type.
        /// </summary>
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

        public override string ToString() => "String";
    }

    /// <summary>
    /// The standard number type. This class is not directly instantiable. All allowed instances of this
    /// type are singletons, and are accessible as static properties on this class.
    /// </summary>
    public sealed class NumberDataViewType : PrimitiveDataViewType
    {
        private NumberDataViewType(Type rawType)
            : base(rawType)
        {
        }

        private static volatile NumberDataViewType _instSByte;
        /// <summary>
        /// The singleton instance of the <see cref="NumberDataViewType"/> with representation type of <see cref="sbyte"/>.
        /// </summary>
        public static NumberDataViewType SByte
        {
            get
            {
                return _instSByte ??
                    Interlocked.CompareExchange(ref _instSByte, new NumberDataViewType(typeof(sbyte)), null) ??
                    _instSByte;
            }
        }

        private static volatile NumberDataViewType _instByte;
        /// <summary>
        /// The singleton instance of the <see cref="NumberDataViewType"/> with representation type of <see cref="byte"/>.
        /// </summary>
        public static NumberDataViewType Byte
        {
            get
            {
                return _instByte ??
                    Interlocked.CompareExchange(ref _instByte, new NumberDataViewType(typeof(byte)), null) ??
                    _instByte;
            }
        }

        private static volatile NumberDataViewType _instInt16;
        /// <summary>
        /// The singleton instance of the <see cref="NumberDataViewType"/> with representation type of <see cref="short"/>.
        /// </summary>
        public static NumberDataViewType Int16
        {
            get
            {
                return _instInt16 ??
                    Interlocked.CompareExchange(ref _instInt16, new NumberDataViewType(typeof(short)), null) ??
                    _instInt16;
            }
        }

        private static volatile NumberDataViewType _instUInt16;
        /// <summary>
        /// The singleton instance of the <see cref="NumberDataViewType"/> with representation type of <see cref="ushort"/>.
        /// </summary>
        public static NumberDataViewType UInt16
        {
            get
            {
                return _instUInt16 ??
                    Interlocked.CompareExchange(ref _instUInt16, new NumberDataViewType(typeof(ushort)), null) ??
                    _instUInt16;
            }
        }

        private static volatile NumberDataViewType _instInt32;
        /// <summary>
        /// The singleton instance of the <see cref="NumberDataViewType"/> with representation type of <see cref="int"/>.
        /// </summary>
        public static NumberDataViewType Int32
        {
            get
            {
                return _instInt32 ??
                    Interlocked.CompareExchange(ref _instInt32, new NumberDataViewType(typeof(int)), null) ??
                    _instInt32;
            }
        }

        private static volatile NumberDataViewType _instUInt32;
        /// <summary>
        /// The singleton instance of the <see cref="NumberDataViewType"/> with representation type of <see cref="uint"/>.
        /// </summary>
        public static NumberDataViewType UInt32
        {
            get
            {
                return _instUInt32 ??
                    Interlocked.CompareExchange(ref _instUInt32, new NumberDataViewType(typeof(uint)), null) ??
                    _instUInt32;
            }
        }

        private static volatile NumberDataViewType _instInt64;
        /// <summary>
        /// The singleton instance of the <see cref="NumberDataViewType"/> with representation type of <see cref="long"/>.
        /// </summary>
        public static NumberDataViewType Int64
        {
            get
            {
                return _instInt64 ??
                    Interlocked.CompareExchange(ref _instInt64, new NumberDataViewType(typeof(long)), null) ??
                    _instInt64;
            }
        }

        private static volatile NumberDataViewType _instUInt64;
        /// <summary>
        /// The singleton instance of the <see cref="NumberDataViewType"/> with representation type of <see cref="ulong"/>.
        /// </summary>
        public static NumberDataViewType UInt64
        {
            get
            {
                return _instUInt64 ??
                    Interlocked.CompareExchange(ref _instUInt64, new NumberDataViewType(typeof(ulong)), null) ??
                    _instUInt64;
            }
        }

        private static volatile NumberDataViewType _instSingle;
        /// <summary>
        /// The singleton instance of the <see cref="NumberDataViewType"/> with representation type of <see cref="float"/>.
        /// </summary>
        public static NumberDataViewType Single
        {
            get
            {
                return _instSingle ??
                    Interlocked.CompareExchange(ref _instSingle, new NumberDataViewType(typeof(float)), null) ??
                    _instSingle;
            }
        }

        private static volatile NumberDataViewType _instDouble;
        /// <summary>
        /// The singleton instance of the <see cref="NumberDataViewType"/> with representation type of <see cref="double"/>.
        /// </summary>
        public static NumberDataViewType Double
        {
            get
            {
                return _instDouble ??
                    Interlocked.CompareExchange(ref _instDouble, new NumberDataViewType(typeof(double)), null) ??
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

        public override string ToString() => RawType.Name;
    }

    /// <summary>
    /// The <see cref="RowIdDataViewType "/> type. This has representation type of <see cref="DataViewRowId"/>.
    /// Note this can have only one possible value, accessible by the singleton static property <see cref="Instance"/>.
    /// </summary>
    public sealed class RowIdDataViewType : PrimitiveDataViewType
    {
        private static volatile RowIdDataViewType _instance;
        /// <summary>
        /// The singleton instance of this type.
        /// </summary>
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
    /// The standard boolean type. This has representation type of <see cref="bool"/>.
    /// Note this can have only one possible value, accessible by the singleton static property <see cref="Instance"/>.
    /// </summary>
    public sealed class BooleanDataViewType : PrimitiveDataViewType
    {
        private static volatile BooleanDataViewType _instance;
        /// <summary>
        /// The singleton instance of this type.
        /// </summary>
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
            return "Boolean";
        }
    }

    /// <summary>
    /// The standard date time type. This has representation type of <see cref="DateTime"/>.
    /// Note this can have only one possible value, accessible by the singleton static property <see cref="Instance"/>.
    /// </summary>
    public sealed class DateTimeDataViewType : PrimitiveDataViewType
    {
        private static volatile DateTimeDataViewType _instance;
        /// <summary>
        /// The singleton instance of this type.
        /// </summary>
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

    /// <summary>
    /// The standard date time offset type. This has representation type of <see cref="DateTimeOffset"/>.
    /// Note this can have only one possible value, accessible by the singleton static property <see cref="Instance"/>.
    /// </summary>
    public sealed class DateTimeOffsetDataViewType : PrimitiveDataViewType
    {
        private static volatile DateTimeOffsetDataViewType _instance;
        /// <summary>
        /// The singleton instance of this type.
        /// </summary>
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

        public override string ToString() => "DateTimeOffset";
    }

    /// <summary>
    /// The standard timespan type. This has representation type of <see cref="TimeSpan"/>.
    /// Note this can have only one possible value, accessible by the singleton static property <see cref="Instance"/>.
    /// </summary>
    public sealed class TimeSpanDataViewType : PrimitiveDataViewType
    {
        private static volatile TimeSpanDataViewType _instance;
        /// <summary>
        /// The singleton instance of this type.
        /// </summary>
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

    /// <summary>
    /// <see cref="DataViewTypeAttribute"/> should be used to decorated class properties and fields, if that class' instances will be loaded as ML.NET <see cref="IDataView"/>.
    /// The function <see cref="Register"/> will be called to register a <see cref="DataViewType"/> for a <see cref="Type"/> with its <see cref="Attribute"/>s.
    /// Whenever a value typed to the registered <see cref="Type"/> and its <see cref="Attribute"/>s, that value's type (i.e., a <see cref="DataViewSchema.Column.Type"/>)
    /// in <see cref="IDataView"/> would be the associated <see cref="DataViewType"/>.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public abstract class DataViewTypeAttribute : Attribute, IEquatable<DataViewTypeAttribute>
    {
        /// <summary>
        /// A function implicitly invoked by ML.NET when processing a custom type. It binds a DataViewType to a custom type plus its attributes.
        /// </summary>
        public abstract void Register();

        /// <summary>
        /// Return <see langword="true"/> if <see langword="this"/> is equivalent to <paramref name="other"/> and <see langword="false"/> otherwise.
        /// </summary>
        /// <param name="other">Another <see cref="DataViewTypeAttribute"/> to be compared with <see langword="this"/>.</param>
        public abstract bool Equals(DataViewTypeAttribute other);
    }
}
