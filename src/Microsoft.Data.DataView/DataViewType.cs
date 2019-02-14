// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.Reflection;
using System.Threading;

namespace Microsoft.Data.DataView
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
        /// For example, most practical instances of ML.NET's KeyType and <see cref="NumberDataViewType.U4"/> will have a
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

        private static volatile NumberDataViewType _instI1;
        public static NumberDataViewType I1
        {
            get
            {
                return _instI1 ??
                    Interlocked.CompareExchange(ref _instI1, new NumberDataViewType(typeof(sbyte), "I1"), null) ??
                    _instI1;
            }
        }

        private static volatile NumberDataViewType _instU1;
        public static NumberDataViewType U1
        {
            get
            {
                return _instU1 ??
                    Interlocked.CompareExchange(ref _instU1, new NumberDataViewType(typeof(byte), "U1"), null) ??
                    _instU1;
            }
        }

        private static volatile NumberDataViewType _instI2;
        public static NumberDataViewType I2
        {
            get
            {
                return _instI2 ??
                    Interlocked.CompareExchange(ref _instI2, new NumberDataViewType(typeof(short), "I2"), null) ??
                    _instI2;
            }
        }

        private static volatile NumberDataViewType _instU2;
        public static NumberDataViewType U2
        {
            get
            {
                return _instU2 ??
                    Interlocked.CompareExchange(ref _instU2, new NumberDataViewType(typeof(ushort), "U2"), null) ??
                    _instU2;
            }
        }

        private static volatile NumberDataViewType _instI4;
        public static NumberDataViewType I4
        {
            get
            {
                return _instI4 ??
                    Interlocked.CompareExchange(ref _instI4, new NumberDataViewType(typeof(int), "I4"), null) ??
                    _instI4;
            }
        }

        private static volatile NumberDataViewType _instU4;
        public static NumberDataViewType U4
        {
            get
            {
                return _instU4 ??
                    Interlocked.CompareExchange(ref _instU4, new NumberDataViewType(typeof(uint), "U4"), null) ??
                    _instU4;
            }
        }

        private static volatile NumberDataViewType _instI8;
        public static NumberDataViewType I8
        {
            get
            {
                return _instI8 ??
                    Interlocked.CompareExchange(ref _instI8, new NumberDataViewType(typeof(long), "I8"), null) ??
                    _instI8;
            }
        }

        private static volatile NumberDataViewType _instU8;
        public static NumberDataViewType U8
        {
            get
            {
                return _instU8 ??
                    Interlocked.CompareExchange(ref _instU8, new NumberDataViewType(typeof(ulong), "U8"), null) ??
                    _instU8;
            }
        }

        private static volatile NumberDataViewType _instUG;
        public static NumberDataViewType UG
        {
            get
            {
                return _instUG ??
                    Interlocked.CompareExchange(ref _instUG, new NumberDataViewType(typeof(DataViewRowId), "UG"), null) ??
                    _instUG;
            }
        }

        private static volatile NumberDataViewType _instR4;
        public static NumberDataViewType R4
        {
            get
            {
                return _instR4 ??
                    Interlocked.CompareExchange(ref _instR4, new NumberDataViewType(typeof(float), "R4"), null) ??
                    _instR4;
            }
        }

        private static volatile NumberDataViewType _instR8;
        public static NumberDataViewType R8
        {
            get
            {
                return _instR8 ??
                    Interlocked.CompareExchange(ref _instR8, new NumberDataViewType(typeof(double), "R8"), null) ??
                    _instR8;
            }
        }

        public static NumberDataViewType Float => R4;

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