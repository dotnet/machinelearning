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
    public abstract class ColumnType : IEquatable<ColumnType>
    {
        /// <summary>
        /// Constructor for extension types, which must be either <see cref="PrimitiveType"/> or <see cref="StructuredType"/>.
        /// </summary>
        private protected ColumnType(Type rawType)
        {
            RawType = rawType ?? throw new ArgumentNullException(nameof(rawType));
        }

        /// <summary>
        /// The raw <see cref="Type"/> for this <see cref="ColumnType"/>. Note that this is the raw representation type
        /// and not the complete information content of the <see cref="ColumnType"/>.
        /// </summary>
        /// <remarks>
        /// Code should not assume that a <see cref="RawType"/> uniquely identifiers a <see cref="ColumnType"/>.
        /// For example, most practical instances of ML.NET's KeyType and <see cref="NumberType.U4"/> will have a
        /// <see cref="RawType"/> of <see cref="uint"/>, but both are very different in the types of information conveyed in that number.
        /// </remarks>
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
            if (typeof(IDisposable).GetTypeInfo().IsAssignableFrom(RawType.GetTypeInfo()))
                throw new ArgumentException("A " + nameof(PrimitiveType) + " cannot have a disposable " + nameof(RawType), nameof(rawType));
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
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new TextType(), null) ??
                    _instance;
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
            Debug.Assert(!(other is TextType));
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
            Debug.Assert(!string.IsNullOrEmpty(name));
            _name = name;
        }

        private static volatile NumberType _instI1;
        public static NumberType I1
        {
            get
            {
                return _instI1 ??
                    Interlocked.CompareExchange(ref _instI1, new NumberType(typeof(sbyte), "I1"), null) ??
                    _instI1;
            }
        }

        private static volatile NumberType _instU1;
        public static NumberType U1
        {
            get
            {
                return _instU1 ??
                    Interlocked.CompareExchange(ref _instU1, new NumberType(typeof(byte), "U1"), null) ??
                    _instU1;
            }
        }

        private static volatile NumberType _instI2;
        public static NumberType I2
        {
            get
            {
                return _instI2 ??
                    Interlocked.CompareExchange(ref _instI2, new NumberType(typeof(short), "I2"), null) ??
                    _instI2;
            }
        }

        private static volatile NumberType _instU2;
        public static NumberType U2
        {
            get
            {
                return _instU2 ??
                    Interlocked.CompareExchange(ref _instU2, new NumberType(typeof(ushort), "U2"), null) ??
                    _instU2;
            }
        }

        private static volatile NumberType _instI4;
        public static NumberType I4
        {
            get
            {
                return _instI4 ??
                    Interlocked.CompareExchange(ref _instI4, new NumberType(typeof(int), "I4"), null) ??
                    _instI4;
            }
        }

        private static volatile NumberType _instU4;
        public static NumberType U4
        {
            get
            {
                return _instU4 ??
                    Interlocked.CompareExchange(ref _instU4, new NumberType(typeof(uint), "U4"), null) ??
                    _instU4;
            }
        }

        private static volatile NumberType _instI8;
        public static NumberType I8
        {
            get
            {
                return _instI8 ??
                    Interlocked.CompareExchange(ref _instI8, new NumberType(typeof(long), "I8"), null) ??
                    _instI8;
            }
        }

        private static volatile NumberType _instU8;
        public static NumberType U8
        {
            get
            {
                return _instU8 ??
                    Interlocked.CompareExchange(ref _instU8, new NumberType(typeof(ulong), "U8"), null) ??
                    _instU8;
            }
        }

        private static volatile NumberType _instUG;
        public static NumberType UG
        {
            get
            {
                return _instUG ??
                    Interlocked.CompareExchange(ref _instUG, new NumberType(typeof(RowId), "UG"), null) ??
                    _instUG;
            }
        }

        private static volatile NumberType _instR4;
        public static NumberType R4
        {
            get
            {
                return _instR4 ??
                    Interlocked.CompareExchange(ref _instR4, new NumberType(typeof(float), "R4"), null) ??
                    _instR4;
            }
        }

        private static volatile NumberType _instR8;
        public static NumberType R8
        {
            get
            {
                return _instR8 ??
                    Interlocked.CompareExchange(ref _instR8, new NumberType(typeof(double), "R8"), null) ??
                    _instR8;
            }
        }

        public static NumberType Float => R4;

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            Debug.Assert(other == null || !(other is NumberType) || other.RawType != RawType);
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
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new BoolType(), null) ??
                    _instance;
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
            Debug.Assert(!(other is BoolType));
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
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new DateTimeType(), null) ??
                    _instance;
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
            Debug.Assert(!(other is DateTimeType));
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
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new DateTimeOffsetType(), null) ??
                    _instance;
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
            Debug.Assert(!(other is DateTimeOffsetType));
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
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new TimeSpanType(), null) ??
                    _instance;
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
            Debug.Assert(!(other is TimeSpanType));
            return false;
        }

        public override string ToString() => "TimeSpan";
    }
}