// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using Microsoft.ML.Internal.DataView;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This type is for data representing some enumerated value. This is an enumeration over a defined, known
    /// cardinality set, as expressed through <see cref="Count"/>. The underlying .NET type is one of the unsigned
    /// integer types. Most commonly this will be <see cref="uint"/>, but could alternately be <see cref="byte"/>,
    /// <see cref="ushort"/>, or <see cref="ulong"/>. Despite this, the information is not inherently numeric, so,
    /// typically, arithmetic is not meaningful. For example, in multi-class classification, the label is typically a
    /// class number which is naturally a <see cref="KeyDataViewType"/>.
    ///
    /// Note that for data of this type, a value of 0, being the default value of the representation type, indicates
    /// the missing value since it would not be sensible for the default value to correspond to any one particular specific
    /// value of the set. The first non-missing value for the enumeration of the set is always <c>1</c>.
    ///
    /// For instance, if you had a key value with a <see cref="Count"/> of 3, then the <see cref="uint"/> value <c>0</c>
    /// would correspond to the missing key value, and one of the values of <c>1</c>, <c>2</c>, or <c>3</c> would be one
    /// of the valid values, and no other values should in principle be used.
    ///
    /// Note that in usage and structure, this is quite close in intended usage and structure to so-called "factor
    /// variables" in R.
    /// </summary>
    public sealed class KeyDataViewType : PrimitiveDataViewType
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="KeyDataViewType"/> class.
        /// </summary>
        /// <param name="type">
        /// The underlying representation type. Should be one of <see cref="byte"/>, <see cref="ushort"/>,
        /// <see cref="uint"/> (the most common choice), or <see cref="ulong"/>.
        /// </param>
        /// <param name="count">
        /// The cardinality of the underlying set. This must not exceed the associated maximum value of the
        /// representation type. For example, if <paramref name="type"/> is <see cref="uint"/>, then this must not
        /// exceed <see cref="uint.MaxValue"/>.
        /// </param>
        public KeyDataViewType(Type type, ulong count)
            : base(type)
        {
            if (!IsValidDataType(type))
                throw Contracts.ExceptParam(nameof(type), $"Type is not valid, it must be {typeof(byte).Name}, {typeof(ushort).Name}, {typeof(uint).Name}, or {typeof(ulong).Name}.");
            if (count == 0 || GetMaxInt(type) < count)
                throw Contracts.ExceptParam(nameof(count), $"The cardinality of a {nameof(KeyDataViewType)} must not exceed {type.Name}.{nameof(uint.MaxValue)} " +
                    $"and must be strictly positive, but got {count}.");
            Count = count;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="KeyDataViewType"/> class. This differs from the hypothetically more general
        /// <see cref="KeyDataViewType.KeyDataViewType(Type, ulong)"/> constructor by taking an <see cref="int"/> for
        /// <paramref name="count"/>, to more naturally facilitate the most common case that the key value is being used
        /// as an enumeration over an array or list of some form.
        /// </summary>
        /// <param name="type">
        /// The underlying representation type. Should be one of <see cref="byte"/>, <see cref="ushort"/>,
        /// <see cref="uint"/> (the most common choice), or <see cref="ulong"/>.
        /// </param>
        /// <param name="count">
        /// The cardinality of the underlying set. This must not exceed the associated maximum value of the
        /// representation type. For example, if <paramref name="type"/> is <see cref="uint"/>, then this must not
        /// exceed <see cref="uint.MaxValue"/>.
        /// </param>
        public KeyDataViewType(Type type, int count)
            : this(type, (ulong)count)
        {
            Contracts.CheckParam(0 < count, nameof(count), "The cardinality of a " + nameof(KeyDataViewType) + " must be strictly positive.");
        }

        /// <summary>
        /// Returns true iff the given type is valid for a <see cref="KeyDataViewType"/>. The valid ones are
        /// <see cref="byte"/>, <see cref="ushort"/>, <see cref="uint"/>, and <see cref="ulong"/>, that is, the unsigned
        /// integer types.
        /// </summary>
        public static bool IsValidDataType(Type type)
        {
            return type == typeof(uint) || type == typeof(ulong) || type == typeof(ushort) || type == typeof(byte);
        }

        private static ulong GetMaxInt(Type type)
        {
            if (type == typeof(uint))
                return uint.MaxValue;
            else if (type == typeof(ulong))
                return ulong.MaxValue;
            else if (type == typeof(ushort))
                return ushort.MaxValue;
            else if (type == typeof(byte))
                return byte.MaxValue;

            return 0;
        }

        private string GetRawTypeName()
        {
            if (RawType == typeof(uint))
                return NumberDataViewType.UInt32.ToString();
            else if (RawType == typeof(ulong))
                return NumberDataViewType.UInt64.ToString();
            else if (RawType == typeof(ushort))
                return NumberDataViewType.UInt16.ToString();
            else if (RawType == typeof(byte))
                return NumberDataViewType.Byte.ToString();

            Debug.Fail("Invalid type");
            return null;
        }

        /// <summary>
        /// <see cref="Count"/> is the cardinality of the <see cref="KeyDataViewType"/>.
        /// </summary>
        /// <remarks>
        /// The typical legal values for data of this type ranges from the missing value of <c>0</c>, and non-missing
        /// values ranging from to <c>1</c> through <see cref="Count"/>, inclusive, being the enumeration into whatever
        /// set the key values are enumerated over.
        /// </remarks>
        public ulong Count { get; }

        /// <summary>
        /// Determine if this <see cref="KeyDataViewType"/> object is equal to another <see cref="DataViewType"/> instance.
        /// Checks if the other item is the type of <see cref="KeyDataViewType"/>, if the <see cref="DataViewType.RawType"/>
        /// is the same, and if the <see cref="Count"/> is the same.
        /// </summary>
        /// <param name="other">The other object to compare against.</param>
        /// <returns><see langword="true" /> if both objects are equal, otherwise <see langword="false"/>.</returns>
        public override bool Equals(DataViewType other)
        {
            if (other == this)
                return true;

            if (!(other is KeyDataViewType tmp))
                return false;
            if (RawType != tmp.RawType)
                return false;
            if (Count != tmp.Count)
                return false;
            return true;
        }

        /// <summary>
        /// Determine if a <see cref="KeyDataViewType"/> instance is equal to another <see cref="KeyDataViewType"/> instance.
        /// Checks if any object is the type of <see cref="KeyDataViewType"/>, if the <see cref="DataViewType.RawType"/>
        /// is the same, and if the <see cref="Count"/> is the same.
        /// </summary>
        /// <param name="other">The other object to compare against.</param>
        /// <returns><see langword="true" /> if both objects are equal, otherwise <see langword="false"/>.</returns>
        public override bool Equals(object other)
            => other is DataViewType tmp && Equals(tmp);

        /// <summary>
        /// Retrieves the hash code.
        /// </summary>
        /// <returns>An integer representing the hash code.</returns>
        public override int GetHashCode()
        {
            return Hashing.CombineHash(RawType.GetHashCode(), Count.GetHashCode());
        }

        /// <summary>
        /// The string representation of the <see cref="KeyDataViewType"/>.
        /// </summary>
        /// <returns>A formatted string.</returns>
        public override string ToString()
        {
            string rawTypeName = GetRawTypeName();
            return string.Format("Key<{0}, {1}-{2}>", rawTypeName, 0, Count - 1);
        }
    }
}