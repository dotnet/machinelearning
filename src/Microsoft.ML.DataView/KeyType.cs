// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// KeyTypes are for "id"-like data. The information happens to be stored in an unsigned integer
    /// type, but the information is not inherently numeric, so, typically, arithmetic is not
    /// meaningful. Examples are SSNs, phone numbers, auto-generated/incremented key values,
    /// class numbers, etc. For example, in multi-class classification, the label is typically
    /// a class number which is naturally a KeyType.
    ///
    /// KeyTypes have a cardinality (i.e., <see cref="Count"/>) that is strictly positive.
    ///
    /// Note that the underlying representation value does not necessarily match the logical value.
    /// For example, if a KeyType has range 0-5000, then it has a <see cref="Count"/> of 5001, but
    /// the representational values are 1-5001. The representation value zero is reserved
    /// to mean a missing value (similar to NaN).
    /// </summary>
    public sealed class KeyType : PrimitiveDataViewType
    {
        public KeyType(Type type, ulong count)
            : base(type)
        {
            if (!IsValidDataType(type))
                throw new ArgumentException(
                    $"Type is not valid, it must be {typeof(byte).FullName}, {typeof(ushort).FullName}, {typeof(uint).FullName}, or {typeof(ulong).FullName}.",
                    nameof(type));
            if (count == 0 || GetMaxInt(type) < count)
                throw new ArgumentOutOfRangeException(
                    nameof(count),
                    $"The cardinality of a {nameof(KeyType)} must not exceed {type.Name}.MaxValue and must be strictly positive but got {count}.");
            Count = count;
        }

        public KeyType(Type type, int count)
            : this(type, (ulong)count)
        {
            if (count <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(count),
                    $"The cardinality of a {nameof(KeyType)} must be strictly positive.");
        }

        /// <summary>
        /// Returns true iff the given type is valid for a <see cref="KeyType"/>. The valid ones are
        /// <see cref="byte"/>, <see cref="ushort"/>, <see cref="uint"/>, and <see cref="ulong"/>, that is, the unsigned integer types.
        /// </summary>
        public static bool IsValidDataType(Type type)
        {
            return type == typeof(byte) || type == typeof(ushort) || type == typeof(uint) || type == typeof(ulong);
        }

        private static ulong GetMaxInt(Type type)
        {
            if (type == typeof(byte))
                return byte.MaxValue;
            else if (type == typeof(ushort))
                return ushort.MaxValue;
            else if (type == typeof(uint))
                return uint.MaxValue;
            else if (type == typeof(ulong))
                return ulong.MaxValue;

            return 0;
        }

        /// <summary>
        /// <see cref="Count"/> is the cardinality of the <see cref="KeyType"/>. Note that such a key type can be converted to a
        /// bit vector representation by mapping to a vector of length <see cref="Count"/>, with "id" mapped to a
        /// vector with 1 in slot (id - 1) and 0 in all other slots. This is the standard "indicator"
        /// representation. Note that an id of 0 is used to represent the notion "none", which is
        /// typically mapped, by for example, one-hot encoding, to a vector of all zeros (of length <see cref="Count"/>).
        /// </summary>
        public ulong Count { get; }

        /// <summary>
        /// Determine if this <see cref="KeyType"/> object is equal to another <see cref="DataViewType"/> instance.
        /// Checks if the other item is the type of <see cref="KeyType"/>, if the <see cref="DataViewType.RawType"/>
        /// is the same, and if the <see cref="Count"/> is the same.
        /// </summary>
        /// <param name="other">The other object to compare against.</param>
        /// <returns><see langword="true" /> if both objects are equal, otherwise <see langword="false"/>.</returns>
        public override bool Equals(DataViewType other)
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

        /// <summary>
        /// Determine if a <see cref="KeyType"/> instance is equal to another <see cref="KeyType"/> instance.
        /// Checks if any object is the type of <see cref="KeyType"/>, if the <see cref="DataViewType.RawType"/>
        /// is the same, and if the <see cref="Count"/> is the same.
        /// </summary>
        /// <param name="other">The other object to compare against.</param>
        /// <returns><see langword="true" /> if both objects are equal, otherwise <see langword="false"/>.</returns>
        public override bool Equals(object other)
        {
            return other is DataViewType tmp && Equals(tmp);
        }

        /// <summary>
        /// Retrieves the hash code.
        /// </summary>
        /// <returns>An integer representing the hash code.</returns>
        public override int GetHashCode()
        {
            return Hashing.CombineHash(RawType.GetHashCode(), Count.GetHashCode());
        }

        // taken from https://github.com/dotnet/machinelearning/blob/08318656ed0f0649aa75370b019cf4bcbda5a6a5/src/Microsoft.ML.Core/Utilities/Hashing.cs#L17-L25
        private static class Hashing
        {
            public static uint CombineHash(uint u1, uint u2)
            {
                return ((u1 << 7) | (u1 >> 25)) ^ u2;
            }

            public static int CombineHash(int n1, int n2)
            {
                return (int)CombineHash((uint)n1, (uint)n2);
            }
        }

        /// <summary>
        /// The string representation of the <see cref="KeyType"/>.
        /// </summary>
        /// <returns>A formatted string.</returns>
        public override string ToString()
        {
            string rawTypeName = RawType.Name;
            return string.Format("Key<{0}, {1}-{2}>", rawTypeName, 0, Count - 1);
        }
    }
}