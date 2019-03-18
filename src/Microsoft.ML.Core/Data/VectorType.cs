// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// The standard vector type.
    /// </summary>
    public sealed class VectorType : StructuredDataViewType
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
        public VectorType(PrimitiveDataViewType itemType, int size = 0)
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
        public VectorType(PrimitiveDataViewType itemType, params int[] dimensions)
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
        internal VectorType(PrimitiveDataViewType itemType, VectorType template)
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
        internal VectorType(PrimitiveDataViewType itemType, VectorType template, params int[] dims)
            : base(GetRawType(itemType))
        {
            Contracts.CheckValue(template, nameof(template));
            Contracts.CheckParam(Utils.Size(dims) > 0, nameof(dims));
            Contracts.CheckParam(dims.All(d => d >= 0), nameof(dims));

            ItemType = itemType;
            Dimensions = template.Dimensions.AddRange(dims);
            Size = ComputeSize(Dimensions);
        }

        private static Type GetRawType(PrimitiveDataViewType itemType)
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
        public PrimitiveDataViewType ItemType { get; }

        /// <summary>
        /// The size of the vector. A value of zero means it is a vector whose size is unknown.
        /// A vector whose size is known should correspond to values that always have the same <see cref="VBuffer{T}.Length"/>,
        /// whereas one whose size is unknown may have values whose <see cref="VBuffer{T}.Length"/> varies from record to record.
        /// Note that this is always the product of the elements in <see cref="Dimensions"/>.
        /// </summary>
        public int Size { get; }

        public override bool Equals(DataViewType other)
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
            return other is DataViewType tmp && Equals(tmp);
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