// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using Microsoft.ML.Data;

namespace Microsoft.ML.Transforms.Onnx
{
    /// <summary>
    /// The corresponding <see cref="DataViewSchema.Column.Type"/> of ONNX's sequence type in <see cref="IDataView"/>'s type system.
    /// In other words, if an ONNX model produces a sequence, a column in <see cref="IDataView"/> may be typed to <see cref="OnnxSequenceType"/>.
    /// Its underlying type is <see cref="IEnumerable{T}"/>, where the generic type "T" is the input argument of
    /// <see cref="OnnxSequenceType.OnnxSequenceType(Type)"/>.
    /// </summary>
    public sealed class OnnxSequenceType : StructuredDataViewType
    {
        private static Type MakeNativeType(Type elementType)
        {
            var enumerableTypeInfo = typeof(IEnumerable<>);
            var enumerableType = enumerableTypeInfo.MakeGenericType(elementType);
            return enumerableType;
        }

        /// <summary>
        /// Create the corresponding <see cref="DataViewType"/> for ONNX sequence.
        /// </summary>
        /// <param name="elementType">The element type of a sequence.</param>
        public OnnxSequenceType(Type elementType) : base(MakeNativeType(elementType))
        {
            DataViewTypeManager.Register(this, RawType, new OnnxSequenceTypeAttribute(elementType));
        }

        public override bool Equals(DataViewType other)
        {
            if (other is OnnxSequenceType)
                return RawType == other.RawType;
            else
                return false;
        }

        public override int GetHashCode()
        {
            return RawType.GetHashCode();
        }
    }

    /// <summary>
    /// To declare <see cref="OnnxSequenceType"/> column in <see cref="IDataView"/> as a field
    /// in a <see langword="class"/>, the associated field should be marked with <see cref="OnnxSequenceTypeAttribute"/>.
    /// Its uses are similar to those of <see cref="VectorTypeAttribute"/> and other <see langword="class"/>es derived
    /// from <see cref="DataViewTypeAttribute"/>.
    /// </summary>
    public sealed class OnnxSequenceTypeAttribute : DataViewTypeAttribute
    {
        private readonly Type _elemType;

        // Make default constructor obsolete.
        // Use default constructor will left the _elemType field empty and cause exception in methods using _elemType.
        // User will receive compile warning when try to use [OnnxSequenceType] attribute directly without specify sequence type
        [Obsolete("Please specify sequence type when use OnnxSequenceType Attribute", false)]
        [EditorBrowsable(EditorBrowsableState.Never)]
        public OnnxSequenceTypeAttribute()
        {
        }

        /// <summary>
        /// Create a <paramref name="elemType"/>-sequence type.
        /// </summary>
        public OnnxSequenceTypeAttribute(Type elemType)
        {
            _elemType = elemType;
        }

        /// <summary>
        /// Sequence types with the same element type should be equal.
        /// </summary>
        public override bool Equals(DataViewTypeAttribute other)
        {
            if (other is OnnxSequenceTypeAttribute otherSequence)
                return _elemType.Equals(otherSequence._elemType);
            return false;
        }

        /// <summary>
        /// Produce the same hash code for sequence types with the same element type.
        /// </summary>
        public override int GetHashCode()
        {
            return _elemType.GetHashCode();
        }

        /// <summary>
        /// An implementation of <see cref="DataViewTypeAttribute.Register"/>.
        /// </summary>
        public override void Register()
        {
            var enumerableType = typeof(IEnumerable<>);
            var type = enumerableType.MakeGenericType(_elemType);
            DataViewTypeManager.Register(new OnnxSequenceType(_elemType), type, this);
        }
    }
}
