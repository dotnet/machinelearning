// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Transforms.Onnx
{
    /// <summary>
    /// The corresponding <see cref="DataViewSchema.Column.Type"/> of ONNX's map type in <see cref="IDataView"/>'s type system.
    /// In other words, if an ONNX model produces a map, a column in <see cref="IDataView"/> may be typed to <see cref="OnnxMapType"/>.
    /// Its underlying type is <see cref="IDictionary{TKey, TValue}"/>, where the generic type "TKey" and "TValue" are the input arguments of
    /// <see cref="OnnxMapType.OnnxMapType(Type,Type)"/>.
    /// </summary>
    public sealed class OnnxMapType : StructuredDataViewType
    {
        /// <summary>
        /// Create the corresponding <see cref="DataViewType"/> for ONNX map.
        /// </summary>
        /// <param name="keyType">Key type of the associated ONNX map.</param>
        /// <param name="valueType">Value type of the associated ONNX map.</param>
        public OnnxMapType(Type keyType, Type valueType) : base(typeof(IDictionary<,>).MakeGenericType(keyType, valueType))
        {
            DataViewTypeManager.Register(this, RawType, new OnnxMapTypeAttribute(keyType, valueType));
        }

        public override bool Equals(DataViewType other)
        {
            if (other is OnnxMapType)
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
    /// To declare <see cref="OnnxMapType"/> column in <see cref="IDataView"/> as a field
    /// in a <see langword="class"/>, the associated field should be marked with <see cref="OnnxMapTypeAttribute"/>.
    /// Its uses are similar to those of <see cref="VectorTypeAttribute"/> and other <see langword="class"/>es derived
    /// from <see cref="DataViewTypeAttribute"/>.
    /// </summary>
    public sealed class OnnxMapTypeAttribute : DataViewTypeAttribute
    {
        private readonly Type _keyType;
        private readonly Type _valueType;

        /// <summary>
        /// Create a map (aka dictionary) type.
        /// </summary>
        public OnnxMapTypeAttribute()
        {
        }

        /// <summary>
        /// Create a map (aka dictionary) type. A map is a collection of key-value
        /// pairs. <paramref name="keyType"/> specifies the type of keys and <paramref name="valueType"/>
        /// is the type of values.
        /// </summary>
        public OnnxMapTypeAttribute(Type keyType, Type valueType)
        {
            _keyType = keyType;
            _valueType = valueType;
        }

        /// <summary>
        /// Map types with the same key type and the same value type should be equal.
        /// </summary>
        public override bool Equals(DataViewTypeAttribute other)
        {
            if (other is OnnxMapTypeAttribute otherSequence)
                return _keyType.Equals(otherSequence._keyType) && _valueType.Equals(otherSequence._valueType);
            return false;
        }

        /// <summary>
        /// Produce the same hash code for map types with the same key type and the same value type.
        /// </summary>
        public override int GetHashCode()
        {
            return Hashing.CombineHash(_keyType.GetHashCode(), _valueType.GetHashCode());
        }

        /// <summary>
        /// An implementation of <see cref="DataViewTypeAttribute.Register"/>.
        /// </summary>
        public override void Register()
        {
            var enumerableType = typeof(IDictionary<,>);
            var type = enumerableType.MakeGenericType(_keyType, _valueType);
            DataViewTypeManager.Register(new OnnxMapType(_keyType, _valueType), type, this);
        }
    }
}
