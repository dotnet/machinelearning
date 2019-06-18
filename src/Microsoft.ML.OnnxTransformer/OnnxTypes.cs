// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.Onnx
{
    internal class OnnxTypeHelper
    {
        public static Type GetNativeScalarType(OnnxCSharpToProtoWrapper.TensorProto.Types.DataType dataType)
        {
            Type scalarType = null;
            switch (dataType)
            {
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Bool:
                    scalarType = typeof(System.Boolean);
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int8:
                    scalarType = typeof(System.SByte);
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Uint8:
                    scalarType = typeof(System.Byte);
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int16:
                    scalarType = typeof(System.Int16);
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Uint16:
                    scalarType = typeof(System.UInt16);
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int32:
                    scalarType = typeof(System.Int32);
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Uint32:
                    scalarType = typeof(System.UInt32);
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int64:
                    scalarType = typeof(System.Int64);
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Uint64:
                    scalarType = typeof(System.UInt64);
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Double:
                    scalarType = typeof(System.Double);
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Float:
                    scalarType = typeof(System.Single);
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.String:
                    scalarType = typeof(string);
                    break;
                default:
                    throw Contracts.Except("Unsupported ONNX scalar type: " + dataType.ToString());
            }
            return scalarType;
        }

        public static Type GetNativeType(OnnxCSharpToProtoWrapper.TypeProto typeProto)
        {
            var oneOfFieldName = typeProto.ValueCase.ToString();
            if (oneOfFieldName == "TensorType")
            {
                if (typeProto.TensorType.Shape == null || typeProto.TensorType.Shape.Dim.Count == 0)
                {
                    return GetNativeScalarType(typeProto.TensorType.ElemType);
                }
                else
                {
                    Type tensorType = typeof(VBuffer<>);
                    Type elementType = GetNativeScalarType(typeProto.TensorType.ElemType);
                    return tensorType.MakeGenericType(elementType);
                }
            }
            else if (oneOfFieldName == "SequenceType")
            {
                var enumerableType = typeof(IEnumerable<>);
                var elementType = GetNativeType(typeProto.SequenceType.ElemType);
                return enumerableType.MakeGenericType(elementType);
            }
            else if (oneOfFieldName == "MapType")
            {
                var dictionaryType = typeof(IDictionary<,>);
                Type keyType = GetNativeScalarType(typeProto.MapType.KeyType);
                Type valueType = GetNativeType(typeProto.MapType.ValueType);
                return dictionaryType.MakeGenericType(keyType, valueType);
            }
            return null;
        }

        public static DataViewType GetScalarDataViewType(OnnxCSharpToProtoWrapper.TensorProto.Types.DataType dataType)
        {
            DataViewType scalarType = null;
            switch (dataType)
            {
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Bool:
                    scalarType = BooleanDataViewType.Instance;
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int8:
                    scalarType = NumberDataViewType.SByte;
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Uint8:
                    scalarType = NumberDataViewType.Byte;
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int16:
                    scalarType = NumberDataViewType.Int16;
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Uint16:
                    scalarType = NumberDataViewType.UInt16;
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int32:
                    scalarType = NumberDataViewType.Int32;
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Uint32:
                    scalarType = NumberDataViewType.UInt32;
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Int64:
                    scalarType = NumberDataViewType.Int64;
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Uint64:
                    scalarType = NumberDataViewType.UInt64;
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Float:
                    scalarType = NumberDataViewType.Single;
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.Double:
                    scalarType = NumberDataViewType.Double;
                    break;
                case OnnxCSharpToProtoWrapper.TensorProto.Types.DataType.String:
                    scalarType = TextDataViewType.Instance;
                    break;
                default:
                    throw Contracts.Except("Unsupported ONNX scalar type: " + dataType.ToString());
            }
            return scalarType;
        }

        public static int GetDimValue(OnnxCSharpToProtoWrapper.TensorShapeProto.Types.Dimension dim)
        {
            int value = 0;
            switch (dim.ValueCase)
            {
                case OnnxCSharpToProtoWrapper.TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue:
                    // The vector length in ML.NET is typed to 32-bit integer, so the check below is added for perverting overflowing.
                    if (dim.DimValue > int.MaxValue)
                        throw Contracts.ExceptParamValue(dim.DimValue, nameof(dim), $"Dimension {dim} in ONNX tensor cannot exceed the maximum of 32-bit signed integer.");
                    // Variable-length dimension is translated to 0.
                    value = dim.DimValue > 0 ? (int)dim.DimValue : 0;
                    break;
                case OnnxCSharpToProtoWrapper.TensorShapeProto.Types.Dimension.ValueOneofCase.DimParam:
                    // Variable-length dimension is translated to 0.
                    value = 0;
                    break;
                default:
                    throw Contracts.ExceptParamValue(dim.DimValue, nameof(dim), $"Dimension {dim} in ONNX tensor cannot exceed the maximum of 32-bit signed integer.");
            }
            return value;
        }

        public static IEnumerable<int> GetTensorDims(Microsoft.ML.Model.OnnxConverter.OnnxCSharpToProtoWrapper.TensorShapeProto tensorShapeProto)
        {
            if (tensorShapeProto == null)
                // Scalar has null dimensionality.
                return null;

            List<int> dims = new List<int>();
            foreach(var d in tensorShapeProto.Dim)
            {
                var dimValue = GetDimValue(d);
                dims.Add(dimValue);
            }
            return dims;
        }

        public static DataViewType GetDataViewType(OnnxCSharpToProtoWrapper.TypeProto typeProto)
        {
            var oneOfFieldName = typeProto.ValueCase.ToString();
            if (typeProto.ValueCase == OnnxCSharpToProtoWrapper.TypeProto.ValueOneofCase.TensorType)
            {
                if (typeProto.TensorType.Shape.Dim.Count == 0)
                    // ONNX scalar is a tensor without shape information; that is,
                    // ONNX scalar's shape is an empty list.
                    return GetScalarDataViewType(typeProto.TensorType.ElemType);
                else
                {
                    var shape = GetTensorDims(typeProto.TensorType.Shape);
                    if (shape == null)
                        // Scalar has null shape.
                        return GetScalarDataViewType(typeProto.TensorType.ElemType);
                    else if (shape.Count() != 0 && shape.Aggregate((x, y) => x * y) > 0)
                        // Known shape tensor.
                        return new VectorDataViewType((PrimitiveDataViewType)GetScalarDataViewType(typeProto.TensorType.ElemType), shape.ToArray());
                    else
                        // Tensor with unknown shape.
                        return new VectorDataViewType((PrimitiveDataViewType)GetScalarDataViewType(typeProto.TensorType.ElemType), 0);
                }
            }
            else if (typeProto.ValueCase == OnnxCSharpToProtoWrapper.TypeProto.ValueOneofCase.SequenceType)
            {
                var elemTypeProto = typeProto.SequenceType.ElemType;
                var elemType = GetNativeType(elemTypeProto);
                return new OnnxSequenceType(elemType);
            }
            else if (typeProto.ValueCase == OnnxCSharpToProtoWrapper.TypeProto.ValueOneofCase.MapType)
            {
                var keyType = GetNativeScalarType(typeProto.MapType.KeyType);
                var valueType = GetNativeType(typeProto.MapType.ValueType);
                return new OnnxMapType(keyType, valueType);
            }
            else
                throw Contracts.ExceptParamValue(typeProto, nameof(typeProto), $"Unsupported ONNX variable type {typeProto}");
        }
    }

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
            DataViewTypeManager.Register(this, RawType, new[] { new OnnxSequenceTypeAttribute(elementType) });
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

    public sealed class OnnxMapType : StructuredDataViewType
    {
        public OnnxMapType(Type keyType, Type valueType) : base(typeof(IDictionary<,>).MakeGenericType(keyType, valueType))
        {
            DataViewTypeManager.Register(this, RawType, new[] { new OnnxMapTypeAttribute(keyType, valueType) });
        }

        public override bool Equals(DataViewType other)
        {
            if (other is OnnxSequenceType)
                return RawType == other.RawType;
            else
                return false;
        }
    }

    public sealed class OnnxSequenceTypeAttribute : DataViewTypeAttribute
    {
        private Type _elemType;
        /// <summary>
        /// Create an image type without knowing its height and width.
        /// </summary>
        public OnnxSequenceTypeAttribute()
        {
        }

        /// <summary>
        /// Create an image type with known height and width.
        /// </summary>
        public OnnxSequenceTypeAttribute(Type elemType)
        {
            _elemType = elemType;
        }

        /// <summary>
        /// Images with the same width and height should equal.
        /// </summary>
        public override bool Equals(DataViewTypeAttribute other)
        {
            if (other is OnnxSequenceTypeAttribute otherSequence)
                return _elemType.Equals(otherSequence._elemType);
            return false;
        }

        /// <summary>
        /// Produce the same hash code for all images with the same height and the same width.
        /// </summary>
        public override int GetHashCode()
        {
            return _elemType.GetHashCode();
        }

        public override void Register()
        {
            var enumerableType = typeof(IEnumerable<>);
            var type = enumerableType.MakeGenericType(_elemType);
            DataViewTypeManager.Register(new OnnxSequenceType(_elemType), type, new[] { this });
        }
    }

    public sealed class OnnxMapTypeAttribute : DataViewTypeAttribute
    {
        private Type _keyType;
        private Type _valueType;
        /// <summary>
        /// Create an image type without knowing its height and width.
        /// </summary>
        public OnnxMapTypeAttribute()
        {
        }

        /// <summary>
        /// Create an image type with known height and width.
        /// </summary>
        public OnnxMapTypeAttribute(Type keyType, Type valueType)
        {
            _keyType = keyType;
            _valueType = valueType;
        }

        /// <summary>
        /// Images with the same width and height should equal.
        /// </summary>
        public override bool Equals(DataViewTypeAttribute other)
        {
            if (other is OnnxMapTypeAttribute otherSequence)
                return _keyType.Equals(otherSequence._keyType) && _valueType.Equals(otherSequence._valueType);
            return false;
        }

        /// <summary>
        /// Produce the same hash code for all images with the same height and the same width.
        /// </summary>
        public override int GetHashCode()
        {
            return Hashing.CombineHash(_keyType.GetHashCode(), _valueType.GetHashCode());
        }

        public override void Register()
        {
            var enumerableType = typeof(IDictionary<,>);
            var type = enumerableType.MakeGenericType(_keyType, _valueType);
            DataViewTypeManager.Register(new OnnxMapType(_keyType, _valueType), type, new[] { this });
        }
    }
}
