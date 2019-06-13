// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Runtime;
using OnnxShape = System.Collections.Generic.List<int>;

namespace Microsoft.ML.Transforms.Onnx
{
    /// <summary>
    /// OnnxModel is a utility class to load ONNX models and retrieve metadata
    /// for inputs and outputs. The metadata includes the names, shapes and types
    /// It provides API to open a session, score tensors (NamedOnnxValues) and return
    /// the results.
    /// </summary>
    internal sealed class OnnxModel
    {
        private static Type GetScalarType(OnnxCSharpToProtoWrapper.TensorProto.Types.DataType dataType)
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

        private static Type GetNativeType(OnnxCSharpToProtoWrapper.TypeProto typeProto)
        {
            var oneOfFieldName = typeProto.ValueCase.ToString();
            if (oneOfFieldName == "TensorType")
            {
                if (typeProto.TensorType.Shape == null || typeProto.TensorType.Shape.Dim.Count == 0)
                {
                    return GetScalarType(typeProto.TensorType.ElemType);
                }
                else
                {
                    Type tensorType = typeof(VBuffer<>);
                    Type elementType = GetScalarType(typeProto.TensorType.ElemType);
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
                Type keyType = GetScalarType(typeProto.MapType.KeyType);
                Type valueType = GetNativeType(typeProto.MapType.ValueType);
                return dictionaryType.MakeGenericType(keyType, valueType);
            }
            return null;
        }

        private static DataViewType GetScalarDataViewType(OnnxCSharpToProtoWrapper.TensorProto.Types.DataType dataType)
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

        private static IEnumerable<int> GetTensorDims(Microsoft.ML.Model.OnnxConverter.OnnxCSharpToProtoWrapper.TensorShapeProto tensorShapeProto)
        {
            var dims = new List<int>();
            if (tensorShapeProto == null)
                return dims;
            foreach(var d in tensorShapeProto.Dim)
            {
                switch (d.ValueCase)
                {
                    case OnnxCSharpToProtoWrapper.TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue:
                        if (d.DimValue <= 0)
                            return new List<int>();
                        dims.Add((int)d.DimValue);
                        break;
                    case OnnxCSharpToProtoWrapper.TensorShapeProto.Types.Dimension.ValueOneofCase.DimParam:
                        return new List<int>();
                }
            }
            return dims;
        }

        private static DataViewType GetDataViewType(OnnxCSharpToProtoWrapper.TypeProto typeProto)
        {
            var oneOfFieldName = typeProto.ValueCase.ToString();
            if (typeProto.ValueCase == OnnxCSharpToProtoWrapper.TypeProto.ValueOneofCase.TensorType)
            {
                if (typeProto.TensorType.Shape.Dim.Count == 0)
                    return GetScalarDataViewType(typeProto.TensorType.ElemType);
                else
                {
                    var shape = GetTensorDims(typeProto.TensorType.Shape).ToArray();
                    if (shape.Length > 0)
                        return new VectorDataViewType((PrimitiveDataViewType)GetScalarDataViewType(typeProto.TensorType.ElemType), shape);
                    else
                        return new VectorDataViewType((PrimitiveDataViewType)GetScalarDataViewType(typeProto.TensorType.ElemType), 0);
                }
            }
            else if (typeProto.ValueCase == OnnxCSharpToProtoWrapper.TypeProto.ValueOneofCase.SequenceType)
            {
                if (typeProto.SequenceType.ElemType.ValueCase != OnnxCSharpToProtoWrapper.TypeProto.ValueOneofCase.MapType)
                    throw new NotImplementedException($"Element type {typeProto.SequenceType.ElemType} is not allowed.");
                var mapType = typeProto.SequenceType.ElemType.MapType;
                var keyType = GetScalarType(mapType.KeyType);
                var valueType = GetNativeType(mapType.ValueType);
                return new OnnxSequenceMapType(keyType, valueType);
            }
            else if (typeProto.ValueCase == OnnxCSharpToProtoWrapper.TypeProto.ValueOneofCase.MapType)
            {
                var dictionaryType = typeof(IDictionary<,>);
                Type keyType = GetScalarType(typeProto.MapType.KeyType);
                Type valueType = GetNativeType(typeProto.MapType.ValueType);
                return new OnnxDictionaryType(keyType, valueType);
            }
            return null;
        }

        public static class OnnxCaster
        {
            public static T GetValue<T>(OnnxSequenceMapType dataViewType, NamedOnnxValue namedOnnxValue)
            {
                var dictionaryMethodInfo = typeof(NamedOnnxValue).GetMethod(nameof(NamedOnnxValue.AsDictionary));
                var dictionaryMethod = dictionaryMethodInfo.MakeGenericMethod(dataViewType.KeyType, dataViewType.ValueType);
                var enumerable = namedOnnxValue.AsEnumerable<NamedOnnxValue>().Select(value => (T)dictionaryMethod.Invoke(value, null));
                return default;
            }
        }

        public sealed class OnnxSequenceType : StructuredDataViewType
        {
            private static Type MakeNativeType(Type elementType)
            {
                var enumerableTypeInfo = typeof(IEnumerable<>);
                var enumerableType = enumerableTypeInfo.MakeGenericType(elementType);
                return enumerableType;
            }

            public OnnxSequenceType(Type elementType) : base(MakeNativeType(elementType))
            {
                DataViewTypeManager.Register(this, RawType);
            }

            public override bool Equals(DataViewType other)
            {
                if (other is OnnxSequenceType)
                    return RawType == other.RawType;
                else
                    return false;
            }
        }

        public sealed class OnnxSequenceMapType : StructuredDataViewType
        {
            private static Type MakeNativeType(Type keyType, Type valueType)
            {
                var enumerableTypeInfo = typeof(IEnumerable<>);
                var dictionaryTypeInfo = typeof(IDictionary<,>);

                var dictionaryType = dictionaryTypeInfo.MakeGenericType(keyType, valueType);
                var enumerableType = enumerableTypeInfo.MakeGenericType(dictionaryType);

                return enumerableType;
            }

            public Type KeyType { get; }
            public Type ValueType { get; }

            public OnnxSequenceMapType(Type keyType, Type valueType) : base(MakeNativeType(keyType, valueType))
            {
                KeyType = keyType;
                ValueType = valueType;
                DataViewTypeManager.Register(this, RawType);
            }

            public override bool Equals(DataViewType other)
            {
                return RawType == other.RawType;
            }
        }

        public sealed class OnnxDictionaryType : StructuredDataViewType
        {
            public OnnxDictionaryType(Type keyType, Type elementType) : base(typeof(IDictionary<,>).MakeGenericType(keyType, elementType))
            {
                DataViewTypeManager.Register(this, RawType);
            }

            public override bool Equals(DataViewType other)
            {
                return RawType == other.RawType;
            }
        }

        /// <summary>
        /// OnnxModelInfo contains the data that we should get from
        /// OnnxRuntime API once that functionality is added.
        /// </summary>
        public sealed class OnnxModelInfo
        {
            public readonly OnnxNodeInfo[] InputsInfo;
            public readonly OnnxNodeInfo[] OutputsInfo;

            public OnnxModelInfo(IEnumerable<OnnxNodeInfo> inputsInfo, IEnumerable<OnnxNodeInfo> outputsInfo)
            {
                InputsInfo = inputsInfo.ToArray();
                OutputsInfo = outputsInfo.ToArray();
            }
        }

        /// <summary>
        /// OnnxNodeInfo contains all the information for a given node (e.g. inputs/outputs)
        /// of an Onnx model.
        /// </summary>
        public class OnnxNodeInfo
        {
            /// <summary>
            /// The Name of the node
            /// </summary>
            public readonly string Name;
            /// <summary>
            /// The shape of the node
            /// </summary>
            public readonly OnnxShape Shape;
            /// <summary>
            /// The type of the node
            /// </summary>
            public readonly System.Type Type;

            public OnnxNodeInfo(string name, OnnxShape shape, System.Type type)
            {
                Name = name;
                Shape = shape;
                Type = type;
            }
        }

        public readonly OnnxModelInfo ModelInfo;
        private readonly InferenceSession _session;
        private readonly string _modelFile;
        public readonly List<string> InputNames;
        public readonly List<string> OutputNames;
        public readonly List<DataViewType> InputTypes;
        public readonly List<DataViewType> OutputTypes;

        /// <summary>
        /// Constructs OnnxModel object from file.
        /// </summary>
        /// <param name="modelFile">Model file path.</param>
        /// <param name="gpuDeviceId">GPU device ID to execute on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If true, resumes CPU execution quitely upon GPU error.</param>
        public OnnxModel(string modelFile, int? gpuDeviceId = null, bool fallbackToCpu = false)
        {
            if (gpuDeviceId != null)
            {
                try
                {
                    _session = new InferenceSession(modelFile,
                        SessionOptions.MakeSessionOptionWithCudaProvider(gpuDeviceId.Value));
                }
                catch (OnnxRuntimeException)
                {
                    if (fallbackToCpu)
                        _session = new InferenceSession(modelFile);
                    else
                        // if called from OnnxTranform, is caught and rethrown.
                        throw;
                }
            }
            else
            {
                _session = new InferenceSession(modelFile);
            }

            _modelFile = modelFile;
            var model = new OnnxCSharpToProtoWrapper.ModelProto();
            using (var modelStream = File.OpenRead(modelFile))
                model = OnnxCSharpToProtoWrapper.ModelProto.Parser.ParseFrom(modelStream);
            InputTypes = model.Graph.Input.Select(valueInfo => GetDataViewType(valueInfo.Type)).ToList();
            OutputTypes = model.Graph.Output.Select(valueInfo => GetDataViewType(valueInfo.Type)).ToList();

            ModelInfo = new OnnxModelInfo(GetInputsInfo(), GetOutputsInfo());
            InputNames = ModelInfo.InputsInfo.Select(i => i.Name).ToList();
            OutputNames = ModelInfo.OutputsInfo.Select(i => i.Name).ToList();
        }

        /// <summary>
        /// Create an OnnxModel from a byte[]
        /// </summary>
        /// <param name="modelBytes">Bytes of the serialized model</param>
        /// <returns>OnnxModel</returns>
        public static OnnxModel CreateFromBytes(byte[] modelBytes)
        {
            return CreateFromBytes(modelBytes, null, false);
        }

        /// <summary>
        /// Create an OnnxModel from a byte[]. Set execution to GPU if required.
        /// </summary>
        /// <param name="modelBytes">Bytes of the serialized model.</param>
        /// <param name="gpuDeviceId">GPU device ID to execute on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If true, resumes CPU execution quitely upon GPU error.</param>
        /// <returns>OnnxModel</returns>
        public static OnnxModel CreateFromBytes(byte[] modelBytes, int? gpuDeviceId = null, bool fallbackToCpu = false)
        {
            var tempModelDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
            Directory.CreateDirectory(tempModelDir);

            var tempModelFile = Path.Combine(tempModelDir, "model.onnx");
            File.WriteAllBytes(tempModelFile, modelBytes);
            return new OnnxModel(tempModelFile, gpuDeviceId, fallbackToCpu);

            // TODO:
            // tempModelFile is needed in case the model needs to be saved
            // Either have to save the modelbytes and delete the temp dir/file,
            // or keep the dir/file and write proper cleanup when application closes
        }

        /// <summary>
        /// Uses an open session to score a list of NamedOnnxValues.
        /// </summary>
        /// <param name="inputNamedOnnxValues">The NamedOnnxValues to score.</param>
        /// <returns>Resulting output NamedOnnxValues list.</returns>
        public IReadOnlyCollection<NamedOnnxValue> Run(List<NamedOnnxValue> inputNamedOnnxValues)
        {
            return _session.Run(inputNamedOnnxValues);
        }

        /// <summary>
        /// Convert the model to a byte array.
        /// </summary>
        /// <returns>byte[]</returns>
        public byte[] ToByteArray()
        {
            return File.ReadAllBytes(_modelFile);
        }

        /// <summary>
        /// Returns input metadata of the ONNX model.
        /// </summary>
        /// <returns>OnnxNodeInfo[]</returns>
        private IEnumerable<OnnxNodeInfo> GetInputsInfo()
        {
            return _session.InputMetadata.Select(kv => new OnnxNodeInfo(kv.Key, kv.Value.Dimensions.ToList(), kv.Value.ElementType));
        }

        /// <summary>
        /// Returns output metadata of the ONNX model.
        /// </summary>
        /// <returns></returns>
        private IEnumerable<OnnxNodeInfo> GetOutputsInfo()
        {
            return _session.OutputMetadata.Select(kv => new OnnxNodeInfo(kv.Key, kv.Value.Dimensions.ToList(), kv.Value.ElementType));
        }
    }

    internal sealed class OnnxUtils
    {
        private static HashSet<System.Type> _onnxTypeMap =
            new HashSet<System.Type>
                {
                     typeof(Double),
                     typeof(Single),
                     typeof(Int16),
                     typeof(Int32),
                     typeof(Int64),
                     typeof(UInt16),
                     typeof(UInt32),
                     typeof(UInt64)
                };
        private static Dictionary<System.Type, InternalDataKind> _typeToKindMap=
            new Dictionary<System.Type, InternalDataKind>
                {
                    { typeof(Single) , InternalDataKind.R4},
                    { typeof(Double) , InternalDataKind.R8},
                    { typeof(Int16) , InternalDataKind.I2},
                    { typeof(Int32) , InternalDataKind.I4},
                    { typeof(Int64) , InternalDataKind.I8},
                    { typeof(UInt16) , InternalDataKind.U2},
                    { typeof(UInt32) , InternalDataKind.U4},
                    { typeof(UInt64) , InternalDataKind.U8},
                    { typeof(String) , InternalDataKind.TX},
                    { typeof(Boolean) , InternalDataKind.BL},
                };

        /// <summary>
        /// Creates a NamedOnnxValue from a scalar value.
        /// </summary>
        /// <typeparam name="T">The type of the Tensor contained in the NamedOnnxValue.</typeparam>
        /// <param name="name">The name of the NamedOnnxValue.</param>
        /// <param name="data">The data values of the Tensor.</param>
        /// <returns>NamedOnnxValue</returns>
        public static NamedOnnxValue CreateScalarNamedOnnxValue<T>(string name, T data)
        {
            if (!_onnxTypeMap.Contains(typeof(T)))
                throw new NotImplementedException($"Not implemented type {typeof(T)}");
            return NamedOnnxValue.CreateFromTensor<T>(name, new DenseTensor<T>(new T[] { data }, new int[] { 1 }));
        }

        /// <summary>
        /// Create a NamedOnnxValue from vbuffer span. Checks if the tensor type
        /// is supported by OnnxRuntime prior to execution.
        /// </summary>
        /// <typeparam name="T">The type of the Tensor contained in the NamedOnnxValue.</typeparam>
        /// <param name="name">The name of the NamedOnnxValue.</param>
        /// <param name="data">A span containing the data</param>
        /// <param name="shape">The shape of the Tensor being created.</param>
        /// <returns>NamedOnnxValue</returns>
        public static NamedOnnxValue CreateNamedOnnxValue<T>(string name, ReadOnlySpan<T> data, OnnxShape shape)
        {
            if (!_onnxTypeMap.Contains(typeof(T)))
                throw new NotImplementedException($"Not implemented type {typeof(T)}");
            return NamedOnnxValue.CreateFromTensor<T>(name, new DenseTensor<T>(data.ToArray(), shape.Select(x => (int)x).ToArray()));
        }

        /// <summary>
        /// Converts a Onnx type, that follows the System.Type convention
        /// to the type system ML.NET recognizes (e.g. I4, I8, R4 etc.)
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static PrimitiveDataViewType OnnxToMlNetType(System.Type type)
        {
            if (!_typeToKindMap.ContainsKey(type))
               throw Contracts.ExceptNotSupp("Onnx type not supported", type);
            return ColumnTypeExtensions.PrimitiveTypeFromKind(_typeToKindMap[type]);
        }
    }
}
