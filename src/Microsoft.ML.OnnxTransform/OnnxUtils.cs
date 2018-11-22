// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Scoring;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.StaticPipe;

using OnnxShape = System.Collections.Generic.List<long>;
using Microsoft.ML.Data;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// OnnxModel is a facad for ModelManager. ModelManager is provided by Sonoma API,
    /// and it has a lot of functionality (multiple models, multiple versions) that are not
    /// needed by Onnx transform, which only needs a single model. This facad simplifies the
    /// usage of onnx model.
    /// </summary>
    internal sealed class OnnxModel
    {
        /// <summary>
        /// OnnxModelInfo contains the data that we should get from
        /// Sonoma API once that functionality is added.
        /// </summary>
        public sealed class OnnxModelInfo
        {
            public readonly OnnxNodeInfo[] InputsInfo;
            public readonly OnnxNodeInfo[] OutputsInfo;

            public OnnxModelInfo(OnnxNodeInfo[] inputsInfo, OnnxNodeInfo[] outputsInfo)
            {
                InputsInfo = inputsInfo;
                OutputsInfo = outputsInfo;
            }
        }

        /// <summary>
        /// OnnxNodeInfo contains all the information for a given node (e.g. inputs/outputs)
        /// of an Onnx model.
        /// </summary>
        public class OnnxNodeInfo
        {
            public readonly string Name;
            public readonly OnnxShape Shape;
            public readonly DataType Type;

            public OnnxNodeInfo(string name, OnnxShape shape, DataType type)
            {
                Name = name;
                Shape = shape;
                Type = type;
            }
        }

        public readonly OnnxModelInfo ModelInfo;

        private static readonly int _ignoredVersion = int.MaxValue;
        private readonly ModelManager _modelManager;
        private readonly string _modelFile;
        private readonly string _modelName;
        public readonly List<string> InputNames;
        public readonly List<string> OutputNames;

        public OnnxModel(string modelFile)
        {
            _modelFile = modelFile;

            // Load the onnx model
            var modelFileInfo = new FileInfo(modelFile);
            _modelName = Path.GetFileNameWithoutExtension(modelFileInfo.Name);
            _modelManager = new ModelManager(modelFileInfo.Directory.FullName, true);
            _modelManager.InitOnnxModel(_modelName, _ignoredVersion);

            ModelInfo = new OnnxModelInfo(GetInputsInfo(), GetOutputsInfo());
            InputNames = ModelInfo.InputsInfo.Select(i => i.Name).ToList();
            OutputNames = ModelInfo.OutputsInfo.Select(i => i.Name).ToList();
        }

        public static OnnxModel CreateFromBytes(byte[] modelBytes)
        {
            var tempModelDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
            Directory.CreateDirectory(tempModelDir);

            var tempModelFile = Path.Combine(tempModelDir, "model.onnx");
            File.WriteAllBytes(tempModelFile, modelBytes);
            return new OnnxModel(tempModelFile);

            // TODO:
            // tempModelFile is needed in case the model needs to be saved
            // Either have to save the modelbytes and delete the temp dir/file,
            // or keep the dir/file and write proper cleanup when application closes
        }

        public List<Tensor> Run(List<Tensor> inputTensors)
        {
            var outputTensors = _modelManager.RunModel(
                _modelName, _ignoredVersion, InputNames, inputTensors, OutputNames);

            return outputTensors;
        }

        public byte[] ToByteArray()
        {
            return File.ReadAllBytes(_modelFile);
        }

        private OnnxNodeInfo[] GetInputsInfo()
        {
            return DictToNodesInfo(
                    _modelManager.GetInputTypeDict(_modelName, _ignoredVersion),
                    _modelManager.GetInputShapesDict(_modelName, _ignoredVersion));
        }

        private OnnxNodeInfo[] GetOutputsInfo()
        {
            return DictToNodesInfo(
                    _modelManager.GetOutputTypeDict(_modelName, _ignoredVersion),
                    _modelManager.GetOutputShapesDict(_modelName, _ignoredVersion));
        }

        private static OnnxNodeInfo[] DictToNodesInfo(
            Dictionary<string, DataType> typeDict,
            Dictionary<string, long[]> shapeDictArray)
        {
            var shapeDict = new Dictionary<string, List<long>>();
            foreach (var key in shapeDictArray.Keys)
                shapeDict.Add(key, shapeDictArray[key].ToList());

            var sameKey = typeDict.Count == shapeDict.Count &&
                          typeDict.Keys.SequenceEqual(shapeDict.Keys);
            Contracts.Assert(sameKey, "Type and shape dictionaries should have the same keys");
            return typeDict.Select(kv => new OnnxNodeInfo(
                        name: kv.Key, type: kv.Value, shape: shapeDict[kv.Key])).OrderBy(x => x.Name).ToArray();
        }
    }

    internal sealed class OnnxUtils
    {
        /// <summary>
        /// Sonoma API only provides Tensor() constructors with overloaded
        /// versions based on data type.
        /// </summary>

        private static Dictionary<System.Type, DataType> _typeMap;

        public static Tensor CreateScalarTensor<T>(T data)
        {
            if (typeof(T) == typeof(System.Boolean))
            {
                return new Tensor((System.Boolean)(object)data);
            }
            else if (typeof(T) == typeof(System.Byte))
            {
                return new Tensor((System.Byte)(object)data);
            }
            else if (typeof(T) == typeof(System.Char))
            {
                return new Tensor((System.Char)(object)data);
            }
            else if (typeof(T) == typeof(System.Double))
            {
                return new Tensor((System.Double)(object)data);
            }
            else if (typeof(T) == typeof(System.Single))
            {
                return new Tensor((System.Single)(object)data);
            }
            else if (typeof(T) == typeof(System.Int32))
            {
                return new Tensor((System.Int32)(object)data);
            }
            else if (typeof(T) == typeof(System.Int64))
            {
                return new Tensor((System.Int64)(object)data);
            }
            else if (typeof(T) == typeof(System.SByte))
            {
                return new Tensor((System.SByte)(object)data);
            }
            else if (typeof(T) == typeof(System.Int16))
            {
                return new Tensor((System.Int16)(object)data);
            }
            else if (typeof(T) == typeof(System.UInt32))
            {
                return new Tensor((System.UInt32)(object)data);
            }
            else if (typeof(T) == typeof(System.UInt64))
            {
                return new Tensor((System.UInt64)(object)data);
            }
            else if (typeof(T) == typeof(System.UInt16))
            {
                return new Tensor((System.UInt16)(object)data);
            }
            throw new NotSupportedException($"Unsupported type {typeof(T)}");
        }

        /// <summary>
        /// Sonoma API only provides Tensor() constructors with overloaded versions
        /// based on data type. ML.NET cannot use the overloaded version and requires
        /// generic version. CreateTensor&lt;T&gt; is generic wrapper on top of
        /// overloaded Tensor(T[] data, OnnxShape shape) constructors.
        /// </summary>
        public static Tensor CreateTensor<T>(ReadOnlySpan<T> data, OnnxShape shape)
        {
            if (typeof(T) == typeof(System.Boolean))
            {
                return new Tensor((System.Boolean[])(object)data.ToArray(), shape.ToArray());
            }
            else if (typeof(T) == typeof(System.Double))
            {
                return new Tensor((System.Double[])(object)data.ToArray(), shape.ToArray());
            }
            else if (typeof(T) == typeof(System.Single))
            {
                return new Tensor((System.Single[])(object)data.ToArray(), shape.ToArray());
            }
            else if (typeof(T) == typeof(System.Int32))
            {
                return new Tensor((System.Int32[])(object)data.ToArray(), shape.ToArray());
            }
            else if (typeof(T) == typeof(System.Int64))
            {
                return new Tensor((System.Int64[])(object)data.ToArray(), shape.ToArray());
            }
            throw new NotImplementedException($"Not implemented type {typeof(T)}");
        }

        /// <summary>
        /// Sonoma API only provides CopyTo() functions with overloaded versions
        /// based on data type. ML.NET cannot use the overloaded version and requires
        /// generic version. CopyTo&lt;T&gt; is generic wrapper on top of
        /// overloaded Tensor.CopyTo(List&lt;T&gt; dst) methods.
        /// Also Tensor.CopyTo(List&lt;T&gt; dst) requires a list input, whereas ML.NET
        /// provides array buffers to copy values to. This mismatch causes an extra copy.
        /// </summary>
        public static unsafe void CopyTo<T>(Tensor tensor, Span<T> dst)
        {
            var typeMap = SystemTypeToOnnxType();
            if (typeMap.ContainsKey(typeof(T)))
            {
                if (tensor.GetDataType() != typeMap[typeof(T)])
                {
                    throw new InvalidOperationException( string.Format("Cannot copy source tensor of type {0} to managed type {1}.", tensor.GetDataType(), typeof(T)));
                }
                Span<T> tensorSpan = new Span<T>(tensor.UnsafeGetData().ToPointer(), tensor.GetSize());
                tensorSpan.CopyTo(dst);
                // TODO: the CopyTo() function is susceptible to GC reclaiming tensor
                // during the method call. Use KeepAlive for now, and remove
                // after permanent fix in CopyTo().
            }
            else
                throw new NotImplementedException($"Not implemented type {typeof(T)}");
            GC.KeepAlive(tensor);
        }

        public static PrimitiveType OnnxToMlNetType(DataType type)
        {
            DataKind kind;
            switch (type)
            {
                case DataType.Type_Float:
                    kind = DataKind.R4;
                    break;

                case DataType.Type_Double:
                    kind = DataKind.R8;
                    break;

                case DataType.Type_Int8:
                    kind = DataKind.I1;
                    break;

                case DataType.Type_Int16:
                    kind = DataKind.I2;
                    break;

                case DataType.Type_Int32:
                    kind = DataKind.I4;
                    break;

                case DataType.Type_Int64:
                    kind = DataKind.I8;
                    break;

                case DataType.Type_Uint8:
                    kind = DataKind.U1;
                    break;

                case DataType.Type_Uint16:
                    kind = DataKind.U2;
                    break;

                case DataType.Type_String:
                    kind = DataKind.TX;
                    break;

                case DataType.Type_Bool:
                    kind = DataKind.BL;
                    break;

                case DataType.Type_Invalid:
                default:
                    throw Contracts.ExceptNotSupp("Onnx type not supported", type);
            }

            return PrimitiveType.FromKind(kind);
        }

        internal static Dictionary<System.Type, DataType> SystemTypeToOnnxType()
        {
            if (_typeMap == null)
            {
                _typeMap = new Dictionary<System.Type, DataType>
                {
                    { typeof(Boolean) , DataType.Type_Bool },
                    { typeof(Double) , DataType.Type_Double },
                    { typeof(Single) , DataType.Type_Float },
                    { typeof(Int16) , DataType.Type_Int16 },
                    { typeof(Int32) , DataType.Type_Int32 },
                    { typeof(Int64) , DataType.Type_Int64 },
                    { typeof(UInt16) , DataType.Type_Uint16 }
                };
            }
            return _typeMap;
        }
    }
}
