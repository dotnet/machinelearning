// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using Microsoft.ML.Data;
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

        /// <summary>
        /// Constructs OnnxModel object from file.
        /// </summary>
        /// <param name="modelFile">Model file path.</param>
        /// <param name="gpuDeviceId">GPU device ID to execute on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If true, resumes CPU execution quitely upon GPU error.</param>
        public OnnxModel(string modelFile, int? gpuDeviceId = null, bool fallbackToCpu = false)
        {
            _modelFile = modelFile;

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
