// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
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
    internal sealed class OnnxModel : IDisposable
    {
        /// <summary>
        /// OnnxModelInfo contains the data that we should get from
        /// OnnxRuntime API once that functionality is added.
        /// </summary>
        public sealed class OnnxModelInfo
        {
            /// <summary>
            /// InputNames[i] is the name of the i-th element in <see cref="InputsInfo"/>.
            /// </summary>
            public List<string> InputNames { get; }
            /// <summary>
            /// OutputNames[i] is the name of the i-th element in <see cref="OutputsInfo"/>.
            /// </summary>
            public List<string> OutputNames { get; }
            /// <summary>
            /// Inputs of the containing <see cref="OnnxModel"/>.
            /// </summary>
            public OnnxVariableInfo[] InputsInfo { get; }
            /// <summary>
            /// Outputs of the containing <see cref="OnnxModel"/>.
            /// </summary>
            public OnnxVariableInfo[] OutputsInfo { get; }

            public OnnxModelInfo(IEnumerable<OnnxVariableInfo> inputsInfo, IEnumerable<OnnxVariableInfo> outputsInfo)
            {
                InputNames = inputsInfo.Select(val => val.Name).ToList();
                InputsInfo = inputsInfo.ToArray();
                OutputNames = outputsInfo.Select(val => val.Name).ToList();
                OutputsInfo = outputsInfo.ToArray();
            }

            /// <summary>
            /// Return the ONNX value for a <see cref="IDataView"/> input column called <paramref name="name"/>.
            /// </summary>
            public OnnxVariableInfo GetInput(string name)
            {
                var index = InputNames.IndexOf(name);
                if (index < 0)
                    throw Contracts.ExceptParamValue(name, nameof(name), $"Input tensor, {name}, does not exist in the ONNX model. " +
                        $"Available input names are [{string.Join(",", InputNames)}].");
                return InputsInfo[index];
            }

            /// <summary>
            /// Return the ONNX value for a <see cref="IDataView"/> output column called <paramref name="name"/>.
            /// </summary>
            public OnnxVariableInfo GetOutput(string name)
            {
                var index = OutputNames.IndexOf(name);
                if (index < 0)
                    throw Contracts.ExceptParamValue(name, nameof(name), $"Onput tensor, {name}, does not exist in the ONNX model. " +
                        $"Available output names are [{string.Join(",", OutputNames)}].");
                return OutputsInfo[index];
            }
        }

        /// <summary>
        /// OnnxNodeInfo contains all the information for a given node (e.g. inputs/outputs)
        /// of an Onnx model.
        /// </summary>
        public class OnnxVariableInfo
        {
            /// <summary>
            /// The Name of the variable. Note that ONNX variable are named.
            /// </summary>
            public string Name { get; }
            /// <summary>
            /// The shape of the variable if the variable is a tensor. For other
            /// types such sequence and dictionary, <see cref="Shape"/> would be
            /// <see langword="null"/>.
            /// </summary>
            public OnnxShape Shape { get; }
            /// <summary>
            /// The type of the variable produced by ONNXRuntime.
            /// </summary>
            public Type TypeInOnnxRuntime { get; }
            /// <summary>
            /// The <see cref="Data.DataViewType"/> that this ONNX variable corresponds
            /// to in <see cref="IDataView"/>'s type system.
            /// </summary>
            public DataViewType DataViewType { get; }
            /// <summary>
            /// A method to case <see cref="NamedOnnxValue"/> produced by
            /// ONNXRuntime to the type specified in <see cref="DataViewType"/>.
            /// </summary>
            public Func<NamedOnnxValue, object> Caster { get; }

            public OnnxVariableInfo(string name, OnnxShape shape, Type typeInOnnxRuntime, DataViewType mlnetType, Func<NamedOnnxValue, object> caster)
            {
                Name = name;
                Shape = shape;
                TypeInOnnxRuntime = typeInOnnxRuntime;
                DataViewType = mlnetType;
                Caster = caster;
            }
        }

        /// <summary>
        /// The ONNXRuntime facility to execute the loaded ONNX model.
        /// </summary>
        private readonly InferenceSession _session;
        /// <summary>
        /// Indicates if <see cref="ModelFile"/> is a temporal file created by <see cref="CreateFromBytes(byte[], int?, bool, IDictionary{string, int[]})"/>
        /// or <see cref="CreateFromBytes(byte[])"/>. If <see langword="true"/>, <see cref="Dispose(bool)"/> should delete <see cref="ModelFile"/>.
        /// </summary>
        private bool _ownModelFile;
        /// <summary>
        /// The location where the used ONNX model loaded from.
        /// </summary>
        internal string ModelFile { get; }
        /// <summary>
        /// The ONNX model's information from ONNXRuntime's perspective. ML.NET can change the input and output of that model in some ways.
        /// For example, ML.NET can shuffle the inputs so that the i-th ONNX input becomes the j-th input column of <see cref="OnnxTransformer"/>.
        /// ML.NET can also only exposes a subset of ONNX outputs in <see cref="OnnxTransformer"/>.
        /// </summary>
        internal OnnxModelInfo ModelInfo { get; }

        /// <summary>
        /// Constructs OnnxModel object from file.
        /// </summary>
        /// <param name="modelFile">Model file path.</param>
        /// <param name="gpuDeviceId">GPU device ID to execute on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If true, resumes CPU execution quitely upon GPU error.</param>
        /// <param name="ownModelFile">If true, the <paramref name="modelFile"/> will be deleted when <see cref="OnnxModel"/> is
        /// no longer needed.</param>
        /// <param name="shapeDictionary"></param>
        public OnnxModel(string modelFile, int? gpuDeviceId = null, bool fallbackToCpu = false,
            bool ownModelFile=false, IDictionary<string, int[]> shapeDictionary = null)
        {
            ModelFile = modelFile;
            // If we don't own the model file, _disposed should be false to prevent deleting user's file.
            _ownModelFile = ownModelFile;
            _disposed = false;

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

            // Load ONNX model file and parse its input and output schema. The reason of doing so is that ONNXRuntime
            // doesn't expose full type information via its C# APIs.
            ModelFile = modelFile;
            var model = new OnnxCSharpToProtoWrapper.ModelProto();
            using (var modelStream = File.OpenRead(modelFile))
            using (var codedStream = Google.Protobuf.CodedInputStream.CreateWithLimits(modelStream, Int32.MaxValue, 10))
                model = OnnxCSharpToProtoWrapper.ModelProto.Parser.ParseFrom(codedStream);

            // Parse actual input and output types stored in the loaded ONNX model to get their DataViewType's.
            var inputTypePool = new Dictionary<string, DataViewType>();
            foreach (var valueInfo in model.Graph.Input)
                inputTypePool[valueInfo.Name] = OnnxTypeParser.GetDataViewType(valueInfo.Type);
            var outputTypePool = new Dictionary<string, DataViewType>();

            // Build casters which maps NamedOnnxValue to .NET objects.
            var casterPool = new Dictionary<string, Func<NamedOnnxValue, object>>();
            foreach (var valueInfo in model.Graph.Output)
            {
                outputTypePool[valueInfo.Name] = OnnxTypeParser.GetDataViewType(valueInfo.Type);
                casterPool[valueInfo.Name] = OnnxTypeParser.GetDataViewValueCasterAndResultedType(valueInfo.Type, out Type actualType);
            }

            var onnxRuntimeInputInfos = new List<OnnxVariableInfo>();
            // Collect input information for this ONNX model from ONNXRuntime's perspective.
            foreach (var pair in _session.InputMetadata)
            {
                var name = pair.Key;
                var meta = pair.Value;
                var dataViewType = inputTypePool[name];

                OnnxVariableInfo info = null;
                if (shapeDictionary != null && shapeDictionary.ContainsKey(name))
                {
                    // If user provides a shape of a specific tensor, the provided shape overwrites the corresponding one loaded from
                    // ONNX model file and the deduced DataViewVectorType.

                    if (!CheckOnnxShapeCompatibility(shapeDictionary[name].ToList(), meta.Dimensions.ToList()))
                        throw Contracts.ExceptParamValue(shapeDictionary[name], nameof(shapeDictionary),
                            "The specified shape " + string.Join(",", shapeDictionary[name]) +
                            " is not compatible with the shape " + string.Join(",", meta.Dimensions) +
                            " loaded from the ONNX model file. Only unknown dimension can replace or " +
                            "be replaced by another dimension.");

                    if (dataViewType is VectorDataViewType vectorType)
                    {
                        if (shapeDictionary[name].All(value => value > 0))
                            dataViewType = new VectorDataViewType(vectorType.ItemType, shapeDictionary[name]);
                        else
                            dataViewType = new VectorDataViewType(vectorType.ItemType);
                    }

                    info = new OnnxVariableInfo(name, shapeDictionary[name].ToList(), meta.ElementType, dataViewType, null);
                }
                else
                {
                    // No user-specified shape is found, so the shape loaded from ONNX model file is used.
                    info = new OnnxVariableInfo(name, meta.Dimensions.ToList(), meta.ElementType, dataViewType, null);
                }
                onnxRuntimeInputInfos.Add(info);
            }

            var onnxRuntimeOutputInfos = new List<OnnxVariableInfo>();
            // Collect output information for this ONNX model from ONNXRuntime's perspective.
            foreach (var pair in _session.OutputMetadata)
            {
                var name = pair.Key;
                var meta = pair.Value;
                var dataViewType = outputTypePool[name];
                var caster = casterPool[name];

                OnnxVariableInfo info = null;
                if (shapeDictionary != null && shapeDictionary.ContainsKey(name))
                {
                    // If user provide a shape of a specific tensor, the provided shape overwrites the corresponding one loaded from
                    // ONNX model file.

                    if (!CheckOnnxShapeCompatibility(shapeDictionary[name].ToList(), meta.Dimensions.ToList()))
                        throw Contracts.ExceptParamValue(shapeDictionary[name], nameof(shapeDictionary),
                            "The specified shape " + string.Join(",", shapeDictionary[name]) +
                            " is not compatible with the shape " + string.Join(",", meta.Dimensions) +
                            " loaded from the ONNX model file. Only unknown dimension can replace or " +
                            "be replaced by another dimension.");

                    if (dataViewType is VectorDataViewType vectorType)
                    {
                        if (shapeDictionary[name].All(value => value > 0))
                            dataViewType = new VectorDataViewType(vectorType.ItemType, shapeDictionary[name]);
                        else
                            dataViewType = new VectorDataViewType(vectorType.ItemType);
                    }

                    info = new OnnxVariableInfo(name, shapeDictionary[name].ToList(), meta.ElementType, dataViewType, caster);
                }
                else
                {
                    // No user-specified shape is found, so the shape loaded from ONNX model file is used.
                    info = new OnnxVariableInfo(name, meta.Dimensions.ToList(), meta.ElementType, dataViewType, caster);
                }

                onnxRuntimeOutputInfos.Add(info);
            }

            // Create a view to the used ONNX model from ONNXRuntime's perspective.
            ModelInfo = new OnnxModelInfo(onnxRuntimeInputInfos, onnxRuntimeOutputInfos);
        }

        /// <summary>
        /// This function returns <see langword="true"/> if <paramref name="left"/> and <paramref name="right"/> are
        /// compatible. Otherwise, <see langword="false"/> is returned.
        ///
        /// Patterns leads to <see langword="true"/>.
        /// Left:        Right:
        ///   [-1, 3]      [2, 3]
        ///   [2, 3]       [-1, 3]
        ///   [-1, 3, -3]  [-2, 3, -1]
        ///
        /// </summary>
        /// <param name="left">An ONNX shape.</param>
        /// <param name="right">An ONNX shape.</param>
        /// <returns><see langword="true"/> if <paramref name="left"/> and <paramref name="right"/> are compatible and
        /// <see langword="false"/> otherwise.</returns>
        private static bool CheckOnnxShapeCompatibility(IEnumerable<int> left, IEnumerable<int> right)
        {
            if (left.Count() != right.Count())
                return false;
            foreach(var (l, r) in left.Zip(right, (l, r) => (l, r)))
            {
                // Along a specific axis, if any of left or right have unknown dimension, the overwriting can happen.
                if (l != r && l > 0 && r > 0)
                    return false;
            };
            return true;
        }

        /// <summary>
        /// Create an OnnxModel from a byte[]. Usually, a ONNX model is consumed by <see cref="OnnxModel"/> as a file.
        /// With <see cref="CreateFromBytes(byte[])"/> and <see cref="CreateFromBytes(byte[], int?, bool, IDictionary{string, int[]})"/>,
        /// it's possible to use in-memory model (type: byte[]) to create <see cref="OnnxModel"/>.
        /// </summary>
        /// <param name="modelBytes">Bytes of the serialized model</param>
        public static OnnxModel CreateFromBytes(byte[] modelBytes)
        {
            return CreateFromBytes(modelBytes, null, false);
        }

        /// <summary>
        /// Create an OnnxModel from a byte[]. Set execution to GPU if required.
        /// Usually, a ONNX model is consumed by <see cref="OnnxModel"/> as a file.
        /// With <see cref="CreateFromBytes(byte[])"/> and
        /// <see cref="CreateFromBytes(byte[], int?, bool, IDictionary{string, int[]})"/>,
        /// it's possible to use in-memory model (type: byte[]) to create <see cref="OnnxModel"/>.
        /// </summary>
        /// <param name="modelBytes">Bytes of the serialized model.</param>
        /// <param name="gpuDeviceId">GPU device ID to execute on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If true, resumes CPU execution quietly upon GPU error.</param>
        /// <param name="shapeDictionary">User-provided shapes. If the key "myTensorName" is associated
        /// with the value [1, 3, 5], the shape of "myTensorName" will be set to [1, 3, 5].
        /// The shape loaded from <paramref name="modelBytes"/> would be overwritten.</param>
        /// <returns>An <see cref="OnnxModel"/></returns>
        public static OnnxModel CreateFromBytes(byte[] modelBytes, int? gpuDeviceId = null, bool fallbackToCpu = false,
            IDictionary<string, int[]> shapeDictionary = null)
        {
            var tempModelDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
            Directory.CreateDirectory(tempModelDir);

            var tempModelFile = Path.Combine(tempModelDir, "model.onnx");
            File.WriteAllBytes(tempModelFile, modelBytes);
            return new OnnxModel(tempModelFile, gpuDeviceId, fallbackToCpu,
                ownModelFile: true, shapeDictionary: shapeDictionary);
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
        /// Flag used to indicate if the unmanaged resources (aka the model file <see cref="ModelFile"/>
        /// and <see cref="_session"/>) have been deleted.
        /// </summary>
        private bool _disposed;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// There are two unmanaged resources we can dispose, <see cref="_session"/> and <see cref="ModelFile"/>
        /// if <see cref="_ownModelFile"/> is <see langword="true"/>.
        /// </summary>
        /// <param name="disposing"></param>
        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                // There are two things to be disposed.
                if (disposing)
                {
                    // First, we release the resource token by ONNXRuntime.
                    _session.Dispose();
                    // Second, we delete the model file if that file is not created by the user.
                    if (_ownModelFile && File.Exists(ModelFile))
                        File.Delete(ModelFile);
                }
                _disposed = true;
            }
        }

        ~OnnxModel()
        {
            Dispose(false);
        }
    }

    internal sealed class OnnxUtils
    {
        private static HashSet<Type> _onnxTypeMap =
            new HashSet<Type>
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
        private static Dictionary<Type, InternalDataKind> _typeToKindMap=
            new Dictionary<Type, InternalDataKind>
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
        public static PrimitiveDataViewType OnnxToMlNetType(Type type)
        {
            if (!_typeToKindMap.ContainsKey(type))
               throw Contracts.ExceptNotSupp("Onnx type not supported", type);
            return ColumnTypeExtensions.PrimitiveTypeFromKind(_typeToKindMap[type]);
        }
    }
}
