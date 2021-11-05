// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Runtime;
using static Microsoft.ML.Model.OnnxConverter.OnnxCSharpToProtoWrapper;
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
            /// Initializers[i] is the name of the i-th initializer in <see cref="InitializersInfo"/>.
            /// </summary>
            public List<string> InitializerNames { get; }
            /// <summary>
            /// Inputs of the containing <see cref="OnnxModel"/>.
            /// </summary>
            public OnnxVariableInfo[] InputsInfo { get; }
            /// <summary>
            /// Outputs of the containing <see cref="OnnxModel"/>.
            /// </summary>
            public OnnxVariableInfo[] OutputsInfo { get; }

            /// <summary>
            /// Initializers of the containing <see cref="OnnxModel"/>
            /// </summary>
            public OnnxVariableInfo[] InitializersInfo { get; }

            public OnnxModelInfo(IEnumerable<OnnxVariableInfo> inputsInfo, IEnumerable<OnnxVariableInfo> outputsInfo, IEnumerable<OnnxVariableInfo> initializersInfo)
            {
                InputNames = inputsInfo.Select(val => val.Name).ToList();
                InputsInfo = inputsInfo.ToArray();
                OutputNames = outputsInfo.Select(val => val.Name).ToList();
                OutputsInfo = outputsInfo.ToArray();
                InitializerNames = initializersInfo.Select(val => val.Name).ToList();
                InitializersInfo = initializersInfo.ToArray();
            }

            /// <summary>
            /// Return the ONNX value for a <see cref="IDataView"/> input column called <paramref name="name"/>.
            /// </summary>
            public OnnxVariableInfo GetInput(string name)
            {
                var index = InputNames.IndexOf(name);
                if (index >= 0)
                    return InputsInfo[index];

                index = InitializerNames.IndexOf(name);
                if (index >= 0)
                    return InitializersInfo[index];

                // If we dont find the index in the input, try find it in the initializers
                throw Contracts.ExceptParamValue(name, nameof(name), $"Input tensor, {name}, does not exist in the ONNX model. " +
                    $"Available input names are [{string.Join(",", InputNames)}]. Available initializers are [{string.Join(",", InitializerNames)}]");
            }

            /// <summary>
            /// Return the ONNX value for a <see cref="IDataView"/> output column called <paramref name="name"/>.
            /// </summary>
            public OnnxVariableInfo GetOutput(string name)
            {
                var index = OutputNames.IndexOf(name);
                if (index < 0)
                    throw Contracts.ExceptParamValue(name, nameof(name), $"Ouput tensor, {name}, does not exist in the ONNX model. " +
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
        /// The FileStream holding onto the loaded ONNX model.
        /// </summary>
        internal FileStream ModelStream { get; }
        /// <summary>
        /// The ONNX model's information from ONNXRuntime's perspective. ML.NET can change the input and output of that model in some ways.
        /// For example, ML.NET can shuffle the inputs so that the i-th ONNX input becomes the j-th input column of <see cref="OnnxTransformer"/>.
        /// ML.NET can also only exposes a subset of ONNX outputs in <see cref="OnnxTransformer"/>.
        /// </summary>
        internal OnnxModelInfo ModelInfo { get; }

        internal GraphProto Graph { get; }

        /// <summary>
        /// Constructs OnnxModel object from file.
        /// </summary>
        /// <param name="modelFile">Model file path.</param>
        /// <param name="gpuDeviceId">GPU device ID to execute on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If true, resumes CPU execution quietly upon GPU error.</param>
        /// <param name="ownModelFile">If true, the <paramref name="modelFile"/> will be deleted when <see cref="OnnxModel"/> is
        /// no longer needed.</param>
        /// <param name="shapeDictionary"></param>
        /// <param name="recursionLimit">Optional, specifies the Protobuf CodedInputStream recursion limit. Default value is 100.</param>
        /// <param name="interOpNumThreads">Controls the number of threads used to parallelize the execution of the graph (across nodes).</param>
        /// <param name="intraOpNumThreads">Controls the number of threads to use to run the model.</param>
        public OnnxModel(string modelFile, int? gpuDeviceId = null, bool fallbackToCpu = false,
            bool ownModelFile = false, IDictionary<string, int[]> shapeDictionary = null, int recursionLimit = 100,
            int? interOpNumThreads = null, int? intraOpNumThreads = null)
        {
            // If we don't own the model file, _disposed should be false to prevent deleting user's file.
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
                    {
                        var sessionOptions = new SessionOptions()
                        {
                            InterOpNumThreads = interOpNumThreads.GetValueOrDefault(),
                            IntraOpNumThreads = intraOpNumThreads.GetValueOrDefault()
                        };
                        _session = new InferenceSession(modelFile, sessionOptions);
                    }
                    else
                        // If called from OnnxTransform, is caught and rethrown
                        throw;
                }
            }
            else
            {
                var sessionOptions = new SessionOptions()
                {
                    InterOpNumThreads = interOpNumThreads.GetValueOrDefault(),
                    IntraOpNumThreads = intraOpNumThreads.GetValueOrDefault()
                };
                _session = new InferenceSession(modelFile, sessionOptions);
            }

            try
            {
                // Load ONNX model file and parse its input and output schema. The reason of doing so is that ONNXRuntime
                // doesn't expose full type information via its C# APIs.
                var model = new OnnxCSharpToProtoWrapper.ModelProto();
                // If we own the model file set the DeleteOnClose flag so it is always deleted.
                if (ownModelFile)
                    ModelStream = new FileStream(modelFile, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.DeleteOnClose);
                else
                    ModelStream = new FileStream(modelFile, FileMode.Open, FileAccess.Read);

                // The CodedInputStream auto closes the stream, and we need to make sure that our main stream stays open, so creating a new one here.
                using (var modelStream = new FileStream(modelFile, FileMode.Open, FileAccess.Read, FileShare.Delete | FileShare.Read))
                using (var codedStream = Google.Protobuf.CodedInputStream.CreateWithLimits(modelStream, Int32.MaxValue, recursionLimit))
                    model = OnnxCSharpToProtoWrapper.ModelProto.Parser.ParseFrom(codedStream);

                // Parse actual input and output types stored in the loaded ONNX model to get their DataViewType's.
                var inputTypePool = new Dictionary<string, DataViewType>();
                foreach (var valueInfo in model.Graph.Input)
                    inputTypePool[valueInfo.Name] = OnnxTypeParser.GetDataViewType(valueInfo.Type);

                var initializerTypePool = new Dictionary<string, DataViewType>();
                foreach (var valueInfo in model.Graph.Initializer)
                    initializerTypePool[valueInfo.Name] = OnnxTypeParser.GetScalarDataViewType(valueInfo.DataType);

                var outputTypePool = new Dictionary<string, DataViewType>();
                // Build casters which maps NamedOnnxValue to .NET objects.
                var casterPool = new Dictionary<string, Func<NamedOnnxValue, object>>();
                foreach (var valueInfo in model.Graph.Output)
                {
                    outputTypePool[valueInfo.Name] = OnnxTypeParser.GetDataViewType(valueInfo.Type);
                    casterPool[valueInfo.Name] = OnnxTypeParser.GetDataViewValueCasterAndResultedType(valueInfo.Type, out Type actualType);
                }

                var inputInfos = GetOnnxVariablesFromMetadata(_session.InputMetadata, shapeDictionary, inputTypePool, null);
                var outputInfos = GetOnnxVariablesFromMetadata(_session.OutputMetadata, shapeDictionary, outputTypePool, casterPool);
                var overrideableInitializers = GetOnnxVariablesFromMetadata(_session.OverridableInitializerMetadata, shapeDictionary, inputTypePool, null);

                // Create a view to the used ONNX model from ONNXRuntime's perspective.
                ModelInfo = new OnnxModelInfo(inputInfos, outputInfos, overrideableInitializers);

                Graph = model.Graph;
            }
            catch
            {
                _session.Dispose();
                _session = null;
                throw;
            }
        }

        private List<OnnxVariableInfo> GetOnnxVariablesFromMetadata(IReadOnlyDictionary<string, NodeMetadata> nodeMetadata,
            IDictionary<string, int[]> shapeDictionary,
            Dictionary<string, DataViewType> typePool,
            Dictionary<string, Func<NamedOnnxValue, object>> casterPool)
        {
            var onnxVariableInfos = new List<OnnxVariableInfo>();

            foreach (var pair in nodeMetadata)
            {
                var name = pair.Key;
                var meta = pair.Value;
                var dataViewType = typePool[name];
                var caster = casterPool?[name];

                if (name.StartsWith("mlnet.") &&
                    (name.EndsWith(".unusedInput") || name.EndsWith(".unusedOutput")))
                    continue;

                OnnxVariableInfo info = null;
                if (shapeDictionary != null && shapeDictionary.ContainsKey(name))
                {
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

                onnxVariableInfos.Add(info);
            }
            return onnxVariableInfos;
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
            foreach (var (l, r) in left.Zip(right, (l, r) => (l, r)))
            {
                // Along a specific axis, if any of left or right have unknown dimension, the overwriting can happen.
                if (l != r && l > 0 && r > 0)
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Create an OnnxModel from a byte[]. Usually, a ONNX model is consumed by <see cref="OnnxModel"/> as a file.
        /// With <see cref="CreateFromBytes(byte[], IHostEnvironment)"/> and <see cref="CreateFromBytes(byte[], IHostEnvironment, int?, bool, IDictionary{string, int[]}, int)"/>,
        /// it's possible to use in-memory model (type: byte[]) to create <see cref="OnnxModel"/>.
        /// </summary>
        /// <param name="modelBytes">Bytes of the serialized model</param>
        /// <param name="env">IHostEnvironment</param>
        public static OnnxModel CreateFromBytes(byte[] modelBytes, IHostEnvironment env)
        {
            return CreateFromBytes(modelBytes, env, null, false);
        }

        /// <summary>
        /// Create an OnnxModel from a byte[]. Set execution to GPU if required.
        /// Usually, a ONNX model is consumed by <see cref="OnnxModel"/> as a file.
        /// With <see cref="CreateFromBytes(byte[], IHostEnvironment)"/> and
        /// <see cref="CreateFromBytes(byte[], IHostEnvironment, int?, bool, IDictionary{string, int[]}, int)"/>,
        /// it's possible to use in-memory model (type: byte[]) to create <see cref="OnnxModel"/>.
        /// </summary>
        /// <param name="modelBytes">Bytes of the serialized model.</param>
        /// <param name="env">IHostEnvironment</param>
        /// <param name="gpuDeviceId">GPU device ID to execute on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If true, resumes CPU execution quietly upon GPU error.</param>
        /// <param name="shapeDictionary">User-provided shapes. If the key "myTensorName" is associated
        /// with the value [1, 3, 5], the shape of "myTensorName" will be set to [1, 3, 5].
        /// The shape loaded from <paramref name="modelBytes"/> would be overwritten.</param>
        /// <param name="recursionLimit">Optional, specifies the Protobuf CodedInputStream recursion limit. Default value is 100.</param>
        /// <returns>An <see cref="OnnxModel"/></returns>
        public static OnnxModel CreateFromBytes(byte[] modelBytes, IHostEnvironment env, int? gpuDeviceId = null, bool fallbackToCpu = false,
            IDictionary<string, int[]> shapeDictionary = null, int recursionLimit = 100)
        {
            var tempModelFile = Path.Combine(((IHostEnvironmentInternal)env).TempFilePath, Path.GetRandomFileName());
            File.WriteAllBytes(tempModelFile, modelBytes);
            return new OnnxModel(tempModelFile, gpuDeviceId, fallbackToCpu,
                ownModelFile: true, shapeDictionary: shapeDictionary, recursionLimit);
        }

        /// <summary>
        /// Uses an open session to score a list of NamedOnnxValues.
        /// </summary>
        /// <param name="inputNamedOnnxValues">The NamedOnnxValues to score.</param>
        /// <param name="outputColumns">The active output columns.</param>
        /// <returns>Resulting output NamedOnnxValues list.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(List<NamedOnnxValue> inputNamedOnnxValues, List<string> outputColumns)
        {
            return _session.Run(inputNamedOnnxValues, outputColumns);
        }

        /// <summary>
        /// Flag used to indicate if the unmanaged resources (aka the model file handle <see cref="ModelStream"/>
        /// and <see cref="_session"/>) have been deleted.
        /// </summary>
        private bool _disposed;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// There are two unmanaged resources we can dispose, <see cref="_session"/> and <see cref="ModelStream"/>
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
                    // Second, Dispose of the model file stream.
                    ModelStream.Dispose();
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
        private static readonly HashSet<Type> _onnxTypeMap =
            new HashSet<Type>
                {
                     typeof(Double),
                     typeof(Single),
                     typeof(Int16),
                     typeof(Int32),
                     typeof(Int64),
                     typeof(UInt16),
                     typeof(UInt32),
                     typeof(UInt64),
                     typeof(ReadOnlyMemory<Char>),
                     typeof(Boolean),
                     typeof(SByte),
                     typeof(Byte)
                };
        private static readonly Dictionary<Type, InternalDataKind> _typeToKindMap =
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
                    { typeof(SByte) , InternalDataKind.I1},
                    { typeof(Byte) , InternalDataKind.U1},
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

            if (typeof(T) == typeof(ReadOnlyMemory<char>))
                return NamedOnnxValue.CreateFromTensor<string>(name, new DenseTensor<string>(new string[] { data.ToString() }, new int[] { 1, 1 }));

            return NamedOnnxValue.CreateFromTensor<T>(name, new DenseTensor<T>(new T[] { data }, new int[] { 1, 1 }));
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

            var dimensions = shape.Select(x => (int)x).ToArray();

            if (typeof(T) == typeof(ReadOnlyMemory<char>))
            {
                string[] stringData = new string[data.Length];
                for (int i = 0; i < data.Length; i++)
                    stringData[i] = data[i].ToString();

                return NamedOnnxValue.CreateFromTensor<string>(name, new DenseTensor<string>(stringData, dimensions));
            }

            return NamedOnnxValue.CreateFromTensor<T>(name, new DenseTensor<T>(data.ToArray(), dimensions));
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
