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

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// IdvToTensorAdapter adapts an Idv (row-iterator interface) to a tensor-iterator interface.
    /// For an Idv, you'd need to create a cursor and iterate over it to get each rows of the Idv.
    /// After adaptation, you'd call GetTensor() on the IdvToTensorAdapter object to get the Tensor equivalent of
    /// each row.
    /// </summary>
    internal sealed class IdvToTensorAdapter
    {
        // Idv information
        private readonly string _idvColumnName;
        internal readonly int IdvColumnIndex;
        private readonly bool _idvIsVectorColumn;
        public readonly ColumnType IdvColumnType;

        // Onnx tensor information
        private readonly OnnxShape _onnxTensorShape;

        private ITensorValueGetter _tensorValueGetter;

        public IdvToTensorAdapter(Schema idvSchema, string idvColumnName,
                                    OnnxModel.OnnxNodeInfo onnxInputNodeInfo)
        {
            _idvColumnName = idvColumnName;
            if (!idvSchema.TryGetColumnIndex(_idvColumnName, out IdvColumnIndex))
                throw Contracts.Except($"Column '{_idvColumnName}' does not exist");
            IdvColumnType = idvSchema.GetColumnType(IdvColumnIndex);
            _idvIsVectorColumn = IdvColumnType.IsVector;
            _onnxTensorShape = onnxInputNodeInfo.Shape;

            // TODO: Check that the idv and tensor sizes match
            // TODO: Check type matches

            // TODO: Add Yaels shape logic here
            if (_onnxTensorShape[0] == -1)
                _onnxTensorShape[0] = 1;
        }

        public void InitializeValueGetters(IRow idvRow)
        {
            var type = IdvColumnType.ItemType.RawType;
            _tensorValueGetter = Utils.MarshalInvoke(
                CreateTensorValueGetter<int>, type, idvRow, _idvIsVectorColumn, IdvColumnIndex, _onnxTensorShape);
        }

        public Tensor GetTensor()
        {
            return _tensorValueGetter.GetTensor();
        }

        private ITensorValueGetter CreateTensorValueGetter<T>(IRow input, bool isVector, int colIndex, OnnxShape tensorShape)
        {
            if (isVector)
                return new TensorValueGetterVec<T>(input, colIndex, tensorShape);
            else
                return new TensorValueGetter<T>(input, colIndex);
        }

        private interface ITensorValueGetter
        {
            Tensor GetTensor();
        }

        private class TensorValueGetter<T> : ITensorValueGetter
        {
            private readonly ValueGetter<T> _srcgetter;

            public TensorValueGetter(IRow input, int colIndex)
            {
                _srcgetter = input.GetGetter<T>(colIndex);
            }
            public Tensor GetTensor()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                return OnnxUtils.CreateScalarTensor(scalar);
            }
        }

        private class TensorValueGetterVec<T> : ITensorValueGetter
        {
            private readonly ValueGetter<VBuffer<T>> _srcgetter;
            private readonly OnnxShape _tensorShape;
            private VBuffer<T> _vBuffer;
            private VBuffer<T> _vBufferDense;
            public TensorValueGetterVec(IRow input, int colIndex, OnnxShape tensorShape)
            {
                _srcgetter = input.GetGetter<VBuffer<T>>(colIndex);
                _tensorShape = tensorShape;
                _vBuffer = default;
                _vBufferDense = default;
            }
            public Tensor GetTensor()
            {
                _srcgetter(ref _vBuffer);
                _vBuffer.CopyToDense(ref _vBufferDense);
                return OnnxUtils.CreateTensor(_vBufferDense.Values, _tensorShape);
            }
        }
    }

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
        private readonly List<string> _inputNames;
        private readonly List<string> _outputNames;

        public OnnxModel(string modelFile)
        {
            _modelFile = modelFile;

            // Load the onnx model
            var modelFileInfo = new FileInfo(modelFile);
            _modelName = Path.GetFileNameWithoutExtension(modelFileInfo.Name);
            _modelManager = new ModelManager(modelFileInfo.Directory.FullName, true);
            _modelManager.InitOnnxModel(_modelName, _ignoredVersion);

            ModelInfo = new OnnxModelInfo(GetInputsInfo(), GetOutputsInfo());
            _inputNames = ModelInfo.InputsInfo.Select(i => i.Name).ToList();
            _outputNames = ModelInfo.OutputsInfo.Select(i => i.Name).ToList();
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
                _modelName, _ignoredVersion, _inputNames, inputTensors, _outputNames);

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
        public static Tensor CreateTensor<T>(T[] data, OnnxShape shape)
        {
            if (typeof(T) == typeof(System.Boolean))
            {
                return new Tensor(((System.Boolean[])(object)data).ToList(), shape);
            }
            else if (typeof(T) == typeof(System.Double))
            {
                return new Tensor(((System.Double[])(object)data).ToList(), shape);
            }
            else if (typeof(T) == typeof(System.Single))
            {
                return new Tensor(((System.Single[])(object)data).ToList(), shape);
            }
            else if (typeof(T) == typeof(System.Int32))
            {
                return new Tensor(((System.Int32[])(object)data).ToList(), shape);
            }
            else if (typeof(T) == typeof(System.Int64))
            {
                return new Tensor(((System.Int64[])(object)data).ToList(), shape);
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
        public static void CopyTo<T>(Tensor tensor, T[] dst)
        {
            if (typeof(T) == typeof(System.Single))
            {
                var typedDst = (System.Single[])(object)dst;
                tensor.CopyTo(typedDst);
            }
            else
                throw new NotImplementedException($"Not implemented type {typeof(T)}");
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
    }
}
