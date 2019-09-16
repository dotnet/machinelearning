﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Dnn;
using Microsoft.ML.Transforms.TensorFlow;
using NumSharp;
using Tensorflow;
using static Microsoft.ML.Transforms.Dnn.DnnUtils;
using static Tensorflow.Binding;

[assembly: LoadableClass(TensorFlowTransformer.Summary, typeof(IDataTransform), typeof(TensorFlowTransformer),
    typeof(TensorFlowEstimator.Options), typeof(SignatureDataTransform), TensorFlowTransformer.UserName, TensorFlowTransformer.ShortName)]

[assembly: LoadableClass(TensorFlowTransformer.Summary, typeof(IDataTransform), typeof(TensorFlowTransformer), null, typeof(SignatureLoadDataTransform),
    TensorFlowTransformer.UserName, TensorFlowTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(TensorFlowTransformer), null, typeof(SignatureLoadModel),
    TensorFlowTransformer.UserName, TensorFlowTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TensorFlowTransformer), null, typeof(SignatureLoadRowMapper),
    TensorFlowTransformer.UserName, TensorFlowTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(TensorFlowTransformer))]

namespace Microsoft.ML.Transforms
{
    public sealed class TensorFlowTransformer : RowToRowTransformerBase
    {
        private readonly string _savedModelPath;
        private readonly bool _isTemporarySavedModel;
        private readonly bool _addBatchDimensionInput;
        internal readonly Session Session;
        internal readonly Runner Runner;
        internal readonly DataViewType[] OutputTypes;
        internal readonly TF_DataType[] TFOutputTypes;
        internal readonly TF_DataType[] TFInputTypes;
        internal readonly TensorShape[] TFInputShapes;
        internal readonly (Operation, int)[] TFInputOperations;
        internal readonly (Operation, int)[] TFOutputOperations;
        internal TF_Output[] TFInputNodes;
        internal TF_Output[] TFOutputNodes;
        internal IntPtr[] TFOperations;
        internal Graph Graph => Session.graph;

        internal readonly string[] Inputs;
        internal readonly string[] Outputs;
        internal static int BatchSize = 1;
        internal const string Summary = "Transforms the data using the TensorFlow model.";
        internal const string UserName = "TensorFlowTransform";
        internal const string ShortName = "TFTransform";
        internal const string LoaderSignature = "TensorFlowTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TENSFLOW",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002,  // Added Support for Multiple Outputs and SavedModel.
                verWrittenCur: 0x00010003,  // Added Support for adding batch dimension in inputs.
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TensorFlowTransformer).Assembly.FullName);
        }

        /// <summary>
        /// Transform for scoring Tensorflow models. Input data column names/types must exactly match
        /// all model input names. Only the output columns specified will be generated.
        /// This convenience method avoids reloading of TensorFlow model.
        /// It is useful in a situation where user has already loaded TensorFlow model using <see cref="TensorFlowUtils.LoadTensorFlowModel(IHostEnvironment, string)"/> for inspecting model schema.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="tfModelInfo"> <see cref="TensorFlowModel"/> object created with <see cref="TensorFlowUtils.LoadTensorFlowModel(IHostEnvironment, string)"/>.</param>
        /// <param name="outputColumnName">The output columns to generate. Names must match model specifications. Data types are inferred from model.</param>
        /// <param name="inputColumnName">The name of the input data columns. Must match model's input names. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        internal TensorFlowTransformer(IHostEnvironment env, TensorFlowModel tfModelInfo, string outputColumnName, string inputColumnName = null, bool addBatchDimensionInput = false)
            : this(env, tfModelInfo.Session, new[] { outputColumnName }, new[] { inputColumnName ?? outputColumnName }, DnnUtils.IsSavedModel(env, tfModelInfo.ModelPath) ? tfModelInfo.ModelPath : null, false, addBatchDimensionInput)
        {
        }

        /// <summary>
        /// Transform for scoring Tensorflow models. Input data column names/types must exactly match
        /// all model input names. Only the output columns specified will be generated.
        /// This convenience method avoids reloading of TensorFlow model.
        /// It is useful in a situation where user has already loaded TensorFlow model using <see cref="TensorFlowUtils.LoadTensorFlowModel(IHostEnvironment, string)"/> for inspecting model schema.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="tfModelInfo"> <see cref="TensorFlowModel"/> object created with <see cref="TensorFlowUtils.LoadTensorFlowModel(IHostEnvironment, string)"/>.</param>
        /// <param name="inputColumnNames">The name of the input data columns. Must match model's input names.</param>
        /// <param name="outputColumnNames">The output columns to generate. Names must match model specifications. Data types are inferred from model.</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        internal TensorFlowTransformer(IHostEnvironment env, TensorFlowModel tfModelInfo, string[] outputColumnNames, string[] inputColumnNames, bool addBatchDimensionInput = false)
            : this(env, tfModelInfo.Session, outputColumnNames, inputColumnNames, DnnUtils.IsSavedModel(env, tfModelInfo.ModelPath) ? tfModelInfo.ModelPath : null, false, addBatchDimensionInput)
        {
        }

        // Factory method for SignatureLoadModel.
        private static TensorFlowTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // byte: indicator for frozen models
            // byte: indicator for adding batch dimension in input
            // stream: tensorFlow model.
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            GetModelInfo(env, ctx, out string[] inputs, out string[] outputs, out bool isFrozen, out bool addBatchDimensionInput);
            if (isFrozen)
            {
                byte[] modelBytes = null;
                if (!ctx.TryLoadBinaryStream("TFModel", r => modelBytes = r.ReadByteArray()))
                    throw env.ExceptDecode();
                return new TensorFlowTransformer(env, DnnUtils.LoadTFSession(env, modelBytes), outputs, inputs, null, false, addBatchDimensionInput);
            }

            var tempDirPath = Path.GetFullPath(Path.Combine(Path.GetTempPath(), nameof(TensorFlowTransformer) + "_" + Guid.NewGuid()));
            DnnUtils.CreateFolderWithAclIfNotExists(env, tempDirPath);
            try
            {
                var load = ctx.TryLoadBinaryStream("TFSavedModel", br =>
                {
                    int count = br.ReadInt32();
                    for (int n = 0; n < count; n++)
                    {
                        string relativeFile = br.ReadString();
                        long fileLength = br.ReadInt64();

                        string fullFilePath = Path.Combine(tempDirPath, relativeFile);
                        string fullFileDir = Path.GetDirectoryName(fullFilePath);
                        if (fullFileDir != tempDirPath)
                        {
                            DnnUtils.CreateFolderWithAclIfNotExists(env, fullFileDir);
                        }
                        using (var fs = new FileStream(fullFilePath, FileMode.Create, FileAccess.Write))
                        {
                            long actualRead = br.BaseStream.CopyRange(fs, fileLength);
                            env.Assert(actualRead == fileLength);
                        }
                    }
                });

                return new TensorFlowTransformer(env, DnnUtils.GetSession(env, tempDirPath), outputs, inputs, tempDirPath, true, addBatchDimensionInput);
            }
            catch (Exception)
            {
                DnnUtils.DeleteFolderWithRetries(env, tempDirPath);
                throw;
            }
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, TensorFlowEstimator.Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.InputColumns, nameof(options.InputColumns));
            env.CheckValue(options.OutputColumns, nameof(options.OutputColumns));

            return new TensorFlowTransformer(env, options, input).MakeDataTransform(input);
        }

        internal TensorFlowTransformer(IHostEnvironment env, TensorFlowEstimator.Options options, IDataView input)
            : this(env, options, TensorFlowUtils.LoadTensorFlowModel(env, options.ModelLocation), input)
        {
        }

        internal TensorFlowTransformer(IHostEnvironment env, TensorFlowEstimator.Options options, TensorFlowModel tensorFlowModel, IDataView input, IDataView validationSet = null)
            : this(env, tensorFlowModel.Session, options.OutputColumns, options.InputColumns,
                  DnnUtils.IsSavedModel(env, options.ModelLocation) ? options.ModelLocation : null,
                  false, options.AddBatchDimensionInputs, options.BatchSize, options, input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
        }

        private static ITensorValueGetter CreateTensorValueGetter<T>(DataViewRow input, bool isVector, int colIndex, TensorShape tfShape)
        {
            if (isVector)
                return new TensorValueGetterVec<T>(input, colIndex, tfShape);
            return new TensorValueGetter<T>(input, colIndex, tfShape);
        }

        private static ITensorValueGetter CreateTensorValueGetter(DataViewRow input, TF_DataType tfType, bool isVector, int colIndex, TensorShape tfShape)
        {
            var type = DnnUtils.Tf2MlNetType(tfType);
            return Utils.MarshalInvoke(CreateTensorValueGetter<int>, type.RawType, input, isVector, colIndex, tfShape);
        }

        private static ITensorValueGetter[] GetTensorValueGetters(
            DataViewRow input,
            int[] inputColIndices,
            bool[] isInputVector,
            TF_DataType[] tfInputTypes,
            TensorShape[] tfInputShapes)
        {
            var srcTensorGetters = new ITensorValueGetter[inputColIndices.Length];
            for (int i = 0; i < inputColIndices.Length; i++)
            {
                int colIndex = inputColIndices[i];
                srcTensorGetters[i] = CreateTensorValueGetter(input, tfInputTypes[i], isInputVector[i], colIndex, tfInputShapes[i]);
            }
            return srcTensorGetters;
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private static void GetModelInfo(IHostEnvironment env, ModelLoadContext ctx, out string[] inputs, out string[] outputs, out bool isFrozen, out bool addBatchDimensionInput)
        {
            isFrozen = true;
            bool isNonFrozenModelSupported = ctx.Header.ModelVerReadable >= 0x00010002;
            if (isNonFrozenModelSupported)
                isFrozen = ctx.Reader.ReadBoolByte();

            addBatchDimensionInput = false;
            bool isAddingBatchDimensionSupported = ctx.Header.ModelVerReadable >= 0x00010003;
            if (isAddingBatchDimensionSupported)
                addBatchDimensionInput = ctx.Reader.ReadBoolByte();

            var numInputs = ctx.Reader.ReadInt32();
            env.CheckDecode(numInputs > 0);
            inputs = new string[numInputs];
            for (int j = 0; j < inputs.Length; j++)
                inputs[j] = ctx.LoadNonEmptyString();

            bool isMultiOutput = ctx.Header.ModelVerReadable >= 0x00010002;
            var numOutputs = 1;
            if (isMultiOutput)
                numOutputs = ctx.Reader.ReadInt32();

            env.CheckDecode(numOutputs > 0);
            outputs = new string[numOutputs];
            for (int j = 0; j < outputs.Length; j++)
                outputs[j] = ctx.LoadNonEmptyString();
        }

        internal TensorFlowTransformer(IHostEnvironment env, Session session, string[] outputColumnNames,
            string[] inputColumnNames, string savedModelPath, bool isTemporarySavedModel,
            bool addBatchDimensionInput, int batchSize = 1, TensorFlowEstimator.Options options = null, IDataView input = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TensorFlowTransformer)))

        {
            Host.CheckValue(session, nameof(session));
            Host.CheckNonEmpty(inputColumnNames, nameof(inputColumnNames));
            Host.CheckNonEmpty(outputColumnNames, nameof(outputColumnNames));

            _savedModelPath = savedModelPath;
            _isTemporarySavedModel = isTemporarySavedModel;
            Session = session;
            _addBatchDimensionInput = addBatchDimensionInput;
            Inputs = inputColumnNames;
            Outputs = outputColumnNames;

            (TFOutputTypes, OutputTypes, TFOutputOperations) = GetOutputInfo(Host, Session, Outputs);
            (TFInputTypes, TFInputShapes, TFInputOperations) = GetInputInfo(Host, Session, Inputs, batchSize);

            TFInputNodes = new TF_Output[Inputs.Length];
            TFOutputNodes = new TF_Output[Outputs.Length];

            for (int index = 0; index < TFInputOperations.Length; index += 1)
                TFInputNodes[index] = new TF_Output(TFInputOperations[index].Item1, TFInputOperations[index].Item2);

            for (int index = 0; index < TFOutputOperations.Length; index += 1)
                TFOutputNodes[index] = new TF_Output(TFOutputOperations[index].Item1, TFOutputOperations[index].Item2);
        }

        private static (Operation, int) GetOperationFromName(string operation, Session session)
        {
            var p = operation.IndexOf(':');

            if (p != -1 && p != operation.Length - 1)
            {
                var op = operation.Substring(0, p);
                if (int.TryParse(operation.Substring(p + 1), out var idx))
                {

                    return (session.graph.OperationByName(op), idx);
                }
            }
            return (session.graph.OperationByName(operation), 0);
        }

        internal static (TF_DataType[] tfInputTypes, TensorShape[] tfInputShapes, (Operation, int)[]) GetInputInfo(IHost host, Session session, string[] inputs, int batchSize = 1)
        {
            var tfInputTypes = new TF_DataType[inputs.Length];
            var tfInputShapes = new TensorShape[inputs.Length];
            var tfInputOperations = new (Operation, int)[inputs.Length];

            int index = 0;
            foreach (var input in inputs)
            {
                host.CheckNonWhiteSpace(input, nameof(inputs));
                (Operation inputTensor, int inputTensorIndex) = GetOperationFromName(input, session);

                if (inputTensor == null)
                    throw host.ExceptParam(nameof(inputs), $"Input column '{input}' does not exist in the model");

                TF_DataType tfInputType = string.Compare(inputTensor.OpType, "PlaceHolder", true) == 0 ? inputTensor.OutputType(inputTensorIndex) : inputTensor.InputType(inputTensorIndex);
                if (!DnnUtils.IsTypeSupported(tfInputType))
                    throw host.ExceptParam(nameof(session), $"Input type '{tfInputType}' of input column '{input}' is not supported in TensorFlow");

                tfInputTypes[index] = tfInputType;
                tfInputShapes[index] = ((Tensor)inputTensor).TensorShape;
                tfInputOperations[index] = (inputTensor, inputTensorIndex);
                index++;
            }

            return (tfInputTypes, tfInputShapes, tfInputOperations);
        }

        internal static TensorShape GetTensorShape(TF_Output output, Graph graph, Status status = null)
        {
            if (graph == IntPtr.Zero)
                new ObjectDisposedException(nameof(graph));

            var cstatus = status == null ? new Status() : status;
            var n = c_api.TF_GraphGetTensorNumDims(graph, output, cstatus);

            cstatus.Check();

            if (n == -1)
                return new TensorShape(new int[0]);

            var dims = new long[n];
            c_api.TF_GraphGetTensorShape(graph, output, dims, dims.Length, cstatus);
            cstatus.Check();
            return new TensorShape(dims.Select(x => (int)x).ToArray());
        }

        internal static (TF_DataType[] tfOutputTypes, DataViewType[] outputTypes, (Operation, int)[]) GetOutputInfo(IHost host, Session session, string[] outputs)
        {
            var tfOutputTypes = new TF_DataType[outputs.Length];
            var outputTypes = new DataViewType[outputs.Length];
            var newNames = new HashSet<string>();
            var tfOutputOperations = new (Operation, int)[outputs.Length];

            for (int i = 0; i < outputs.Length; i++)
            {
                host.CheckNonWhiteSpace(outputs[i], nameof(outputs));
                if (!newNames.Add(outputs[i]))
                    throw host.ExceptParam(nameof(outputs), $"Output column '{outputs[i]}' specified multiple times");

                (Tensor outputTensor, int outputIndex) = GetOperationFromName(outputs[i], session);
                if (outputTensor == null)
                    throw host.ExceptParam(nameof(outputs), $"Output column '{outputs[i]}' does not exist in the model");

                var tfOutputType = ((Operation)outputTensor).OutputType(outputIndex);
                var shape = GetTensorShape(new TF_Output((Operation)outputTensor, outputIndex), session.graph);

                // The transformer can only retreive the output as fixed length vector with shape of kind [-1, d1, d2, d3, ...]
                // i.e. the first dimension (if unknown) is assumed to be batch dimension.
                // If there are other dimension that are unknown the transformer will return a variable length vector.
                // This is the work around in absence of reshape transformer.
                var idims = shape.dims;
                int[] dims = shape.ndim > 0 ? idims.Skip(idims[0] == -1 ? 1 : 0).ToArray() : new[] { 0 };
                for (int j = 0; j < dims.Length; j++)
                    dims[j] = dims[j] == -1 ? 0 : dims[j];
                if (dims == null || dims.Length == 0)
                {
                    dims = new[] { 1 };
                    outputTypes[i] = DnnUtils.Tf2MlNetType(tfOutputType);
                }
                else
                {
                    var type = DnnUtils.Tf2MlNetType(tfOutputType);
                    outputTypes[i] = new VectorDataViewType(type, dims);
                }

                tfOutputTypes[i] = tfOutputType;
                tfOutputOperations[i] = (outputTensor, outputIndex);
            }

            return (tfOutputTypes, outputTypes, tfOutputOperations);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema) => new Mapper(this, inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // byte: indicator for frozen models
            // byte: indicator for adding batch dimension in input
            // stream: tensorFlow model.
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            var isFrozen = string.IsNullOrEmpty(_savedModelPath);
            ctx.Writer.WriteBoolByte(isFrozen);
            ctx.Writer.WriteBoolByte(_addBatchDimensionInput);
            if (isFrozen)
            {
                Status status = new Status();
                var buffer = Session.graph.ToGraphDef(status);
                ctx.SaveBinaryStream("TFModel", w =>
                {
                    w.WriteByteArray(buffer.MemoryBlock.ToArray());
                });
            }

            Host.AssertNonEmpty(Inputs);
            ctx.Writer.Write(Inputs.Length);
            foreach (var colName in Inputs)
                ctx.SaveNonEmptyString(colName);

            Host.AssertNonEmpty(Outputs);
            ctx.Writer.Write(Outputs.Length);
            foreach (var colName in Outputs)
                ctx.SaveNonEmptyString(colName);
        }

        ~TensorFlowTransformer()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            // Ensure that the Session is not null and it's handle is not Zero, as it may have already been disposed/finalized.
            // Technically we shouldn't be calling this if disposing == false, since we're running in finalizer
            // and the GC doesn't guarantee ordering of finalization of managed objects, but we have to make sure
            // that the Session is closed before deleting our temporary directory.
            try
            {
                if (Session != null && Session != IntPtr.Zero)
                {
                    Session.close(); // invoked Dispose()
                }
            }
            finally
            {
                if (!string.IsNullOrEmpty(_savedModelPath) && _isTemporarySavedModel)
                {
                    DnnUtils.DeleteFolderWithRetries(Host, _savedModelPath);
                }
            }
        }

        private sealed class Mapper : MapperBase
        {
            private readonly TensorFlowTransformer _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private readonly TensorShape[] _fullySpecifiedShapes;
            private readonly ConcurrentBag<Runner> _runners;

            public Mapper(TensorFlowTransformer parent, DataViewSchema inputSchema) :
                   base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                Host.CheckValue(parent, nameof(parent));
                _parent = parent;
                _inputColIndices = new int[_parent.Inputs.Length];
                _isInputVector = new bool[_parent.Inputs.Length];
                _fullySpecifiedShapes = new TensorShape[_parent.Inputs.Length];
                for (int i = 0; i < _parent.Inputs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.Inputs[i], out _inputColIndices[i]))
                        throw Host.ExceptSchemaMismatch(nameof(InputSchema), "source", _parent.Inputs[i]);

                    var type = inputSchema[_inputColIndices[i]].Type;
                    if (type is VectorDataViewType vecType && vecType.Size == 0)
                        throw Host.Except("Variable length input columns not supported");

                    _isInputVector[i] = type is VectorDataViewType;
                    if (!_isInputVector[i])
                        throw Host.Except("Non-vector columns are not supported and should be loaded as vector columns of size 1");
                    vecType = (VectorDataViewType)type;
                    var expectedType = DnnUtils.Tf2MlNetType(_parent.TFInputTypes[i]);
                    if (type.GetItemType() != expectedType)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.Inputs[i], expectedType.ToString(), type.ToString());
                    var originalShape = _parent.TFInputShapes[i];
                    var shape = originalShape.dims;

                    var colTypeDims = vecType.Dimensions.Select(dim => (int)dim).ToArray();
                    if (shape == null || (shape.Length == 0))
                        _fullySpecifiedShapes[i] = new TensorShape(colTypeDims);
                    else
                    {
                        // If the column is one dimension we make sure that the total size of the TF shape matches.
                        // Compute the total size of the known dimensions of the shape.
                        int valCount = 1;
                        int numOfUnkDim = 0;
                        foreach (var s in shape)
                        {
                            if (s > 0)
                                valCount *= s;
                            else
                                numOfUnkDim++;
                        }
                        // The column length should be divisible by this, so that the other dimensions can be integral.
                        int typeValueCount = type.GetValueCount();
                        if (typeValueCount % valCount != 0)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {typeValueCount}.");

                        // If the shape is multi-dimensional, we should be able to create the length of the vector by plugging
                        // in a single value for the unknown shapes. For example, if the shape is [?,?,3], then there should exist a value
                        // d such that d*d*3 is equal to the length of the input column.
                        var d = numOfUnkDim > 0 ? Math.Pow(typeValueCount / valCount, 1.0 / numOfUnkDim) : 0;
                        if (d - (int)d != 0)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {typeValueCount}.");

                        // Fill in the unknown dimensions.
                        var originalShapeNdim = originalShape.ndim;
                        var originalShapeDims = originalShape.dims;
                        var l = new int[originalShapeNdim];
                        for (int ishape = 0; ishape < originalShapeNdim; ishape++)
                            l[ishape] = originalShapeDims[ishape] == -1 ? (int)d : originalShapeDims[ishape];
                        _fullySpecifiedShapes[i] = new TensorShape(l);
                    }

                    if (_parent._addBatchDimensionInput)
                    {
                        var l = new int[_fullySpecifiedShapes[i].ndim + 1];
                        l[0] = 1;
                        for (int ishape = 1; ishape < l.Length; ishape++)
                            l[ishape] = _fullySpecifiedShapes[i].dims[ishape - 1];
                        _fullySpecifiedShapes[i] = new TensorShape(l);
                    }
                }

                _runners = new ConcurrentBag<Runner>();
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            private class OutputCache
            {
                public long Position;
                public Dictionary<string, Tensor> Outputs;
                public OutputCache()
                {
                    Position = -1;
                    Outputs = new Dictionary<string, Tensor>();
                }
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.AssertValue(input);

                var outputCache = new OutputCache();
                var activeOutputColNames = _parent.Outputs.Where((x, i) => activeOutput(i)).ToArray();

                var type = DnnUtils.Tf2MlNetType(_parent.TFOutputTypes[iinfo]).RawType;
                Host.Assert(type == _parent.OutputTypes[iinfo].GetItemType().RawType);
                var srcTensorGetters = GetTensorValueGetters(input, _inputColIndices, _isInputVector, _parent.TFInputTypes, _fullySpecifiedShapes);
                return Utils.MarshalInvoke(MakeGetter<int>, type, input, iinfo, srcTensorGetters, activeOutputColNames, outputCache);
            }

            private Delegate MakeGetter<T>(DataViewRow input, int iinfo, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache) where T : unmanaged
            {
                Host.AssertValue(input);

                if (_parent.OutputTypes[iinfo].IsStandardScalar())
                {
                    ValueGetter<T> valuegetter = (ref T dst) =>
                    {
                        UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                        var tensor = outputCache.Outputs[_parent.Outputs[iinfo]];
                        dst = tensor.ToArray<T>()[0];
                    };
                    return valuegetter;
                }
                else
                {
                    if (_parent.TFOutputTypes[iinfo] == TF_DataType.TF_STRING)
                    {
                        ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                        {
                            UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                            var tensor = outputCache.Outputs[_parent.Outputs[iinfo]];
                            var tensorSize = tensor.TensorShape.dims.Where(x => x > 0).Aggregate((x, y) => x * y);

                            var editor = VBufferEditor.Create(ref dst, (int)tensorSize);
                            DnnUtils.FetchStringData(tensor, editor.Values);
                            dst = editor.Commit();
                        };
                        return valuegetter;
                    }
                    else
                    {
                        ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                        {
                            UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                            var tensor = outputCache.Outputs[_parent.Outputs[iinfo]];
                            var tensorSize = tensor.TensorShape.dims.Where(x => x > 0).Aggregate((x, y) => x * y);

                            var editor = VBufferEditor.Create(ref dst, (int)tensorSize);

                            DnnUtils.FetchData<T>(tensor.ToArray<T>(), editor.Values);
                            dst = editor.Commit();
                        };
                        return valuegetter;
                    }
                }
            }

            private void UpdateCacheIfNeeded(long position, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                if (outputCache.Position != position)
                {
                    if (_parent.Graph.graph_key != tf.get_default_graph().graph_key)
                        _parent.Session.graph.as_default();
                    Runner runner = new Runner(_parent.Session);

                    // Feed inputs to the graph.
                    for (int i = 0; i < _parent.Inputs.Length; i++)
                    {
                        var tensor = srcTensorGetters[i].GetTensor();
                        runner.AddInput(_parent.Inputs[i], tensor);
                    }

                    // Add outputs.
                    for (int i = 0; i < _parent.Outputs.Length; i++)
                        runner.AddOutputs(_parent.Outputs[i]);

                    // Execute the graph.
                    var tensors = runner.Run();

                    Contracts.Assert(tensors.Length > 0);

                    for (int j = 0; j < activeOutputColNames.Length; j++)
                    {
                        if (outputCache.Outputs.TryGetValue(activeOutputColNames[j], out Tensor outTensor))
                            outTensor.Dispose();

                        outputCache.Outputs[activeOutputColNames[j]] = tensors[j];
                    }
                    outputCache.Position = position;
                }
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent.Outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                    info[i] = new DataViewSchema.DetachedColumn(_parent.Outputs[i], _parent.OutputTypes[i], null);
                return info;
            }
        }

        [TlcModule.EntryPoint(Name = "Transforms.TensorFlowScorer",
            Desc = Summary,
            UserName = UserName,
            ShortName = ShortName)]
        internal static CommonOutputs.TransformOutput TensorFlowScorer(IHostEnvironment env, TensorFlowEstimator.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "TensorFlow", input);
            var view = Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        private interface ITensorValueGetter
        {
            Tensor GetTensor();

            void BufferTrainingData();

            Tensor GetBufferedBatchTensor();
        }

        private class TensorValueGetter<T> : ITensorValueGetter
        {
            private readonly ValueGetter<T> _srcgetter;
            private readonly T[] _bufferedData;
            private readonly TensorShape _tfShape;
            private int _position;

            public TensorValueGetter(DataViewRow input, int colIndex, TensorShape tfShape)
            {
                _srcgetter = input.GetGetter<T>(input.Schema[colIndex]);
                _tfShape = tfShape;
                long size = 0;
                _position = 0;
                if (tfShape.dims.Length != 0)
                {
                    size = 1;
                    foreach (var dim in tfShape.dims)
                        size *= dim;
                }
                _bufferedData = new T[size];
            }

            public Tensor GetTensor()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                var tensor = new Tensor(new[] { scalar });
                tensor.set_shape(_tfShape);
                return tensor;
            }

            public void BufferTrainingData()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                _bufferedData[_position++] = scalar;
            }

            public Tensor GetBufferedBatchTensor()
            {
                var tensor = new Tensor(new NDArray(_bufferedData, _tfShape));
                _position = 0;
                return tensor;
            }
        }

        private class TensorValueGetterVec<T> : ITensorValueGetter
        {
            private readonly ValueGetter<VBuffer<T>> _srcgetter;
            private readonly TensorShape _tfShape;
            private VBuffer<T> _vBuffer;
            private T[] _denseData;
            private T[] _bufferedData;
            private int _position;
            private long[] _dims;
            private readonly long _bufferedDataSize;

            public TensorValueGetterVec(DataViewRow input, int colIndex, TensorShape tfShape)
            {
                _srcgetter = input.GetGetter<VBuffer<T>>(input.Schema[colIndex]);
                _tfShape = tfShape;
                _vBuffer = default;
                _denseData = default;

                long size = 0;
                _position = 0;
                if (tfShape.dims.Length != 0)
                {
                    size = 1;
                    foreach (var dim in tfShape.dims)
                        size *= dim;
                }
                _bufferedData = new T[size];
                if (_tfShape.dims != null)
                    _dims = _tfShape.dims.Select(x => (long)x).ToArray();
                _bufferedDataSize = size;
            }

            public Tensor GetTensor()
            {
                _srcgetter(ref _vBuffer);

                // _denseData.Length can be greater than _vBuffer.Length sometime after
                // Utils.EnsureSize is executed. Use _vBuffer.Length to access the elements in _denseData.
                // This is done to reduce memory allocation every time tensor is created.
                _denseData = new T[_vBuffer.Length];
                _vBuffer.CopyTo(_denseData);
                var tensor = CastDataAndReturnAsTensor(_denseData);
                return tensor;
            }

            private Tensor CastDataAndReturnAsTensor(T[] data)
            {
                if (typeof(T) == typeof(sbyte))
                    return new Tensor((sbyte[])(object)data, _dims, TF_DataType.TF_INT8);
                else if (typeof(T) == typeof(long))
                    return new Tensor((long[])(object)data, _dims, TF_DataType.TF_INT64);
                else if (typeof(T) == typeof(Int32))
                    return new Tensor((Int32[])(object)data, _dims, TF_DataType.TF_INT32);
                else if (typeof(T) == typeof(Int16))
                    return new Tensor((Int16[])(object)data, _dims, TF_DataType.TF_INT16);
                else if (typeof(T) == typeof(byte))
                    return new Tensor((byte[])(object)data, _dims, TF_DataType.TF_UINT8);
                else if (typeof(T) == typeof(ulong))
                    return new Tensor((ulong[])(object)data, _dims, TF_DataType.TF_UINT64);
                else if (typeof(T) == typeof(UInt32))
                    return new Tensor((UInt32[])(object)data, _dims, TF_DataType.TF_UINT32);
                else if (typeof(T) == typeof(UInt16))
                    return new Tensor((UInt16[])(object)data, _dims, TF_DataType.TF_UINT16);
                else if (typeof(T) == typeof(bool))
                    return new Tensor((bool[])(object)data, _dims, TF_DataType.TF_BOOL);
                else if (typeof(T) == typeof(float))
                    return new Tensor((float[])(object)data, _dims, TF_DataType.TF_FLOAT);
                else if (typeof(T) == typeof(double))
                    return new Tensor((double[])(object)data, _dims, TF_DataType.TF_DOUBLE);
                else if (typeof(T) == typeof(ReadOnlyMemory<char>))
                {
                    byte[][] bytes = new byte[_vBuffer.Length][];
                    for (int i = 0; i < bytes.Length; i++)
                    {
                        bytes[i] = Encoding.UTF8.GetBytes(((ReadOnlyMemory<char>)(object)data[i]).ToArray());
                    }

                    return new Tensor(bytes, _tfShape.dims.Select(x => (long)x).ToArray());
                }

                return new Tensor(new NDArray(data, _tfShape));
            }

            public void BufferTrainingData()
            {
                _srcgetter(ref _vBuffer);
                _vBuffer.CopyTo(_bufferedData, _position);
                _position += _vBuffer.Length;
            }

            public Tensor GetBufferedBatchTensor()
            {
                _position = 0;
                var tensor = CastDataAndReturnAsTensor(_bufferedData);

                _bufferedData = new T[_bufferedDataSize];
                return tensor;
            }
        }
    }
    /// <include file='doc.xml' path='doc/members/member[@name="TensorFlowTransfomer"]/*' />
    public sealed class TensorFlowEstimator : IEstimator<TensorFlowTransformer>
    {
        /// <summary>
        /// The options for the <see cref="TensorFlowTransformer"/>.
        /// </summary>
        internal sealed class Options : TransformInputBase
        {
            /// <summary>
            /// Location of the TensorFlow model.
            /// </summary>
            [Argument(ArgumentType.Required, HelpText = "TensorFlow model used by the transform. Please see https://www.tensorflow.org/mobile/prepare_models for more details.", SortOrder = 0)]
            public string ModelLocation;

            /// <summary>
            /// The names of the model inputs.
            /// </summary>
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string[] InputColumns;

            /// <summary>
            /// The names of the requested model outputs.
            /// </summary>
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the outputs", ShortName = "outputs", SortOrder = 2)]
            public string[] OutputColumns;

            /// <summary>
            /// Number of samples to use for mini-batch training.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of samples to use for mini-batch training.", SortOrder = 9)]
            public int BatchSize = 64;

            /// <summary>
            /// Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
            /// </summary>
            /// <remarks>
            /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.
            /// In this case, there is no way to induce shape from the model's inputs or input data.
            /// </remarks>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].", SortOrder = 16)]
            public bool AddBatchDimensionInputs = false;
        }

        private readonly IHost _host;
        private readonly Options _options;
        private readonly TensorFlowModel _tensorFlowModel;
        private readonly TF_DataType[] _tfInputTypes;
        private readonly DataViewType[] _outputTypes;
        private TensorFlowTransformer _transformer;

        [BestFriend]
        internal TensorFlowEstimator(IHostEnvironment env, string[] outputColumnNames, string[] inputColumnNames, string modelLocation, bool addBatchDimensionInput)
            : this(env, outputColumnNames, inputColumnNames, TensorFlowUtils.LoadTensorFlowModel(env, modelLocation), addBatchDimensionInput)
        {
        }

        internal TensorFlowEstimator(IHostEnvironment env, string[] outputColumnNames, string[] inputColumnNames, TensorFlowModel tensorFlowModel, bool addBatchDimensionInput)
            : this(env, CreateArguments(tensorFlowModel, outputColumnNames, inputColumnNames, addBatchDimensionInput), tensorFlowModel)
        {
        }

        internal TensorFlowEstimator(IHostEnvironment env, Options options)
            : this(env, options, TensorFlowUtils.LoadTensorFlowModel(env, options.ModelLocation))
        {
        }

        internal TensorFlowEstimator(IHostEnvironment env, Options options, TensorFlowModel tensorFlowModel)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(TensorFlowEstimator));
            _options = options;
            _tensorFlowModel = tensorFlowModel;
            var inputTuple = TensorFlowTransformer.GetInputInfo(_host, tensorFlowModel.Session, options.InputColumns);
            _tfInputTypes = inputTuple.tfInputTypes;
            var outputTuple = TensorFlowTransformer.GetOutputInfo(_host, tensorFlowModel.Session, options.OutputColumns);
            _outputTypes = outputTuple.outputTypes;
        }

        private static Options CreateArguments(TensorFlowModel tensorFlowModel, string[] outputColumnNames, string[] inputColumnName, bool addBatchDimensionInput)
        {
            var options = new Options();
            options.ModelLocation = tensorFlowModel.ModelPath;
            options.InputColumns = inputColumnName;
            options.OutputColumns = outputColumnNames;
            options.AddBatchDimensionInputs = addBatchDimensionInput;
            return options;
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            var resultDic = inputSchema.ToDictionary(x => x.Name);
            for (var i = 0; i < _options.InputColumns.Length; i++)
            {
                var input = _options.InputColumns[i];
                if (!inputSchema.TryFindColumn(input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
                if (!(col.Kind == SchemaShape.Column.VectorKind.Vector))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "vector", col.GetTypeString());
                var expectedType = DnnUtils.Tf2MlNetType(_tfInputTypes[i]);
                if (col.ItemType != expectedType)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, expectedType.ToString(), col.ItemType.ToString());
            }
            for (var i = 0; i < _options.OutputColumns.Length; i++)
            {
                resultDic[_options.OutputColumns[i]] = new SchemaShape.Column(_options.OutputColumns[i],
                    _outputTypes[i].IsKnownSizeVector() ? SchemaShape.Column.VectorKind.Vector
                    : SchemaShape.Column.VectorKind.VariableVector, _outputTypes[i].GetItemType(), false);
            }
            return new SchemaShape(resultDic.Values);
        }

        /// <summary>
        /// Trains and returns a <see cref="TensorFlowTransformer"/>.
        /// </summary>
        public TensorFlowTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            if (_transformer == null)
            {
                _transformer = new TensorFlowTransformer(_host, _tensorFlowModel.Session, _options.OutputColumns, _options.InputColumns,
                    DnnUtils.IsSavedModel(_host, _options.ModelLocation) ? _options.ModelLocation : null, false, _options.AddBatchDimensionInputs);
            }
            // Validate input schema.
            _transformer.GetOutputSchema(input.Schema);
            return _transformer;
        }
    }
}
