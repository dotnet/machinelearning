﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using System.Security.AccessControl;
using System.Security.Principal;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TensorFlow;
using Microsoft.ML.Transforms;
using Tensorflow;
using static Tensorflow.Binding;

namespace Microsoft.ML.TensorFlow
{
    internal static class TensorFlowUtils
    {
        /// <summary>
        /// Key to access operator's type (a string) in <see cref="DataViewSchema.Column.Annotations"/>.
        /// Its value describes the Tensorflow operator that produces this <see cref="DataViewSchema.Column"/>.
        /// </summary>
        internal const string TensorflowOperatorTypeKind = "TensorflowOperatorType";
        /// <summary>
        /// Key to access upstream operators' names (a string array) in <see cref="DataViewSchema.Column.Annotations"/>.
        /// Its value states operators that the associated <see cref="DataViewSchema.Column"/>'s generator depends on.
        /// </summary>
        internal const string TensorflowUpstreamOperatorsKind = "TensorflowUpstreamOperators";

        internal static DataViewSchema GetModelSchema(IExceptionContext ectx, Graph graph, string opType = null)
        {
            var schemaBuilder = new DataViewSchema.Builder();
            foreach (Operation op in graph)
            {
                if (opType != null && opType != op.OpType)
                    continue;

                var tfType = op.OutputType(0);
                // Determine element type in Tensorflow tensor. For example, a vector of floats may get NumberType.R4 here.
                var mlType = Tf2MlNetTypeOrNull(tfType);

                // If the type is not supported in ML.NET then we cannot represent it as a column in an Schema.
                // We also cannot output it with a TensorFlowTransform, so we skip it.
                // Furthermore, operators which have NumOutputs <= 0 needs to be filtered.
                // The 'GetTensorShape' method crashes TensorFlow runtime
                // (https://github.com/dotnet/machinelearning/issues/2156) when the operator has no outputs.
                if (mlType == null || op.NumOutputs <= 0)
                    continue;

                // Construct the final ML.NET type of a Tensorflow variable.
                var tensorShape = op.output.TensorShape.dims;
                var columnType = new VectorDataViewType(mlType);
                if (!(Utils.Size(tensorShape) == 1 && tensorShape[0] <= 0) &&
                    (Utils.Size(tensorShape) > 0 && tensorShape.Skip(1).All(x => x > 0)))
                    columnType = new VectorDataViewType(mlType, tensorShape[0] > 0 ? tensorShape : tensorShape.Skip(1).ToArray());

                // There can be at most two metadata fields.
                //  1. The first field always presents. Its value is this operator's type. For example,
                //     if an output is produced by an "Softmax" operator, the value of this field should be "Softmax".
                //  2. The second field stores operators whose outputs are consumed by this operator. In other words,
                //     these values are names of some upstream operators which should be evaluated before executing
                //     the current operator. It's possible that one operator doesn't need any input, so this field
                //     can be missing.
                var metadataBuilder = new DataViewSchema.Annotations.Builder();
                // Create the first metadata field.
                metadataBuilder.Add(TensorflowOperatorTypeKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => value = op.OpType.AsMemory());
                if (op.NumInputs > 0)
                {
                    // Put upstream operators' names to an array (type: VBuffer) of string (type: ReadOnlyMemory<char>).
                    VBuffer<ReadOnlyMemory<char>> upstreamOperatorNames = default;
                    var bufferEditor = VBufferEditor.Create(ref upstreamOperatorNames, op.NumInputs);
                    for (int i = 0; i < op.NumInputs; ++i)
                        bufferEditor.Values[i] = op.inputs[i].op.name.AsMemory();
                    upstreamOperatorNames = bufferEditor.Commit(); // Used in metadata's getter.

                    // Create the second metadata field.
                    metadataBuilder.Add(TensorflowUpstreamOperatorsKind, new VectorDataViewType(TextDataViewType.Instance, op.NumInputs),
                        (ref VBuffer<ReadOnlyMemory<char>> value) => { upstreamOperatorNames.CopyTo(ref value); });
                }

                schemaBuilder.AddColumn(op.name, columnType, metadataBuilder.ToAnnotations());
            }
            return schemaBuilder.ToSchema();
        }

        /// <summary>
        /// This method retrieves the information about the graph nodes of a TensorFlow model as an <see cref="DataViewSchema"/>.
        /// For every node in the graph that has an output type that is compatible with the types supported by
        /// <see cref="TensorFlowTransformer"/>, the output schema contains a column with the name of that node, and the
        /// type of its output (including the item type and the shape, if it is known). Every column also contains metadata
        /// of kind <see cref="TensorflowOperatorTypeKind"/>, indicating the operation type of the node, and if that node has inputs in the graph,
        /// it contains metadata of kind <see cref="TensorflowUpstreamOperatorsKind"/>, indicating the names of the input nodes.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">Model to load.</param>
        internal static DataViewSchema GetModelSchema(IHostEnvironment env, string modelPath)
        {
            var model = LoadTensorFlowModel(env, modelPath);
            return GetModelSchema(env, model.Session.graph);
        }

        /// <summary>
        /// Load TensorFlow model into memory.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">The model to load.</param>
        /// <returns></returns>
        internal static TensorFlowModel LoadTensorFlowModel(IHostEnvironment env, string modelPath)
        {
            var session = GetSession(env, modelPath);
            return new TensorFlowModel(env, session, modelPath);
        }

        internal static PrimitiveDataViewType Tf2MlNetType(TF_DataType type)
        {
            var mlNetType = Tf2MlNetTypeOrNull(type);
            if (mlNetType == null)
                throw new NotSupportedException("TensorFlow type not supported.");
            return mlNetType;
        }

        internal static PrimitiveDataViewType Tf2MlNetTypeOrNull(TF_DataType type)
        {
            switch (type)
            {
                case TF_DataType.TF_FLOAT:
                    return NumberDataViewType.Single;
                case TF_DataType.DtFloatRef:
                    return NumberDataViewType.Single;
                case TF_DataType.TF_DOUBLE:
                    return NumberDataViewType.Double;
                case TF_DataType.TF_UINT8:
                    return NumberDataViewType.Byte;
                case TF_DataType.TF_UINT16:
                    return NumberDataViewType.UInt16;
                case TF_DataType.TF_UINT32:
                    return NumberDataViewType.UInt32;
                case TF_DataType.TF_UINT64:
                    return NumberDataViewType.UInt64;
                case TF_DataType.TF_INT8:
                    return NumberDataViewType.SByte;
                case TF_DataType.TF_INT16:
                    return NumberDataViewType.Int16;
                case TF_DataType.TF_INT32:
                    return NumberDataViewType.Int32;
                case TF_DataType.TF_INT64:
                    return NumberDataViewType.Int64;
                case TF_DataType.TF_BOOL:
                    return BooleanDataViewType.Instance;
                case TF_DataType.TF_STRING:
                    return TextDataViewType.Instance;
                default:
                    return null;
            }
        }

        internal static Session LoadTFSession(IExceptionContext ectx, byte[] modelBytes, string modelFile = null)
        {
            var graph = new Graph();
            try
            {
                graph.Import(modelBytes, "");
            }
            catch (Exception ex)
            {
                if (!string.IsNullOrEmpty(modelFile))
                    throw ectx.Except($"TensorFlow exception triggered while loading model from '{modelFile}'");
#pragma warning disable MSML_NoMessagesForLoadContext
                throw ectx.ExceptDecode(ex, "Tensorflow exception triggered while loading model.");
#pragma warning restore MSML_NoMessagesForLoadContext

            }
            return new Session(graph);
        }

        internal static void DownloadIfNeeded(IHostEnvironment env, string url, string dir, string fileName, int timeout)
        {
            using (var ch = env.Start("Ensuring meta files are present."))
            {
                var ensureModel = ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, url, fileName, dir, timeout);
                ensureModel.Wait();
                var errorResult = ResourceManagerUtils.GetErrorMessage(out var errorMessage, ensureModel.Result);
                if (errorResult != null)
                {
                    var directory = Path.GetDirectoryName(errorResult.FileName);
                    var name = Path.GetFileName(errorResult.FileName);
                    throw ch.Except($"{errorMessage}\nMeta file could not be downloaded! " +
                        $@"Please copy the model file '{name}' from '{url}' to '{directory}'.");
                }
            }
        }

        internal static Graph LoadMetaGraph(string path)
        {
            var graph = new Graph();
            graph = graph.as_default();
            tf.train.import_meta_graph(path);
            return graph;
        }

        internal static Session LoadTFSessionByModelFilePath(IExceptionContext ectx, string modelFile, bool metaGraph = false)
        {
            if (string.IsNullOrEmpty(modelFile))
                throw ectx.Except($"TensorFlow exception triggered while loading model from '{modelFile}'");

            Graph graph;
            try
            {
                if (metaGraph)
                    graph = LoadMetaGraph(modelFile);
                else
                {
                    graph = new Graph();
                    graph.Import(modelFile, "");
                }
            }
            catch (Exception ex)
            {
#pragma warning disable MSML_NoMessagesForLoadContext
                throw ectx.ExceptDecode(ex, "Tensorflow exception triggered while loading model.");
#pragma warning restore MSML_NoMessagesForLoadContext

            }
            return new Session(graph);
        }

        private static Session LoadTFSession(IHostEnvironment env, string exportDirSavedModel)
        {
            Contracts.Check(env != null, nameof(env));
            env.CheckValue(exportDirSavedModel, nameof(exportDirSavedModel));
            return Session.LoadFromSavedModel(exportDirSavedModel);
        }

        // A TensorFlow frozen model is a single file. An un-frozen (SavedModel) on the other hand has a well-defined folder structure.
        // Given a modelPath, this utility method determines if we should treat it as a SavedModel or not
        internal static bool IsSavedModel(IHostEnvironment env, string modelPath)
        {
            Contracts.Check(env != null, nameof(env));
            env.CheckNonWhiteSpace(modelPath, nameof(modelPath));
            FileAttributes attr = File.GetAttributes(modelPath);
            return attr.HasFlag(FileAttributes.Directory);
        }

        // Currently used in TensorFlowTransform to protect temporary folders used when working with TensorFlow's SavedModel format.
        // Models are considered executable code, so we need to ACL tthe temp folders for high-rights process (so low-rights process can’t access it).
        /// <summary>
        ///  Given a folder path, create it with proper ACL if it doesn't exist.
        ///  Fails if the folder name is empty, or can't create the folder.
        /// </summary>
        internal static void CreateFolderWithAclIfNotExists(IHostEnvironment env, string folder)
        {
            Contracts.Check(env != null, nameof(env));
            env.CheckNonWhiteSpace(folder, nameof(folder));

            //if directory exists, do nothing.
            if (Directory.Exists(folder))
                return;

            WindowsIdentity currentIdentity = null;
            try
            {
                currentIdentity = WindowsIdentity.GetCurrent();
            }
            catch (PlatformNotSupportedException)
            { }

            if (currentIdentity != null && new WindowsPrincipal(currentIdentity).IsInRole(WindowsBuiltInRole.Administrator))
            {
                // Create high integrity dir and set no delete policy for all files under the directory.
                // In case of failure, throw exception.
                CreateTempDirectoryWithAcl(folder, currentIdentity.User.ToString());
            }
            else
            {
                try
                {
                    Directory.CreateDirectory(folder);
                }
                catch (Exception exc)
                {
                    throw Contracts.ExceptParam(nameof(folder), $"Failed to create folder for the provided path: {folder}. \nException: {exc.Message}");
                }
            }
        }

        internal static void DeleteFolderWithRetries(IHostEnvironment env, string folder)
        {
            Contracts.Check(env != null, nameof(env));
            int currentRetry = 0;
            int maxRetryCount = 10;
            using (var ch = env.Start("Delete folder"))
            {
                for (; ; )
                {
                    try
                    {
                        currentRetry++;
                        Directory.Delete(folder, true);
                        break;
                    }
                    catch (IOException e)
                    {
                        if (currentRetry > maxRetryCount)
                            throw;
                        ch.Info("Error deleting folder. {0}. Retry,", e.Message);
                    }
                }
            }
        }

        private static void CreateTempDirectoryWithAcl(string folder, string identity)
        {
            // Dacl Sddl string:
            // D: Dacl type
            // D; Deny access
            // OI; Object inherit ace
            // SD; Standard delete function
            // wIdentity.User Sid of the given user.
            // A; Allow access
            // OICI; Object inherit, container inherit
            // FA File access
            // BA Built-in administrators
            // S: Sacl type
            // ML;; Mandatory Label
            // NW;;; No write policy
            // HI High integrity processes only
            string sddl = "D:(D;OI;SD;;;" + identity + ")(A;OICI;FA;;;BA)S:(ML;OI;NW;;;HI)";

            try
            {
                var dir = Directory.CreateDirectory(folder);
                DirectorySecurity dirSec = new DirectorySecurity();
                dirSec.SetSecurityDescriptorSddlForm(sddl);
                dirSec.SetAccessRuleProtection(true, false);  // disable inheritance
                dir.SetAccessControl(dirSec);

                // Cleaning out the directory, in case someone managed to sneak in between creation and setting ACL.
                DirectoryInfo dirInfo = new DirectoryInfo(folder);
                foreach (FileInfo file in dirInfo.GetFiles())
                {
                    file.Delete();
                }
                foreach (DirectoryInfo subDirInfo in dirInfo.GetDirectories())
                {
                    subDirInfo.Delete(true);
                }
            }
            catch (Exception exc)
            {
                throw Contracts.ExceptParam(nameof(folder), $"Failed to create folder for the provided path: {folder}. \nException: {exc.Message}");
            }
        }

        /// <summary>
        /// Load TensorFlow model into memory.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">The model to load.</param>
        /// <param name="metaGraph"></param>
        /// <returns></returns>
        internal static TensorFlowSessionWrapper LoadDnnModel(IHostEnvironment env, string modelPath, bool metaGraph = false) =>
            new TensorFlowSessionWrapper(GetSession(env, modelPath, metaGraph), modelPath);

        internal static Session GetSession(IHostEnvironment env, string modelPath, bool metaGraph = false)
        {
            Contracts.Check(env != null, nameof(env));
            if (IsSavedModel(env, modelPath))
            {
                env.CheckUserArg(Directory.Exists(modelPath), nameof(modelPath));
                return LoadTFSession(env, modelPath);
            }

            env.CheckUserArg(File.Exists(modelPath), nameof(modelPath));
            return LoadTFSessionByModelFilePath(env, modelPath, metaGraph);
        }

        internal static unsafe void FetchStringData<T>(Tensor tensor, Span<T> result)
        {
            if (tensor == null)
                throw Contracts.ExceptEmpty(nameof(tensor));

            var buffer = tensor.StringData();

            for (int i = 0; i < buffer.Length; i++)
                result[i] = (T)(object)buffer[i].AsMemory();
        }

        internal static bool IsTypeSupported(TF_DataType tfoutput)
        {
            switch (tfoutput)
            {
                case TF_DataType.TF_FLOAT:
                case TF_DataType.TF_DOUBLE:
                case TF_DataType.TF_UINT8:
                case TF_DataType.TF_UINT16:
                case TF_DataType.TF_UINT32:
                case TF_DataType.TF_UINT64:
                case TF_DataType.TF_INT8:
                case TF_DataType.TF_INT16:
                case TF_DataType.TF_INT32:
                case TF_DataType.TF_INT64:
                case TF_DataType.TF_BOOL:
                case TF_DataType.TF_STRING:
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        /// Use the runner class to easily configure inputs, outputs and targets to be passed to the session runner.
        /// </summary>
        public class Runner : IDisposable
        {
            private TF_Output[] _inputs;
            private TF_Output[] _outputs;
            private IntPtr[] _outputValues;
            private IntPtr[] _inputValues;
            private Tensor[] _inputTensors;
            private IntPtr[] _operations;
            private Session _session;
            private Tensor[] _outputTensors;
            private Status _status;

            internal Runner(Session session, TF_Output[] inputs = null, TF_Output[] outputs = null, IntPtr[] operations = null)
            {
                _session = session;
                _inputs = inputs ?? new TF_Output[0];
                _outputs = outputs ?? new TF_Output[0];
                _operations = operations ?? new IntPtr[0];
                _inputValues = new IntPtr[_inputs.Length];
                _inputTensors = new Tensor[_inputs.Length];
                _outputValues = new IntPtr[_outputs.Length];
                _outputTensors = new Tensor[_outputs.Length];
                _status = new Status();
            }

            internal Runner(Session session, string[] inputs = null, string[] outputs = null, string[] operations = null)
            {
                _session = session;
                _inputs = inputs?.Select(x => ParseOutput(session, x)).ToArray() ?? new TF_Output[0];
                _outputs = outputs?.Select(x => ParseOutput(session, x)).ToArray() ?? new TF_Output[0];
                _operations = operations?.Select(x => c_api.TF_GraphOperationByName(session.graph, x)).ToArray() ?? new IntPtr[0];
                _inputValues = new IntPtr[_inputs.Length];
                _inputTensors = new Tensor[_inputs.Length];
                _outputValues = new IntPtr[_outputs.Length];
                _outputTensors = new Tensor[_outputs.Length];
                _status = new Status();
            }

            public Runner AddInput(Tensor value, int index)
            {
                _inputTensors[index]?.Dispose();
                _inputTensors[index] = value;
                _inputValues[index] = value;
                return this;
            }

            // Parses user strings that contain both the operation name and an index.
            public static TF_Output ParseOutput(Session session, string operation)
            {
                var p = operation.IndexOf(':');
                if (p != -1 && p != operation.Length - 1)
                {
                    var op = operation.Substring(0, p);
                    if (int.TryParse(operation.Substring(p + 1), out var idx))
                    {
                        return new TF_Output(session.graph.OperationByName(op), idx);
                    }
                }
                return new TF_Output(session.graph.OperationByName(operation), 0);
            }

            /// <summary>
            /// Executes a pipeline given the specified inputs, inputValues, outputs, targetOpers, runMetadata and runOptions.
            /// A simpler API is available by calling the <see cref="M:GetRunner"/> method which performs all the bookkeeping
            /// necessary.
            /// </summary>
            /// <returns>An array of tensors fetched from the requested outputs.</returns>
            public Tensor[] Run()
            {
                if (_session == IntPtr.Zero)
                    new ObjectDisposedException(nameof(_session));

                unsafe
                {
                    try
                    {
                        c_api.TF_SessionRun(_session, null, _inputs, _inputValues,
                             _inputs.Length, _outputs, _outputValues, _outputValues.Length, _operations,
                            _operations.Length, IntPtr.Zero, _status);
                    }
                    catch (Exception ex)
                    {
                        try
                        {
                            _status.Check(throwException: true);
                        }
                        catch (Exception statusException)
                        {
                            throw new AggregateException(statusException, ex);
                        }

                        // _status didn't provide more information, so just rethrow the original exception
                        throw;
                    }
                }

                _status.Check(true);

                for (int i = 0; i < _outputs.Length; i++)
                    _outputTensors[i] = new Tensor(_outputValues[i]);

                return _outputTensors;
            }

            public void Dispose()
            {
                foreach (var tensor in _inputTensors)
                {
                    if (!tensor.IsDisposed)
                        tensor.Dispose();
                }

                _status.Dispose();
            }
        }

        internal static string GetTemporaryDirectory()
        {
            string tempDirectory = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
            Directory.CreateDirectory(tempDirectory);
            return tempDirectory;
        }
    }
}
