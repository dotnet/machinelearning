// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.AccessControl;
using System.Security.Principal;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Tensorflow;
using static Tensorflow.Binding;

namespace Microsoft.ML.Transforms.Dnn
{
    internal static class DnnUtils
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
        internal static DnnModel LoadDnnModel(IHostEnvironment env, string modelPath, bool metaGraph = false)
        {
            var session = GetSession(env, modelPath, metaGraph);
            return new DnnModel(env, session, modelPath);
        }

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
        /// <remarks>
        /// <para>
        /// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
        /// to construct the parameters that will be passed to the TFSession.Run method.
        /// </para>
        /// <para>
        /// Instances of this class are created by calling the GetRunner method on the TFSession.
        /// </para>
        /// <para>
        /// The various methods in this class return an instance to the Runner itsel, to allow
        /// to easily construct chains of execution like this:
        /// </para>
        /// <code>
        /// var result = session.GetRunner ().AddINput (myInput).Fetch (MyOutput).Run ();
        /// </code>
        /// <para>
        /// You do not need to chain the operations, this works just the same:
        /// </para>
        /// <code>
        /// runner = session.GetRunner ();
        /// runner.AddInput(myInput);
        /// runner.Fetch(myOutput);
        /// var results = runner.Run();
        /// </code>
        /// </remarks>
        public class Runner
        {
            private List<TF_Output> _inputs;
            private List<TF_Output> _outputs;
            private List<Tensor> _inputValues;
            private List<Operation> _operations;
            private Session _session;

            internal Runner(Session session)
            {
                _session = session;
                _inputs = new List<TF_Output>();
                _outputs = new List<TF_Output>();
                _inputValues = new List<Tensor>();
                _operations = new List<Operation>();
            }

            /// <summary>
            /// Adds an input to the session specified by name, with an optional index in the operation (separated by a colon).
            /// </summary>
            /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
            /// <param name="input">Incoming port, with an optional index separated by a colon.</param>
            /// <param name="value">Value to assing to the incoming port.</param>
            public Runner AddInput(string input, Tensor value)
            {
                if (value == null)
                    throw new ArgumentNullException(nameof(value));

                _inputs.Add(ParseOutput(input));
                _inputValues.Add(value);

                return this;
            }

            public Runner AddInput(string input)
            {
                _inputs.Add(ParseOutput(input));
                return this;
            }

            public Runner AddInput(Tensor value, int index)
            {
                if (_inputValues.Count <= index)
                    _inputValues.Add(value);
                else
                {
                    _inputValues[index].Dispose();
                    _inputValues[index] = value;
                }

                return this;
            }

            public List<Tensor> GetInputValues()
            {
                return _inputValues;
            }

            public Runner AddOutputs(string output)
            {
                _outputs.Add(ParseOutput(output));
                return this;
            }

            public Runner AddOperation(string operationName)
            {
                _operations.Add(c_api.TF_GraphOperationByName(_session.graph, operationName));
                return this;
            }

            public Runner AddOperation(Operation operation)
            {
                _operations.Add(operation);
                return this;
            }

            // Parses user strings that contain both the operation name and an index.
            private TF_Output ParseOutput(string operation)
            {
                var p = operation.IndexOf(':');
                if (p != -1 && p != operation.Length - 1)
                {
                    var op = operation.Substring(0, p);
                    if (int.TryParse(operation.Substring(p + 1), out var idx))
                    {
                        return new TF_Output(_session.graph.OperationByName(op) , idx);
                    }
                }
                return new TF_Output(_session.graph.OperationByName(operation), 0);
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

                int oLen = _outputs != null ? _outputs.Count : 0;
                var cstatus = new Status();
                var ovals = _outputs != null ? new IntPtr[_outputs.Count] : null;
                unsafe
                {
                    c_api.TF_SessionRun(_session, null, _inputs.ToArray(), _inputValues.Select(x => (IntPtr)x).ToArray(),
                        _inputs != null ? _inputs.Count : 0, _outputs.ToArray(), ovals, oLen, _operations.Select(x => (IntPtr)x).ToArray(),
                        _operations == null ? 0 : _operations.Count, IntPtr.Zero, cstatus);
                }

                cstatus.Check(true);

                var result = new Tensor[oLen];
                for (int i = 0; i < oLen; i++)
                    result[i] = new Tensor(ovals[i]);

                return result;
            }

        }

    }
}
