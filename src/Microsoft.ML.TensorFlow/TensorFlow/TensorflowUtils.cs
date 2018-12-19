﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.AccessControl;
using System.Security.Principal;

namespace Microsoft.ML.Transforms.TensorFlow
{
    public static class TensorFlowUtils
    {
        public const string OpType = "OpType";
        public const string InputOps = "InputOps";

        internal static Schema GetModelSchema(IExceptionContext ectx, TFGraph graph, string opType = null)
        {
            var res = new List<KeyValuePair<string, ColumnType>>();
            var opTypeGetters = new List<MetadataUtils.MetadataGetter<ReadOnlyMemory<char>>>();
            var inputOpsGetters = new List<MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>>>();
            var inputOpsLengths = new List<int>();
            foreach (var op in graph)
            {
                if (opType != null && opType != op.OpType)
                    continue;
                var tfType = op[0].OutputType;
                var mlType = Tf2MlNetTypeOrNull(tfType);

                // If the type is not supported in ML.NET then we cannot represent it as a column in an ISchema.
                // We also cannot output it with a TensorFlowTransform, so we skip it.
                if (mlType == null)
                    continue;

                var shape = graph.GetTensorShape(op[0]);
                var shapeArray = shape.ToIntArray();

                inputOpsLengths.Add(op.NumInputs);
                MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>> inputOpsGetter = null;
                if (op.NumInputs > 0)
                {
                    var inputOps = new ReadOnlyMemory<char>[op.NumInputs];
                    for (int i = 0; i < op.NumInputs; i++)
                    {
                        var input = op.GetInput(i);
                        inputOps[i] = new ReadOnlyMemory<char>(input.Operation.Name.ToArray());
                    }
                    inputOpsGetter = (int col, ref VBuffer<ReadOnlyMemory<char>> dst) =>
                        dst = new VBuffer<ReadOnlyMemory<char>>(op.NumInputs, inputOps);
                }
                inputOpsGetters.Add(inputOpsGetter);

                MetadataUtils.MetadataGetter<ReadOnlyMemory<char>> opTypeGetter =
                    (int col, ref ReadOnlyMemory<char> dst) => dst = new ReadOnlyMemory<char>(op.OpType.ToArray());
                opTypeGetters.Add(opTypeGetter);

                var columnType = Utils.Size(shapeArray) == 1 && shapeArray[0] <= 0 ? new VectorType(mlType) :
                    Utils.Size(shapeArray) > 0 && shapeArray.Skip(1).All(x => x > 0) ?
                        new VectorType(mlType, shapeArray[0] > 0 ? shapeArray : shapeArray.Skip(1).ToArray())
                        : new VectorType(mlType);
                res.Add(new KeyValuePair<string, ColumnType>(op.Name, columnType));
            }
            return Schema.Create(new TensorFlowSchema(ectx, res.ToArray(), opTypeGetters.ToArray(), inputOpsGetters.ToArray(), inputOpsLengths.ToArray()));
        }

        /// <summary>
        /// This method retrieves the information about the graph nodes of a TensorFlow model as an <see cref="ISchema"/>.
        /// For every node in the graph that has an output type that is compatible with the types supported by
        /// <see cref="TensorFlowTransform"/>, the output schema contains a column with the name of that node, and the
        /// type of its output (including the item type and the shape, if it is known). Every column also contains metadata
        /// of kind <see cref="OpType"/>, indicating the operation type of the node, and if that node has inputs in the graph,
        /// it contains metadata of kind <see cref="InputOps"/>, indicating the names of the input nodes.
        /// </summary>
        /// <param name="ectx">An <see cref="IExceptionContext"/>.</param>
        /// <param name="modelFile">The name of the file containing the TensorFlow model. Currently only frozen model
        /// format is supported.</param>
        public static Schema GetModelSchema(IExceptionContext ectx, string modelFile)
        {
            var bytes = File.ReadAllBytes(modelFile);
            var session = LoadTFSession(ectx, bytes, modelFile);
            return GetModelSchema(ectx, session.Graph);
        }

        /// <summary>
        /// This is a convenience method for iterating over the nodes of a TensorFlow model graph. It
        /// iterates over the columns of the <see cref="ISchema"/> returned by <see cref="GetModelSchema(IExceptionContext, string)"/>,
        /// and for each one it returns a tuple containing the name, operation type, column type and an array of input node names.
        /// This method is convenient for filtering nodes based on certain criteria, for example, by the operation type.
        /// </summary>
        /// <param name="modelFile"></param>
        /// <returns></returns>
        public static IEnumerable<(string, string, ColumnType, string[])> GetModelNodes(string modelFile)
        {
            var schema = GetModelSchema(null, modelFile);

            for (int i = 0; i < schema.Count; i++)
            {
                var name = schema[i].Name;
                var type = schema[i].Type;

                var metadataType = schema[i].Metadata.Schema.GetColumnOrNull(TensorFlowUtils.OpType)?.Type;
                Contracts.Assert(metadataType != null && metadataType.IsText);
                ReadOnlyMemory<char> opType = default;
                schema[i].Metadata.GetValue(TensorFlowUtils.OpType, ref opType);
                metadataType = schema[i].Metadata.Schema.GetColumnOrNull(TensorFlowUtils.InputOps)?.Type;
                VBuffer<ReadOnlyMemory<char>> inputOps = default;
                if (metadataType != null)
                {
                    Contracts.Assert(metadataType.IsKnownSizeVector && metadataType.ItemType.IsText);
                    schema[i].Metadata.GetValue(TensorFlowUtils.InputOps, ref inputOps);
                }

                string[] inputOpsResult = inputOps.DenseValues()
                    .Select(input => input.ToString())
                    .ToArray();

                yield return (name, opType.ToString(), type, inputOpsResult);
            }
        }

        internal static PrimitiveType Tf2MlNetType(TFDataType type)
        {
            var mlNetType = Tf2MlNetTypeOrNull(type);
            if (mlNetType == null)
                throw new NotSupportedException("TensorFlow type not supported.");
            return mlNetType;
        }

        private static PrimitiveType Tf2MlNetTypeOrNull(TFDataType type)
        {
            switch (type)
            {
                case TFDataType.Float:
                    return NumberType.R4;
                case TFDataType.Float_ref:
                    return NumberType.R4;
                case TFDataType.Double:
                    return NumberType.R8;
                case TFDataType.UInt16:
                    return NumberType.U2;
                case TFDataType.UInt8:
                    return NumberType.U1;
                case TFDataType.UInt32:
                    return NumberType.U4;
                case TFDataType.UInt64:
                    return NumberType.U8;
                case TFDataType.Int16:
                    return NumberType.I2;
                case TFDataType.Int32:
                    return NumberType.I4;
                case TFDataType.Int64:
                    return NumberType.I8;
                default:
                    return null;
            }
        }

        internal static TFSession LoadTFSession(IExceptionContext ectx, byte[] modelBytes, string modelFile = null)
        {
            var graph = new TFGraph();
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
            return new TFSession(graph);
        }

        private static TFSession LoadTFSession(IHostEnvironment env, string exportDirSavedModel)
        {
            Contracts.Check(env != null, nameof(env));
            env.CheckValue(exportDirSavedModel, nameof(exportDirSavedModel));
            var sessionOptions = new TFSessionOptions();
            var tags = new string[] { "serve" };
            var graph = new TFGraph();
            var metaGraphDef = new TFBuffer();

            return TFSession.FromSavedModel(sessionOptions, null, exportDirSavedModel, tags, graph, metaGraphDef);
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

        public static TensorFlowModelInfo LoadTensorFlowModel(IHostEnvironment env, string modelPath)
        {
            var session = GetSession(env, modelPath);
            return new TensorFlowModelInfo(env, session, modelPath);
        }

        internal static TFSession GetSession(IHostEnvironment env, string modelPath)
        {
            Contracts.Check(env != null, nameof(env));
            if (IsSavedModel(env, modelPath))
            {
                env.CheckUserArg(Directory.Exists(modelPath), nameof(modelPath));
                return LoadTFSession(env, modelPath);
            }

            env.CheckUserArg(File.Exists(modelPath), nameof(modelPath));
            var bytes = File.ReadAllBytes(modelPath);
            return LoadTFSession(env, bytes, modelPath);
        }

        internal static unsafe void FetchData<T>(IntPtr data, Span<T> result)
        {
            var dataSpan = new Span<T>(data.ToPointer(), result.Length);
            dataSpan.CopyTo(result);
        }

        internal static bool IsTypeSupported(TFDataType tfoutput)
        {
            switch (tfoutput)
            {
                case TFDataType.Float:
                case TFDataType.Double:
                case TFDataType.UInt8:
                case TFDataType.UInt16:
                case TFDataType.UInt32:
                case TFDataType.UInt64:
                case TFDataType.Int16:
                case TFDataType.Int32:
                case TFDataType.Int64:
                    return true;
                default:
                    return false;
            }
        }

        private sealed class TensorFlowSchema : SimpleSchemaBase
        {
            private readonly MetadataUtils.MetadataGetter<ReadOnlyMemory<char>>[] _opTypeGetters;
            private readonly MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>>[] _inputOpsGetters;
            private readonly int[] _inputOpsLengths;

            public TensorFlowSchema(IExceptionContext ectx, KeyValuePair<string, ColumnType>[] columns,
                MetadataUtils.MetadataGetter<ReadOnlyMemory<char>>[] opTypeGetters,
                MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>>[] inputOpsGetters, int[] inputOpsLengths)
                : base(ectx, columns)
            {
                ectx.CheckParam(Utils.Size(opTypeGetters) == ColumnCount, nameof(opTypeGetters));
                ectx.CheckParam(Utils.Size(inputOpsGetters) == ColumnCount, nameof(inputOpsGetters));
                ectx.CheckParam(Utils.Size(inputOpsLengths) == ColumnCount, nameof(inputOpsLengths));

                _opTypeGetters = opTypeGetters;
                _inputOpsGetters = inputOpsGetters;
                _inputOpsLengths = inputOpsLengths;
            }

            protected override void GetMetadataCore<TValue>(string kind, int col, ref TValue value)
            {
                Ectx.Assert(0 <= col && col < ColumnCount);
                if (kind == OpType)
                    _opTypeGetters[col].Marshal(col, ref value);
                else if (kind == InputOps && _inputOpsGetters[col] != null)
                    _inputOpsGetters[col].Marshal(col, ref value);
                else
                    throw Ectx.ExceptGetMetadata();
            }

            protected override ColumnType GetMetadataTypeOrNullCore(string kind, int col)
            {
                Ectx.Assert(0 <= col && col < ColumnCount);
                if (kind == OpType)
                    return TextType.Instance;
                if (kind == InputOps && _inputOpsGetters[col] != null)
                    return new VectorType(TextType.Instance, _inputOpsLengths[col]);
                return null;
            }

            protected override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypesCore(int col)
            {
                Ectx.Assert(0 <= col && col < ColumnCount);
                yield return new KeyValuePair<string, ColumnType>(OpType, TextType.Instance);
                if (_inputOpsGetters[col] != null)
                    yield return new KeyValuePair<string, ColumnType>(InputOps, new VectorType(TextType.Instance, _inputOpsLengths[col]));
            }
        }
    }
}
