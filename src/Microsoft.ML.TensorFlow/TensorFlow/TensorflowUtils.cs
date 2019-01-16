﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.AccessControl;
using System.Security.Principal;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Transforms.TensorFlow
{
    public static class TensorFlowUtils
    {
        /// <summary>
        /// Key to access operator's type (a string) in <see cref="Schema.Column.Metadata"/>.
        /// Its value describes the Tensorflow operator that produces this <see cref="Schema.Column"/>.
        /// </summary>
        public const string TensorflowOperatorTypeKind = "TensorflowOperatorType";
        /// <summary>
        /// Key to access upstream operators' names (a string array) in <see cref="Schema.Column.Metadata"/>.
        /// Its value states operators that the associated <see cref="Schema.Column"/>'s generator depends on.
        /// </summary>
        public const string TensorflowUpstreamOperatorsKind = "TensorflowUpstreamOperators";

        internal static Schema GetModelSchema(IExceptionContext ectx, TFGraph graph, string opType = null)
        {
            var schemaBuilder = new SchemaBuilder();
            foreach (var op in graph)
            {
                if (opType != null && opType != op.OpType)
                    continue;

                var tfType = op[0].OutputType;
                // Determine element type in Tensorflow tensor. For example, a vector of floats may get NumberType.R4 here.
                var mlType = Tf2MlNetTypeOrNull(tfType);

                // If the type is not supported in ML.NET then we cannot represent it as a column in an Schema.
                // We also cannot output it with a TensorFlowTransform, so we skip it.
                if (mlType == null)
                    continue;

                // Construct the final ML.NET type of a Tensorflow variable.
                var tensorShape = graph.GetTensorShape(op[0]).ToIntArray();
                var columnType = new VectorType(mlType);
                if (!(Utils.Size(tensorShape) == 1 && tensorShape[0] <= 0) &&
                    (Utils.Size(tensorShape) > 0 && tensorShape.Skip(1).All(x => x > 0)))
                    columnType = new VectorType(mlType, tensorShape[0] > 0 ? tensorShape : tensorShape.Skip(1).ToArray());

                // There can be at most two metadata fields.
                //  1. The first field always presents. Its value is this operator's type. For example,
                //     if an output is produced by an "Softmax" operator, the value of this field should be "Softmax".
                //  2. The second field stores operators whose outputs are consumed by this operator. In other words,
                //     these values are names of some upstream operators which should be evaluated before executing
                //     the current operator. It's possible that one operator doesn't need any input, so this field
                //     can be missing.
                var metadataBuilder = new MetadataBuilder();
                // Create the first metadata field.
                metadataBuilder.Add(TensorflowOperatorTypeKind, TextType.Instance, (ref ReadOnlyMemory<char> value) => value = op.OpType.AsMemory());
                if (op.NumInputs > 0)
                {
                    // Put upstream operators' names to an array (type: VBuffer) of string (type: ReadOnlyMemory<char>).
                    VBuffer<ReadOnlyMemory<char>> upstreamOperatorNames = default;
                    var bufferEditor = VBufferEditor.Create(ref upstreamOperatorNames, op.NumInputs);
                    for (int i = 0; i < op.NumInputs; ++i)
                        bufferEditor.Values[i] = op.GetInput(i).Operation.Name.AsMemory();
                    upstreamOperatorNames = bufferEditor.Commit(); // Used in metadata's getter.

                    // Create the second metadata field.
                    metadataBuilder.Add(TensorflowUpstreamOperatorsKind, new VectorType(TextType.Instance, op.NumInputs),
                        (ref VBuffer<ReadOnlyMemory<char>> value) => { upstreamOperatorNames.CopyTo(ref value); });
                }

                schemaBuilder.AddColumn(op.Name, columnType, metadataBuilder.GetMetadata());
            }
            return schemaBuilder.GetSchema();
        }

        /// <summary>
        /// This method retrieves the information about the graph nodes of a TensorFlow model as an <see cref="Schema"/>.
        /// For every node in the graph that has an output type that is compatible with the types supported by
        /// <see cref="TensorFlowTransformer"/>, the output schema contains a column with the name of that node, and the
        /// type of its output (including the item type and the shape, if it is known). Every column also contains metadata
        /// of kind <see cref="TensorflowOperatorTypeKind"/>, indicating the operation type of the node, and if that node has inputs in the graph,
        /// it contains metadata of kind <see cref="TensorflowUpstreamOperatorsKind"/>, indicating the names of the input nodes.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">Model to load.</param>
        public static Schema GetModelSchema(IHostEnvironment env, string modelPath)
        {
            var model = LoadTensorFlowModel(env, modelPath);
            return GetModelSchema(env, model.Session.Graph);
        }

        /// <summary>
        /// This is a convenience method for iterating over the nodes of a TensorFlow model graph. It
        /// iterates over the columns of the <see cref="Schema"/> returned by <see cref="GetModelSchema(IHostEnvironment, string)"/>,
        /// and for each one it returns a tuple containing the name, operation type, column type and an array of input node names.
        /// This method is convenient for filtering nodes based on certain criteria, for example, by the operation type.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">Model to load.</param>
        /// <returns></returns>
        public static IEnumerable<(string, string, ColumnType, string[])> GetModelNodes(IHostEnvironment env, string modelPath)
        {
            var schema = GetModelSchema(env, modelPath);

            for (int i = 0; i < schema.Count; i++)
            {
                var name = schema[i].Name;
                var type = schema[i].Type;

                var metadataType = schema[i].Metadata.Schema.GetColumnOrNull(TensorflowOperatorTypeKind)?.Type;
                Contracts.Assert(metadataType != null && metadataType is TextType);
                ReadOnlyMemory<char> opType = default;
                schema[i].Metadata.GetValue(TensorflowOperatorTypeKind, ref opType);
                metadataType = schema[i].Metadata.Schema.GetColumnOrNull(TensorflowUpstreamOperatorsKind)?.Type;
                VBuffer<ReadOnlyMemory<char>> inputOps = default;
                if (metadataType != null)
                {
                    Contracts.Assert(metadataType.IsKnownSizeVector && metadataType.ItemType is TextType);
                    schema[i].Metadata.GetValue(TensorflowUpstreamOperatorsKind, ref inputOps);
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
                case TFDataType.UInt8:
                    return NumberType.U1;
                case TFDataType.UInt16:
                    return NumberType.U2;
                case TFDataType.UInt32:
                    return NumberType.U4;
                case TFDataType.UInt64:
                    return NumberType.U8;
                case TFDataType.Int8:
                    return NumberType.I1;
                case TFDataType.Int16:
                    return NumberType.I2;
                case TFDataType.Int32:
                    return NumberType.I4;
                case TFDataType.Int64:
                    return NumberType.I8;
                case TFDataType.Bool:
                    return BoolType.Instance;
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

        /// <summary>
        /// Load TensorFlow model into memory.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">The model to load.</param>
        /// <returns></returns>
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
                case TFDataType.Int8:
                case TFDataType.Int16:
                case TFDataType.Int32:
                case TFDataType.Int64:
                case TFDataType.Bool:
                    return true;
                default:
                    return false;
            }
        }
    }
}
