﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Transforms.TensorFlow
{
    public static class TensorFlowUtils
    {
        public const string OpType = "OpType";
        public const string InputOps = "InputOps";

        // This method is needed for the Pipeline API, since ModuleCatalog does not load entry points that are located
        // in assemblies that aren't directly used in the code. Users who want to use TensorFlow components will have to call
        // TensorFlowUtils.Initialize() before creating the pipeline.
        /// <summary>
        /// Initialize the TensorFlow environment. Call this method before adding TensorFlow components to a learning pipeline.
        /// </summary>
        public static void Initialize()
        {
            ImageAnalytics.Initialize();
        }

        private static ISchema GetModelSchema(IExceptionContext ectx, TFGraph graph)
        {
            var res = new List<KeyValuePair<string, ColumnType>>();
            var opTypeGetters = new List<MetadataUtils.MetadataGetter<ReadOnlyMemory<char>>>();
            var inputOpsGetters = new List<MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>>>();
            var inputOpsLengths = new List<int>();
            foreach (var op in graph)
            {
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

                var opType = op.OpType;
                MetadataUtils.MetadataGetter<ReadOnlyMemory<char>> opTypeGetter =
                    (int col, ref ReadOnlyMemory<char> dst) => dst = new ReadOnlyMemory<char>(opType.ToArray());
                opTypeGetters.Add(opTypeGetter);

                var columnType = Utils.Size(shapeArray) == 1 && shapeArray[0] == -1 ? new VectorType(mlType) :
                    Utils.Size(shapeArray) > 0 && shapeArray.Skip(1).All(x => x > 0) ?
                        new VectorType(mlType, shapeArray[0] > 0 ? shapeArray : shapeArray.Skip(1).ToArray())
                        : new VectorType(mlType);
                res.Add(new KeyValuePair<string, ColumnType>(op.Name, columnType));
            }
            return new TensorFlowSchema(ectx, res.ToArray(), opTypeGetters.ToArray(), inputOpsGetters.ToArray(), inputOpsLengths.ToArray());
        }

        public static ISchema GetModelSchema(IExceptionContext ectx, string modelFile)
        {
            var bytes = File.ReadAllBytes(modelFile);
            var session = LoadTFSession(ectx, bytes, modelFile);
            return GetModelSchema(ectx, session.Graph);
        }

        public static IEnumerable<(string, string, ColumnType, string[])> GetModelNodes(string modelFile)
        {
            var schema = GetModelSchema(null, modelFile);

            for (int i = 0; i < schema.ColumnCount; i++)
            {
                var name = schema.GetColumnName(i);
                var type = schema.GetColumnType(i);

                var metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.OpType, i);
                Contracts.Assert(metadataType != null && metadataType.IsText);
                ReadOnlyMemory<char> opType = default;
                schema.GetMetadata(TensorFlowUtils.OpType, i, ref opType);
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, i);
                VBuffer<ReadOnlyMemory<char>> inputOps = default;
                if (metadataType != null)
                {
                    Contracts.Assert(metadataType.IsKnownSizeVector && metadataType.ItemType.IsText);
                    schema.GetMetadata(TensorFlowUtils.InputOps, i, ref inputOps);
                }
                yield return (name, opType.ToString(), type,
                    Utils.Size(inputOps.Values) > 0 ? inputOps.Values.Select(input => input.ToString()).ToArray() : new string[0]);
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

        internal static unsafe void FetchData<T>(IntPtr data, T[] result)
        {
            var size = result.Length;

            GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr target = handle.AddrOfPinnedObject();

            Int64 sizeInBytes = size * Marshal.SizeOf((typeof(T)));
            Buffer.MemoryCopy(data.ToPointer(), target.ToPointer(), sizeInBytes, sizeInBytes);
            handle.Free();
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
