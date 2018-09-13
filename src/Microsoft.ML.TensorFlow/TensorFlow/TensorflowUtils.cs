// Licensed to the .NET Foundation under one or more agreements.
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

        private static unsafe ISchema GetModelSchema(IExceptionContext ectx, TFGraph graph)
        {
            var res = new List<KeyValuePair<string, ColumnType>>();
            var opTypeGetters = new List<MetadataUtils.MetadataGetter<DvText>>();
            var inputOpsGetters = new List<MetadataUtils.MetadataGetter<VBuffer<DvText>>>();
            var inputOpsLengths = new List<int>();
            foreach (var oper in graph)
            {
                if (oper.NumOutputs != 1)
                    continue;

                var tfType = oper[0].OutputType;
                var mlType = Tf2MlNetTypeOrNull(tfType);
                if (mlType == null)
                    continue;

                var shape = graph.GetTensorShape(oper[0]);
                var shapeArray = shape.ToIntArray();

                inputOpsLengths.Add(oper.NumInputs);
                MetadataUtils.MetadataGetter<VBuffer<DvText>> inputOpsGetter = null;
                if (oper.NumInputs > 0)
                {
                    var inputOps = new DvText[oper.NumInputs];
                    for (int i = 0; i < oper.NumInputs; i++)
                    {
                        var input = oper.GetInput(i);
                        inputOps[i] = new DvText(input.Operation.Name);
                    }
                    inputOpsGetter = (int col, ref VBuffer<DvText> dst) => dst = new VBuffer<DvText>(oper.NumInputs, inputOps);
                }
                inputOpsGetters.Add(inputOpsGetter);

                var opType = oper.OpType;
                MetadataUtils.MetadataGetter<DvText> opTypeGetter = (int col, ref DvText dst) => dst = new DvText(opType);
                opTypeGetters.Add(opTypeGetter);

                var columnType = Utils.Size(shapeArray) > 0 && shapeArray.Skip(1).All(x => x > 0) ?
                    new VectorType(mlType, shapeArray[0] > 0 ? shapeArray : shapeArray.Skip(1).ToArray())
                    : new VectorType(mlType);
                res.Add(new KeyValuePair<string, ColumnType>(oper.Name, columnType));
            }
            return new TensorFlowSchema(ectx, res.ToArray(), opTypeGetters.ToArray(), inputOpsGetters.ToArray(), inputOpsLengths.ToArray());
        }

        public static ISchema GetModelSchema(IExceptionContext ectx, string modelFile)
        {
            var bytes = File.ReadAllBytes(modelFile);
            var session = LoadTFSession(ectx, bytes, modelFile);
            return GetModelSchema(ectx, session.Graph);
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
            private readonly MetadataUtils.MetadataGetter<DvText>[] _opTypeGetters;
            private readonly MetadataUtils.MetadataGetter<VBuffer<DvText>>[] _inputOpsGetters;
            private readonly int[] _inputOpsLengths;

            public TensorFlowSchema(IExceptionContext ectx, KeyValuePair<string, ColumnType>[] columns,
                MetadataUtils.MetadataGetter<DvText>[] opTypeGetters, MetadataUtils.MetadataGetter<VBuffer<DvText>>[] inputOpsGetters, int[] inputOpsLengths)
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
