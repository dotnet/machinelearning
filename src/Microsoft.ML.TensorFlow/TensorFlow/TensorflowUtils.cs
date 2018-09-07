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

using TF_Operation = System.IntPtr;

namespace Microsoft.ML.Transforms.TensorFlow
{
    public static class TensorFlowUtils
    {
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
            long pos = 0;
            TF_Operation oper;
            var res = new List<KeyValuePair<string, ColumnType>>();
            while ((oper = TFGraph.TF_GraphNextOperation(graph.handle, &pos)) != IntPtr.Zero)
            {
                var name = TFGraph.TF_OperationName(oper);
                var type = TFGraph.TF_OperationOpType(oper);
                var numOutputs = TFGraph.TF_OperationNumOutputs(oper);
                if (numOutputs != 1)
                    continue;

                var numInputs = TFGraph.TF_OperationNumInputs(oper);
                if (numInputs == 0)
                    continue;

                var tfType = TFGraph.TF_OperationOutputType(new TFOutput(graph[name]));
                var mlType = Tf2MlNetTypeOrNull(tfType);
                if (mlType == null)
                    continue;

                var shape = graph.GetTensorShape(new TFOutput(graph[name]));
                var shapeArray = shape.ToIntArray();
                var columnType = Utils.Size(shapeArray) > 0 && shapeArray.Skip(1).All(x => x > 0) ?
                    new VectorType(mlType, shapeArray[0] > 0 ? shapeArray : shapeArray.Skip(1).ToArray())
                    : new VectorType(mlType);
                res.Add(new KeyValuePair<string, ColumnType>(name, columnType));
            }
            return new SimpleSchema(ectx, res.ToArray());
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
                case TFDataType.UInt32:
                    return NumberType.U4;
                case TFDataType.UInt64:
                    return NumberType.U8;
                default:
                    return null;
            }
        }

        internal static TFSession LoadTFSession(IExceptionContext ectx, byte[] modelBytes, string modelArg)
        {
            var graph = new TFGraph();
            try
            {
                graph.Import(modelBytes, "");
            }
            catch (Exception ex)
            {
                if (!string.IsNullOrEmpty(modelArg))
                    throw ectx.Except($"TensorFlow exception triggered while loading model from '{modelArg}'");
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
    }
}
