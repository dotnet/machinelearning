// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Protobuf;
using Microsoft.ML.Data;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using static Microsoft.ML.Model.OnnxConverter.OnnxCSharpToProtoWrapper;

namespace Microsoft.ML
{
    public static class OnnxExportExtensions
    {
        private static ModelProto ConvertToOnnxProtobufCore(IHostEnvironment env, OnnxContextImpl ctx, ITransformer transform, IDataView inputData, string[] outputColumnNamesToKeep = null)
        {
            var outputData = transform.Transform(inputData);
            LinkedList<ITransformCanSaveOnnx> transforms = null;
            using (var ch = env.Start("ONNX conversion"))
            {
                SaveOnnxCommand.GetPipe(ctx, ch, outputData, out IDataView root, out IDataView sink, out transforms);
                // We pass in the output names to keep, but this next call expects a list of ones to drop. Invert the list.
                var outputColumnNamesToDrop = new HashSet<string>();
                if (outputColumnNamesToKeep != null)
                {
                    for (int i = 0; i < sink.Schema.Count; ++i)
                    {
                        if (!outputColumnNamesToKeep.Contains(sink.Schema[i].Name))
                        {
                            outputColumnNamesToDrop.Add(sink.Schema[i].Name);
                        }
                    }
                }
                return SaveOnnxCommand.ConvertTransformListToOnnxModel(ctx, ch, root, sink, transforms, null, outputColumnNamesToDrop);
            }
        }

        /// <summary>
        /// Convert the specified <see cref="ITransformer"/> to ONNX format. Note that ONNX uses Google's Protobuf so the returned value is a Protobuf object.
        /// </summary>
        /// <param name="catalog">The class that <see cref="ConvertToOnnxProtobuf(ModelOperationsCatalog, ITransformer, IDataView, string[])"/> attached to.</param>
        /// <param name="transform">The <see cref="ITransformer"/> that will be converted into ONNX format.</param>
        /// <param name="inputData">The input of the specified transform.</param>
        /// <param name="outputColumns">List of output columns we want to keep.</param>
        /// <returns>An ONNX model equivalent to the converted ML.NET model.</returns>
        [BestFriend]
        internal static ModelProto ConvertToOnnxProtobuf(this ModelOperationsCatalog catalog, ITransformer transform, IDataView inputData, string[] outputColumns = null)
        {
            var env = catalog.GetEnvironment();
            var ctx = new OnnxContextImpl(env, "model", "ML.NET", "0", 0, "machinelearning.dotnet", OnnxVersion.Stable);
            return ConvertToOnnxProtobufCore(env, ctx, transform, inputData, outputColumns);
        }

        /// <summary>
        /// Convert the specified <see cref="ITransformer"/> to ONNX format. Note that ONNX uses Google's Protobuf so the returned value is a Protobuf object.
        /// </summary>
        /// <param name="catalog">The class that <see cref="ConvertToOnnxProtobuf(ModelOperationsCatalog, ITransformer, IDataView, int)"/> attached to.</param>
        /// <param name="transform">The <see cref="ITransformer"/> that will be converted into ONNX format.</param>
        /// <param name="inputData">The input of the specified transform.</param>
        /// <param name="opSetVersion">The OpSet version to use for exporting the model. This value must be greater than or equal to 9 and less than or equal to 12</param>
        /// <returns>An ONNX model equivalent to the converted ML.NET model.</returns>
        [BestFriend]
        internal static ModelProto ConvertToOnnxProtobuf(this ModelOperationsCatalog catalog, ITransformer transform, IDataView inputData, int opSetVersion)
        {
            var env = catalog.GetEnvironment();
            var ctx = new OnnxContextImpl(env, "model", "ML.NET", "0", 0, "machinelearning.dotnet", OnnxVersion.Stable, opSetVersion);
            return ConvertToOnnxProtobufCore(env, ctx, transform, inputData);
        }

        /// <summary>
        /// Convert the specified <see cref="ITransformer"/> to ONNX format and writes to a stream.
        /// </summary>
        /// <param name="catalog">The class that <see cref="ConvertToOnnx(ModelOperationsCatalog, ITransformer, IDataView, Stream)"/> attached to.</param>
        /// <param name="transform">The <see cref="ITransformer"/> that will be converted into ONNX format.</param>
        /// <param name="inputData">The input of the specified transform.</param>
        /// <param name="stream">The stream to write the protobuf model to.</param>
        /// <returns>An ONNX model equivalent to the converted ML.NET model.</returns>
        public static void ConvertToOnnx(this ModelOperationsCatalog catalog, ITransformer transform, IDataView inputData, Stream stream) =>
            ConvertToOnnxProtobuf(catalog, transform, inputData).WriteTo(stream);

        /// <summary>
        /// Convert the specified <see cref="ITransformer"/> to ONNX format and writes to a stream.
        /// </summary>
        /// <param name="catalog">The class that <see cref="ConvertToOnnx(ModelOperationsCatalog, ITransformer, IDataView, int, Stream)"/> attached to.</param>
        /// <param name="transform">The <see cref="ITransformer"/> that will be converted into ONNX format.</param>
        /// <param name="inputData">The input of the specified transform.</param>
        /// <param name="opSetVersion">The OpSet version to use for exporting the model. This value must be greater than or equal to 9 and less than or equal to 12</param>
        /// <param name="stream">The stream to write the protobuf model to.</param>
        /// <returns>An ONNX model equivalent to the converted ML.NET model.</returns>
        public static void ConvertToOnnx(this ModelOperationsCatalog catalog, ITransformer transform, IDataView inputData, int opSetVersion, Stream stream) =>
            ConvertToOnnxProtobuf(catalog, transform, inputData, opSetVersion).WriteTo(stream);

        /// <summary>
        /// Convert the specified <see cref="ITransformer"/> to ONNX format and writes to a stream.
        /// </summary>
        /// <param name="catalog">The class that <see cref="ConvertToOnnx(ModelOperationsCatalog, ITransformer, IDataView, Stream)"/> attached to.</param>
        /// <param name="transform">The <see cref="ITransformer"/> that will be converted into ONNX format.</param>
        /// <param name="inputData">The input of the specified transform.</param>
        /// <param name="stream">The stream to write the protobuf model to.</param>
        /// <param name="outputColumns">List of output columns we want to keep.</param>
        /// <returns>An ONNX model equivalent to the converted ML.NET model.</returns>
        public static void ConvertToOnnx(this ModelOperationsCatalog catalog, ITransformer transform, IDataView inputData, Stream stream, params string[] outputColumns) =>
            ConvertToOnnxProtobuf(catalog, transform, inputData, outputColumns).WriteTo(stream);
    }
}
