// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using Google.Protobuf;
using Microsoft.ML.Data;
using Microsoft.ML.Model.OnnxConverter;
using static Microsoft.ML.Model.OnnxConverter.OnnxCSharpToProtoWrapper;

namespace Microsoft.ML
{
    public static class OnnxExportExtensions
    {

        /// <summary>
        /// Convert the specified <see cref="ITransformer"/> to ONNX format. Note that ONNX uses Google's Protobuf so the returned value is a Protobuf object.
        /// </summary>
        /// <param name="catalog">The class that <see cref="ConvertToOnnxProtobuf(ModelOperationsCatalog, ITransformer, IDataView)"/> attached to.</param>
        /// <param name="transform">The <see cref="ITransformer"/> that will be converted into ONNX format.</param>
        /// <param name="inputData">The input of the specified transform.</param>
        /// <returns>An ONNX model equivalent to the converted ML.NET model.</returns>
        [BestFriend]
        internal static ModelProto ConvertToOnnxProtobuf(this ModelOperationsCatalog catalog, ITransformer transform, IDataView inputData)
        {
            var env = catalog.GetEnvironment();
            var ctx = new OnnxContextImpl(env, "model", "ML.NET", "0", 0, "machinelearning.dotnet", OnnxVersion.Stable);
            var outputData = transform.Transform(inputData);
            LinkedList<ITransformCanSaveOnnx> transforms = null;
            using (var ch = env.Start("ONNX conversion"))
            {
                SaveOnnxCommand.GetPipe(ctx, ch, outputData, out IDataView root, out IDataView sink, out transforms);
                return SaveOnnxCommand.ConvertTransformListToOnnxModel(ctx, ch, root, sink, transforms, null, null);
            }
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
    }
}
