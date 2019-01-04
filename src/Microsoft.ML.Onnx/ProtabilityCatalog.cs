using System.Collections.Generic;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Model.Onnx;
using Microsoft.ML.UniversalModelFormat.Onnx;

namespace Microsoft.ML
{
    public static class ProtabilityCatalog
    {
        /// <summary>
        /// Convert the specified <see cref="ITransformer"/> to ONNX format. Note that ONNX uses Google's Protobuf so the returned value is a Protobuf object.
        /// </summary>
        /// <param name="catalog">A field in <see cref="MLContext"/> which this function associated with.</param>
        /// <param name="transform">The <see cref="ITransformer"/> that will be converted into ONNX format.</param>
        /// <param name="inputData">The input of the specified transform.</param>
        /// <returns></returns>
        public static ModelProto ConvertToOnnx(this ModelOperationsCatalog.PortabilityTransforms catalog, ITransformer transform, IDataView inputData)
        {
            var env = new MLContext(seed: 1);
            var ctx = new OnnxContextImpl(env, "model", "ML.NET", "0", 0, "com.microsoft", OnnxVersion.Stable);
            var outputData = transform.Transform(inputData);
            IDataView root = null;
            IDataView sink = null;
            LinkedList<ITransformCanSaveOnnx> transforms = null;
            using (var ch = (env as IChannelProvider).Start("ONNX conversion"))
                SaveOnnxCommand.GetPipe(ctx, ch, outputData, out root, out sink, out transforms);

            return SaveOnnxCommand.ConvertTransformListToOnnxModel(ctx, root, sink, transforms, null, null);
        }
    }
}
