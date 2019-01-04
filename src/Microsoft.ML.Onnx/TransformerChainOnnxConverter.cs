using System.Collections.Generic;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.UniversalModelFormat.Onnx;

namespace Microsoft.ML.Model.Onnx
{
    public class TransformerChainOnnxConverter
    {
        public static ModelProto Convert<T>(TransformerChain<T> chain, IDataView inputData, HashSet<string> inputColumnNamesToDrop=null, HashSet<string> outputColumnNamesToDrop=null) where T : class, ITransformer
        {
            var env = new MLContext();
            var ctx = new OnnxContextImpl(env, "model", "ML.NET", "0", 0, "com.test", OnnxVersion.Stable);
            var outputData = chain.Transform(inputData);
            IDataView root = null;
            IDataView sink = null;
            LinkedList<ITransformCanSaveOnnx> transforms = null;
            using (var ch = (env as IChannelProvider).Start("ONNX conversion"))
                SaveOnnxCommand.GetPipe(ctx, ch, outputData, out root, out sink, out transforms);

            return SaveOnnxCommand.ConvertTransformListToOnnxModel(ctx, root, sink, transforms, inputColumnNamesToDrop, outputColumnNamesToDrop);
        }

        public static ModelProto Convert(ITransformer transform, IDataView inputData, HashSet<string> inputColumnNamesToDrop=null, HashSet<string> outputColumnNamesToDrop=null)
        {
            var env = new MLContext(seed: 1);
            var ctx = new OnnxContextImpl(env, "model", "ML.NET", "0", 0, "com.test", OnnxVersion.Stable);
            var outputData = transform.Transform(inputData);
            IDataView root = null;
            IDataView sink = null;
            LinkedList<ITransformCanSaveOnnx> transforms = null;
            using (var ch = (env as IChannelProvider).Start("ONNX conversion"))
                SaveOnnxCommand.GetPipe(ctx, ch, outputData, out root, out sink, out transforms);

            return SaveOnnxCommand.ConvertTransformListToOnnxModel(ctx, root, sink, transforms, inputColumnNamesToDrop, outputColumnNamesToDrop);
        }
    }
}
