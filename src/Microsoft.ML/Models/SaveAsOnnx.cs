using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model.Onnx;

namespace Microsoft.ML.Models
{
    /// <summary>
    /// Converts a model to ONNX format.
    /// </summary>
    public sealed class SaveAsOnnx
    {
        /// <summary>
        /// Converts and then saves a model to ONNX format.
        /// </summary>
        /// <param name="args">Arguments such as input model file path, output ONNX file path, etc.</param>
        public static void Save(SaveOnnxCommand.Arguments args)
        {
            using (var env = new TlcEnvironment())
            {
                var cmd = new SaveOnnxCommand(env, args);
                cmd.Run();
            }
        }
    }
}
