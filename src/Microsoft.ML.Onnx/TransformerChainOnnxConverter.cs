using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.UniversalModelFormat.Onnx;

namespace Microsoft.ML.Model.Onnx
{
    public class TransformerChainOnnxConverter
    {
        public static ModelProto Convert<T>(TransformerChain<T> chain, Schema inputSchema) where T : class, ITransformer
        {
            var env = new MLContext();
            var onnxContext = new OnnxContextImpl(env, "model", "ML.NET", "0", 0, "com.test", Model.Onnx.OnnxVersion.Stable);

            for (int i = 0; i < inputSchema.Count; i++)
            {
                string colName = inputSchema[i].Name;
                onnxContext.AddInputVariable(inputSchema[i].Type, colName);
            }

            foreach (var t in chain)
            {
                var mapper = t.GetRowToRowMapper(inputSchema);
                inputSchema = t.GetOutputSchema(inputSchema);
                (mapper as ISaveAsOnnx).SaveAsOnnx(onnxContext);
            }

            for (int i = 0; i < inputSchema.Count; ++i)
            {
                if (inputSchema[i].IsHidden)
                    continue;

                var idataviewColumnName = inputSchema[i].Name;

                var variableName = onnxContext.TryGetVariableName(idataviewColumnName);
                var trueVariableName = onnxContext.AddIntermediateVariable(null, idataviewColumnName, true);
                onnxContext.CreateNode("Identity", variableName, trueVariableName, onnxContext.GetNodeName("Identity"), "");
                onnxContext.AddOutputVariable(inputSchema[i].Type, trueVariableName);
            }
            return onnxContext.MakeModel();
        }
    }
}
