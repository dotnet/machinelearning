using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Transforms
{
    public sealed class DnnImageFeaturizerEstimator : TrivialEstimator<TransformerChain<OnnxTransform>>
    {
        private OnnxTransform _prepTransform;
        private OnnxTransform _mainTransform;

        public DnnImageFeaturizerEstimator(IHostEnvironment env, DnnModelType model, string input, string output)
            : this(env, CreateChain(env, model, input, output))
        {
        }

        public DnnImageFeaturizerEstimator(IHostEnvironment env, TransformerChain<OnnxTransform> transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TransformerChain<OnnxTransform>)), transformer)
        {
        }

        private TransformerChain<OnnxTransform> CreateChain(IHostEnvironment env, DnnModelType model, string input, string output)
        {
            _modelsPreprocess.TryGetValue(model, out string prepModel);
            _modelsMain.TryGetValue(model, out string mainModel);
            string tempCol = "onnxDnnPrep";
            _prepTransform = new OnnxTransform(env, prepModel, input, tempCol);
            _mainTransform = new OnnxTransform(env, mainModel, tempCol, output);
            return new TransformerChain<OnnxTransform>(_prepTransform, _mainTransform);
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }

        private SchemaShape GetIntermediateSchema(SchemaShape inputSchema, OnnxTransform transformer)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);

            var input = transformer.Input;
            if (!inputSchema.TryFindColumn(input, out var col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
            if (!(col.Kind == SchemaShape.Column.VectorKind.VariableVector || col.Kind == SchemaShape.Column.VectorKind.Vector))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, nameof(VectorType), col.GetTypeString());
            var inputNodeInfo = transformer.Model.ModelInfo.InputsInfo[0];
            var expectedType = OnnxUtils.OnnxToMlNetType(inputNodeInfo.Type);
            if (col.ItemType != expectedType)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, expectedType.ToString(), col.ItemType.ToString());

            resultDic[transformer.Output] = new SchemaShape.Column(transformer.Output,
                transformer.OutputType.IsKnownSizeVector ? SchemaShape.Column.VectorKind.Vector
                : SchemaShape.Column.VectorKind.VariableVector, NumberType.R4, false);

            return new SchemaShape(resultDic.Values);
        }

        public enum DnnModelType : byte
        {
            Resnet18 = 10,
            Resnet50 = 20,
            Resnet101 = 30,
            Alexnet = 100
        };

        private static Dictionary<DnnModelType, string> _modelsPreprocess = new Dictionary<DnnModelType, string>()
        {
             { DnnModelType.Resnet18, "glove.6B.50d.txt" },
             { DnnModelType.Resnet50, "glove.6B.100d.txt" },
             { DnnModelType.Resnet101, "glove.6B.200d.txt" },
             { DnnModelType.Alexnet, "glove.6B.300d.txt" },
        };

        private static Dictionary<DnnModelType, string> _modelsMain = new Dictionary<DnnModelType, string>()
        {
             { DnnModelType.Resnet18, "glove.6B.50d.txt" },
             { DnnModelType.Resnet50, "glove.6B.100d.txt" },
             { DnnModelType.Resnet101, "glove.6B.200d.txt" },
             { DnnModelType.Alexnet, "glove.6B.300d.txt" },
        };
    }
}
