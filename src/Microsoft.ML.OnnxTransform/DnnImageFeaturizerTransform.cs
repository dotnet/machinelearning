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
    public sealed class DnnImageFeaturizerEstimator : IEstimator<TransformerChain<OnnxTransform>>
    {
        private readonly EstimatorChain<OnnxTransform> _modelChain;

        public DnnImageFeaturizerEstimator(IHostEnvironment env, string input, string output, DnnModelType model)
        {
            _modelChain = new EstimatorChain<OnnxTransform>();
            _modelsPreprocess.TryGetValue(model, out string prepModel);
            _modelsMain.TryGetValue(model, out string mainModel);
            var tempCol = "onnxDnnPrep";
            var prepEstimator = new OnnxEstimator(env, prepModel, input, tempCol);
            var mainEstimator = new OnnxEstimator(env, mainModel, tempCol, output);
            _modelChain.Append(prepEstimator);
            _modelChain.Append(mainEstimator);
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
             { DnnModelType.Resnet18, "C:\\Models\\DnnImageFeat\\Results\\FinalOnnx\\Prep\\resnetPreprocess.onnx" },
             { DnnModelType.Resnet50, "glove.6B.100d.txt" },
             { DnnModelType.Resnet101, "glove.6B.200d.txt" },
             { DnnModelType.Alexnet, "glove.6B.300d.txt" },
        };

        private static Dictionary<DnnModelType, string> _modelsMain = new Dictionary<DnnModelType, string>()
        {
             { DnnModelType.Resnet18, "C:\\Models\\DnnImageFeat\\Results\\FinalOnnx\\ResNet18\\resnet18.onnx" },
             { DnnModelType.Resnet50, "glove.6B.100d.txt" },
             { DnnModelType.Resnet101, "glove.6B.200d.txt" },
             { DnnModelType.Alexnet, "glove.6B.300d.txt" },
        };

        public TransformerChain<OnnxTransform> Fit(IDataView input)
        {
            return _modelChain.Fit(input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            return _modelChain.GetOutputSchema(inputSchema);
        }
    }
}
