// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Microsoft.ML.Transforms
{
    public sealed class DnnImageFeaturizerEstimator : IEstimator<TransformerChain<OnnxTransform>>
    {
        private readonly EstimatorChain<OnnxTransform> _modelChain;
        private readonly IHost _host;
        private const int Timeout = 10 * 60 * 1000;

        public enum DnnImageModel
        {
            ResNet18 = 0,
            ResNet50 = 1,
            ResNet101 = 2,
            AlexNet = 3
        }

        private static Dictionary<DnnImageModel, string> _modelsPrep = new Dictionary<DnnImageModel, string>()
        {
            { DnnImageModel.ResNet18, "resnetPreprocess.onnx" },
            { DnnImageModel.ResNet50, "resnetPreprocess.onnx" },
            { DnnImageModel.ResNet101, "resnetPreprocess.onnx" },
            { DnnImageModel.AlexNet, "alexnetPreprocess.onnx" }
        };

        private static Dictionary<DnnImageModel, string> _modelsMain = new Dictionary<DnnImageModel, string>()
        {
            { DnnImageModel.ResNet18, "resnet18.onnx" },
            { DnnImageModel.ResNet50, "resnet50.onnx" },
            { DnnImageModel.ResNet101, "resnet101.onnx" },
            { DnnImageModel.AlexNet, "alexnet.onnx" }
        };

        private static Dictionary<string, string> _modelDirs = new Dictionary<string, string>()
        {
            { "resnetPreprocess.onnx", "ResNetPrep" },
            { "alexnetPreprocess.onnx", "AlexNetPrep" },
            { "resnet18.onnx", "ResNet18" },
            { "resnet50.onnx", "ResNet50" },
            { "resnet101.onnx", "ResNet101" },
            { "alexnet.onnx", "AlexNet" },
        };

        public DnnImageFeaturizerEstimator(IHostEnvironment env, string input, string output, DnnImageModel model)
        {
            _host = env.Register(nameof(DnnImageFeaturizerEstimator));
            _modelChain = new EstimatorChain<OnnxTransform>();
            var tempCol = "onnxDnnPrep";
            var prepEstimator = new OnnxEstimator(env, EnsureModelFile(env, _modelsPrep[model], model), input, tempCol);
            var mainEstimator = new OnnxEstimator(env, EnsureModelFile(env, _modelsMain[model], model), tempCol, output);
            _modelChain = _modelChain.Append(prepEstimator);
            _modelChain = _modelChain.Append(mainEstimator);
        }

        private string EnsureModelFile(IHostEnvironment env, string modelFileName, DnnImageModel kind)
        {
            using (var ch = _host.Start("Ensuring resources"))
            {
                var dir = _modelDirs[modelFileName];
                var url = $"{dir}/{modelFileName}";
                var ensureModel = ResourceManagerUtils.Instance.EnsureResource(_host, ch, url, modelFileName, dir, Timeout);
                ensureModel.Wait();
                var errorResult = ResourceManagerUtils.GetErrorMessage(out var errorMessage, ensureModel.Result);
                if (errorResult != null)
                {
                    var directory = Path.GetDirectoryName(errorResult.FileName);
                    var name = Path.GetFileName(errorResult.FileName);
                    throw ch.Except($"{errorMessage}\nModel file for Dnn Image Featurizer transform could not be found! " +
                        $@"Please copy the model file '{name}' from '{url}' to '{directory}'.");
                }
                return ensureModel.Result.FileName;
            }
            throw _host.Except($"Can't map model kind = {kind} to specific file, please refer to https://aka.ms/MLNetIssue for assistance");
        }

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
