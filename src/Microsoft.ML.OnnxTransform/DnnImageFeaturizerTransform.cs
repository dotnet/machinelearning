// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.Transforms.DnnImageFeaturizerEstimator;

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

        // Each main DNN model has preprocessing required based on the model type, these are simple models which do that.
        private static Dictionary<DnnImageModel, string> _modelsPrep = new Dictionary<DnnImageModel, string>()
        {
            { DnnImageModel.ResNet18, "ResNetPreprocess.onnx" },
            { DnnImageModel.ResNet50, "ResNetPreprocess.onnx" },
            { DnnImageModel.ResNet101, "ResNetPreprocess.onnx" },
            { DnnImageModel.AlexNet, "AlexNetPreprocess.onnx" }
        };

        // The main DNN model
        private static Dictionary<DnnImageModel, string> _modelsMain = new Dictionary<DnnImageModel, string>()
        {
            { DnnImageModel.ResNet18, "ResNet18.onnx" },
            { DnnImageModel.ResNet50, "ResNet50.onnx" },
            { DnnImageModel.ResNet101, "ResNet101.onnx" },
            { DnnImageModel.AlexNet, "AlexNet.onnx" }
        };

        // Currently OnnxTransform requires each ONNX model to be in it's own directory, this stores that.
        private static Dictionary<string, string> _modelDirs = new Dictionary<string, string>()
        {
            { "ResNetPreprocess.onnx", "ResNetPrepOnnx" },
            { "AlexNetPreprocess.onnx", "AlexNetPrepOnnx" },
            { "ResNet18.onnx", "ResNet18Onnx" },
            { "ResNet50.onnx", "ResNet50Onnx" },
            { "ResNet101.onnx", "ResNet101Onnx" },
            { "AlexNet.onnx", "AlexNetOnnx" },
        };

        // The actual transformations happen by applying the preprocessing model followed by the DNN model.
        // OnnxTransform is used to run those models, and this transform just combines them for convinience.
        public DnnImageFeaturizerEstimator(IHostEnvironment env, DnnImageModel model, string input, string output)
        {
            _host = env.Register(nameof(DnnImageFeaturizerEstimator));
            _modelChain = new EstimatorChain<OnnxTransform>();
            var tempCol = "onnxDnnPrep";
            var prepEstimator = new OnnxEstimator(env, EnsureModelFile(env, _modelsPrep[model], model), input, tempCol);
            var mainEstimator = new OnnxEstimator(env, EnsureModelFile(env, _modelsMain[model], model), tempCol, output);
            _modelChain = _modelChain.Append(prepEstimator);
            _modelChain = _modelChain.Append(mainEstimator);
        }

        // Downloads the models if necessary from the CDN
        private string EnsureModelFile(IHostEnvironment env, string modelFileName, DnnImageModel kind)
        {
            using (var ch = _host.Start("Ensuring resources"))
            {
                var dir = Path.Combine("image", _modelDirs[modelFileName]);
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

    public static class DnnImageFeaturizerStaticExtensions
    {
        private sealed class OutColumn : Vector<float>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Vector<float> input, DnnImageModel model)
                : base(new Reconciler(model), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly DnnImageModel _model;

            public Reconciler(DnnImageModel model)
            {
                _model = model;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var outCol = (OutColumn)toOutput[0];
                return new DnnImageFeaturizerEstimator(env, _model, inputNames[outCol.Input], outputNames[outCol]);
            }
        }

        public static Vector<float> DnnImageFeaturizer(this Vector<float> input, DnnImageModel model)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutColumn(input, model);
        }
    }
}
