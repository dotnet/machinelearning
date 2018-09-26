// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(void), typeof(ModelOperations), null, typeof(SignatureEntryPointModule), "ModelOperations")]

namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class ModelOperations
    {
        public sealed class CombineTransformModelsInput
        {
            [Argument(ArgumentType.Multiple, HelpText = "Input models", SortOrder = 1)]
            public ITransformModel[] Models;
        }

        public sealed class CombineTransformModelsOutput
        {
            [TlcModule.Output(Desc = "Combined model", SortOrder = 1)]
            public ITransformModel OutputModel;
        }

        public sealed class PredictorModelInput
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Transform model", SortOrder = 1)]
            public ITransformModel[] TransformModels;

            [Argument(ArgumentType.Required, HelpText = "Predictor model", SortOrder = 2)]
            public IPredictorModel PredictorModel;
        }

        public sealed class SimplePredictorModelInput
        {
            [Argument(ArgumentType.Required, HelpText = "Transform model", SortOrder = 1)]
            public ITransformModel TransformModel;

            [Argument(ArgumentType.Required, HelpText = "Predictor model", SortOrder = 2)]
            public IPredictorModel PredictorModel;
        }

        public sealed class PredictorModelOutput
        {
            [TlcModule.Output(Desc = "Predictor model", SortOrder = 1)]
            public IPredictorModel PredictorModel;
        }

        public sealed class CombineOvaPredictorModelsInput : LearnerInputBaseWithWeight
        {
            [Argument(ArgumentType.Multiple, HelpText = "Input models", SortOrder = 1)]
            public IPredictorModel[] ModelArray;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use probabilities from learners instead of raw values.", SortOrder = 2)]
            public bool UseProbabilities = true;
        }

        public sealed class CombinePredictorModelsInput
        {
            [Argument(ArgumentType.Multiple, HelpText = "Input models", SortOrder = 1)]
            public IPredictorModel[] Models;
        }

        public sealed class ApplyTransformModelInput : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "Transform model", SortOrder = 2)]
            public ITransformModel TransformModel;
        }

        public sealed class ApplyTransformModelOutput
        {
            [TlcModule.Output(Desc = "Transformed dataset", SortOrder = 1)]
            public IDataView OutputData;
        }

        [TlcModule.EntryPoint(Name = "Transforms.ModelCombiner", Desc = "Combines a sequence of TransformModels into a single model")]
        public static CombineTransformModelsOutput CombineTransformModels(IHostEnvironment env, CombineTransformModelsInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CombineTransformModels");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            host.CheckNonEmpty(input.Models, nameof(input.Models));

            ITransformModel model = input.Models[input.Models.Length - 1];
            for (int i = input.Models.Length - 2; i >= 0; i--)
                model = model.Apply(env, input.Models[i]);

            return new CombineTransformModelsOutput { OutputModel = model };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ManyHeterogeneousModelCombiner", Desc = "Combines a sequence of TransformModels and a PredictorModel into a single PredictorModel.")]
        public static PredictorModelOutput CombineModels(IHostEnvironment env, PredictorModelInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CombineModels");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            host.CheckNonEmpty(input.TransformModels, nameof(input.TransformModels));

            ITransformModel model = input.TransformModels[input.TransformModels.Length - 1];
            for (int i = input.TransformModels.Length - 2; i >= 0; i--)
                model = model.Apply(env, input.TransformModels[i]);
            return new PredictorModelOutput() { PredictorModel = input.PredictorModel.Apply(env, model) };
        }

        [TlcModule.EntryPoint(Name = "Transforms.TwoHeterogeneousModelCombiner", Desc = "Combines a TransformModel and a PredictorModel into a single PredictorModel.")]
        public static PredictorModelOutput CombineTwoModels(IHostEnvironment env, SimplePredictorModelInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CombineTwoModels");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return new PredictorModelOutput() { PredictorModel = input.PredictorModel.Apply(env, input.TransformModel) };
        }

        [TlcModule.EntryPoint(Name = "Models.DatasetTransformer", Desc = "Applies a TransformModel to a dataset.", UserName = "Apply Transform Model Output")]
        public static ApplyTransformModelOutput Apply(IHostEnvironment env, ApplyTransformModelInput input)
        {
            return new ApplyTransformModelOutput() { OutputData = input.TransformModel.Apply(env, input.Data) };
        }
    }
}
