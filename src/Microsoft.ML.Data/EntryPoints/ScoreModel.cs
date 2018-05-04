// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(void), typeof(ScoreModel), null, typeof(SignatureEntryPointModule), "ScoreModel")]

namespace Microsoft.ML.Runtime.EntryPoints
{
    /// <summary>
    /// This module handles scoring a <see cref="IPredictorModel"/> against a new dataset.
    /// As a result, we return both the scored data and the scoring transform as a <see cref="ITransformModel"/>.
    /// 
    /// REVIEW: This module does not support 'exotic' scoring scenarios, like recommendation and quantile regression 
    /// (those where the user-defined scorer settings are necessary to identify the scorer). We could resolve this by 
    /// adding a sub-component for extra scorer args, or by creating specialized EPs for these scenarios.
    /// </summary>
    public static partial class ScoreModel
    {
        public sealed class Input
        {
            [Argument(ArgumentType.Required, HelpText = "The dataset to be scored", SortOrder = 1)]
            public IDataView Data;

            [Argument(ArgumentType.Required, HelpText = "The predictor model to apply to data", SortOrder = 2)]
            public IPredictorModel PredictorModel;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Suffix to append to the score columns", SortOrder = 3)]
            public string Suffix;
        }

        public sealed class InputTransformScorer
        {
            [Argument(ArgumentType.Required, HelpText = "The dataset to be scored", SortOrder = 1)]
            public IDataView Data;

            [Argument(ArgumentType.Required, HelpText = "The transform model to apply to data", SortOrder = 2)]
            public ITransformModel TransformModel;
        }

        public sealed class Output
        {
            [TlcModule.Output(Desc = "The scored dataset", SortOrder = 1)]
            public IDataView ScoredData;

            [TlcModule.Output(Desc = "The scoring transform", SortOrder = 2)]
            public ITransformModel ScoringTransform;
        }

        public sealed class ModelInput
        {
            [Argument(ArgumentType.Required, HelpText = "The predictor model to turn into a transform", SortOrder = 1)]
            public IPredictorModel PredictorModel;
        }

        public sealed class ModelOutput
        {
            [TlcModule.Output(Desc = "The scoring transform", SortOrder = 1)]
            public ITransformModel ScoringTransform;
        }

        [TlcModule.EntryPoint(Name = "Transforms.DatasetScorer", Desc = "Score a dataset with a predictor model")]
        public static Output Score(IHostEnvironment env, Input input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("ScoreModel");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);


            IPredictor predictor;
            var inputData = input.Data;
            RoleMappedData data;
            input.PredictorModel.PrepareData(host, inputData, out data, out predictor);

            IDataView scoredPipe;
            using (var ch = host.Start("Creating scoring pipeline"))
            {
                ch.Trace("Creating pipeline");
                var bindable = ScoreUtils.GetSchemaBindableMapper(host, predictor, scorerSettings: null);
                ch.AssertValue(bindable);

                var mapper = bindable.Bind(host, data.Schema);
                var scorer = ScoreUtils.GetScorerComponent(mapper);
                Contracts.Assert(string.IsNullOrEmpty(scorer.SubComponentSettings));
                scorer.SubComponentSettings = string.Format("suffix={{{0}}}", input.Suffix);
                scoredPipe = scorer.CreateInstance(host, data.Data, mapper, input.PredictorModel.GetTrainingSchema(host));
                ch.Done();
            }

            return
                new Output
                {
                    ScoredData = scoredPipe,
                    ScoringTransform = new TransformModel(host, scoredPipe, inputData)
                };

        }

        [TlcModule.EntryPoint(Name = "Transforms.DatasetTransformScorer", Desc = "Score a dataset with a transform model")]
        public static Output ScoreUsingTransform(IHostEnvironment env, InputTransformScorer input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("ScoreModelUsingTransform");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return
                    new Output
                    {
                        ScoredData = input.TransformModel.Apply(env, input.Data),
                        ScoringTransform = null
                    };
        }

        [TlcModule.EntryPoint(Name = "Transforms.Scorer", Desc = "Turn the predictor model into a transform model")]
        public static Output MakeScoringTransform(IHostEnvironment env, ModelInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("MakeScoringTransform");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            IPredictor predictor;
            RoleMappedData data;
            var emptyData = new EmptyDataView(host, input.PredictorModel.TransformModel.InputSchema);
            input.PredictorModel.PrepareData(host, emptyData, out data, out predictor);

            IDataView scoredPipe;
            using (var ch = host.Start("Creating scoring pipeline"))
            {
                ch.Trace("Creating pipeline");
                var bindable = ScoreUtils.GetSchemaBindableMapper(host, predictor, scorerSettings: null);
                ch.AssertValue(bindable);

                var mapper = bindable.Bind(host, data.Schema);
                var scorer = ScoreUtils.GetScorerComponent(mapper);
                scoredPipe = scorer.CreateInstance(host, data.Data, mapper, input.PredictorModel.GetTrainingSchema(host));
                ch.Done();
            }

            return new Output
            {
                ScoredData = scoredPipe,
                ScoringTransform = new TransformModel(host, scoredPipe, emptyData)
            };
        }
    }
}
