// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Ensemble.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(typeof(RegressionEnsembleTrainer), typeof(RegressionEnsembleTrainer.Arguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer) },
    RegressionEnsembleTrainer.UserNameValue,
    RegressionEnsembleTrainer.LoadNameValue)]

namespace Microsoft.ML.Runtime.Ensemble
{
    using TScalarPredictor = IPredictorProducing<Single>;
    public sealed class RegressionEnsembleTrainer : EnsembleTrainerBase<Single, TScalarPredictor,
       IRegressionSubModelSelector, IRegressionOutputCombiner, SignatureRegressorTrainer>,
       IModelCombiner<TScalarPredictor, TScalarPredictor>
    {
        public const string LoadNameValue = "EnsembleRegression";
        public const string UserNameValue = "Regression Ensemble (bagging, stacking, etc)";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Algorithm to prune the base learners for selective Ensemble", ShortName = "pt", SortOrder = 4)]
            [TGUI(Label = "Sub-Model Selector(pruning) Type", Description = "Algorithm to prune the base learners for selective Ensemble")]
            public ISupportRegressionSubModelSelectorFactory SubModelSelectorType;

            [Argument(ArgumentType.Multiple, HelpText = "Output combiner", ShortName = "oc", SortOrder = 5)]
            [TGUI(Label = "Output combiner", Description = "Output combiner type")]
            public ISupportRegressionOutputCombinerFactory OutputCombiner;

            public Arguments()
            {
                BasePredictors = new[] { new SubComponent<ITrainer<RoleMappedData, TScalarPredictor>, SignatureRegressorTrainer>("OnlineGradientDescent") };
                OutputCombiner = new MedianFactory();
                SubModelSelectorType = new AllSelectorFactory();
            }
        }

        private readonly ISupportRegressionOutputCombinerFactory _outputCombiner;

        public RegressionEnsembleTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue)
        {
            SubModelSelector = args.SubModelSelectorType.CreateComponent(Host);
            _outputCombiner = args.OutputCombiner;
            Combiner = args.OutputCombiner.CreateComponent(Host);
        }

        public override PredictionKind PredictionKind
        {
            get { return PredictionKind.Regression; }
        }

        public override TScalarPredictor CreatePredictor()
        {
            return new EnsemblePredictor(Host, PredictionKind, CreateModels<TScalarPredictor>(), Combiner);
        }

        public TScalarPredictor CombineModels(IEnumerable<TScalarPredictor> models)
        {
            var combiner = _outputCombiner.CreateComponent(Host);
            var p = models.First();

            var predictor = new EnsemblePredictor(Host, p.PredictionKind,
                    models.Select(k => new FeatureSubsetModel<TScalarPredictor>(k)).ToArray(), combiner);

            return predictor;
        }
    }
}
