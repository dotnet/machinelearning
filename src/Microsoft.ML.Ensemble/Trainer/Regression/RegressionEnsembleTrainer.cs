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
using Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector;
using Microsoft.ML.Runtime.Learners;

[assembly: LoadableClass(typeof(RegressionEnsembleTrainer), typeof(RegressionEnsembleTrainer.Arguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer) },
    RegressionEnsembleTrainer.UserNameValue,
    RegressionEnsembleTrainer.LoadNameValue)]

namespace Microsoft.ML.Runtime.Ensemble
{
    using TScalarPredictor = IPredictorProducing<Single>;
    public sealed class RegressionEnsembleTrainer : EnsembleTrainerBase<Single, TScalarPredictor,
       IRegressionSubModelSelector, IRegressionOutputCombiner, SignatureRegressorTrainer>,
       IModelCombiner<WeightedValue<TScalarPredictor>, TScalarPredictor>
    {
        public const string LoadNameValue = "EnsembleRegression";
        public const string UserNameValue = "Regression Ensemble (bagging, stacking, etc)";

        public sealed class Arguments : ArgumentsBase
        {
            public Arguments()
            {
                BasePredictors = new[] { new SubComponent<ITrainer<RoleMappedData, TScalarPredictor>, SignatureRegressorTrainer>("OnlineGradientDescent") };
                OutputCombiner = new MedianFactory();
                SubModelSelectorType = new AllSelectorFactory();
            }
        }

        public RegressionEnsembleTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue)
        {
        }

        public override PredictionKind PredictionKind
        {
            get { return PredictionKind.Regression; }
        }

        public override TScalarPredictor CreatePredictor()
        {
            return new EnsemblePredictor(Host, PredictionKind, CreateModels<TScalarPredictor>(), Combiner);
        }

        public TScalarPredictor CombineModels(IEnumerable<WeightedValue<TScalarPredictor>> models)
        {
            var weights = models.Select(m => m.Weight).ToArray();
            if (weights.All(w => w == 1))
                weights = null;
            var combiner = Args.OutputCombiner.CreateComponent(Host);
            var p = models.First().Value;

            var predictor = new EnsemblePredictor(Host, p.PredictionKind,
                    models.Select(k => new FeatureSubsetModel<TScalarPredictor>(k.Value)).ToArray(),
                    combiner,
                    weights);

            return predictor;
        }
    }
}
