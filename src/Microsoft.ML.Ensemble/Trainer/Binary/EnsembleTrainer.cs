﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Ensemble.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(EnsembleTrainer.Summary, typeof(EnsembleTrainer), typeof(EnsembleTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) },
    EnsembleTrainer.UserNameValue, EnsembleTrainer.LoadNameValue, "pe", "ParallelEnsemble")]

namespace Microsoft.ML.Runtime.Ensemble
{
    using TDistPredictor = IDistPredictorProducing<Single, Single>;
    using TScalarPredictor = IPredictorProducing<Single>;
    /// <summary>
    /// A generic ensemble trainer for binary classification.
    /// </summary>
    public sealed class EnsembleTrainer : EnsembleTrainerBase<Single, TScalarPredictor,
        IBinarySubModelSelector, IBinaryOutputCombiner, SignatureBinaryClassifierTrainer>,
        IModelCombiner<TScalarPredictor, TScalarPredictor>
    {
        public const string LoadNameValue = "WeightedEnsemble";
        public const string UserNameValue = "Parallel Ensemble (bagging, stacking, etc)";
        public const string Summary = "A generic ensemble classifier for binary classification.";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Algorithm to prune the base learners for selective Ensemble", ShortName = "pt", SortOrder = 4)]
            [TGUI(Label = "Sub-Model Selector(pruning) Type",
                Description = "Algorithm to prune the base learners for selective Ensemble")]
            public ISupportBinarySubModelSelectorFactory SubModelSelectorType = new AllSelectorFactory();

            [Argument(ArgumentType.Multiple, HelpText = "Output combiner", ShortName = "oc", SortOrder = 5)]
            [TGUI(Label = "Output combiner", Description = "Output combiner type")]
            public ISupportBinaryOutputCombinerFactory OutputCombiner = new MedianFactory();

            public Arguments()
            {
                BasePredictors = new[] { new SubComponent<ITrainer<TScalarPredictor>, SignatureBinaryClassifierTrainer>("LinearSVM") };
            }
        }

        private readonly ISupportBinaryOutputCombinerFactory _outputCombiner;

        public EnsembleTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue)
        {
            SubModelSelector = args.SubModelSelectorType.CreateComponent(Host);
            _outputCombiner = args.OutputCombiner;
            Combiner = args.OutputCombiner.CreateComponent(Host);
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private protected override TScalarPredictor CreatePredictor(List<FeatureSubsetModel<TScalarPredictor>> models)
        {
            if (models.All(m => m.Predictor is TDistPredictor))
                return new EnsembleDistributionPredictor(Host, PredictionKind, CreateModels<TDistPredictor>(models), Combiner);
            return new EnsemblePredictor(Host, PredictionKind, CreateModels<TScalarPredictor>(models), Combiner);
        }

        public TScalarPredictor CombineModels(IEnumerable<TScalarPredictor> models)
        {
            var combiner = _outputCombiner.CreateComponent(Host);
            var p = models.First();

            if (p is TDistPredictor)
            {
                return new EnsembleDistributionPredictor(Host, p.PredictionKind,
                    models.Select(k => new FeatureSubsetModel<TDistPredictor>((TDistPredictor)k)).ToArray(), combiner);
            }
            return new EnsemblePredictor(Host, p.PredictionKind,
                    models.Select(k => new FeatureSubsetModel<TScalarPredictor>(k)).ToArray(), combiner);
        }
    }
}