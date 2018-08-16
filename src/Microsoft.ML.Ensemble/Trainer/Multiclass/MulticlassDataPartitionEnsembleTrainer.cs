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
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Learners;

[assembly: LoadableClass(MulticlassDataPartitionEnsembleTrainer.Summary, typeof(MulticlassDataPartitionEnsembleTrainer),
    typeof(MulticlassDataPartitionEnsembleTrainer.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    MulticlassDataPartitionEnsembleTrainer.UserNameValue,
    MulticlassDataPartitionEnsembleTrainer.LoadNameValue)]

namespace Microsoft.ML.Runtime.Ensemble
{
    using TVectorPredictor = IPredictorProducing<VBuffer<Single>>;
    /// <summary>
    /// A generic ensemble classifier for multi-class classification
    /// </summary>
    public sealed class MulticlassDataPartitionEnsembleTrainer :
        EnsembleTrainerBase<VBuffer<Single>, EnsembleMultiClassPredictor,
        IMulticlassSubModelSelector, IMultiClassOutputCombiner>,
        IModelCombiner<TVectorPredictor, TVectorPredictor>
    {
        public const string LoadNameValue = "WeightedEnsembleMulticlass";
        public const string UserNameValue = "Multi-class Parallel Ensemble (bagging, stacking, etc)";
        public const string Summary = "A generic ensemble classifier for multi-class classification.";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Algorithm to prune the base learners for selective Ensemble", ShortName = "pt", SortOrder = 4)]
            [TGUI(Label = "Sub-Model Selector(pruning) Type", Description = "Algorithm to prune the base learners for selective Ensemble")]
            public ISupportMulticlassSubModelSelectorFactory SubModelSelectorType = new AllSelectorMultiClassFactory();

            [Argument(ArgumentType.Multiple, HelpText = "Output combiner", ShortName = "oc", SortOrder = 5)]
            [TGUI(Label = "Output combiner", Description = "Output combiner type")]
            public ISupportMulticlassOutputCombinerFactory OutputCombiner = new MultiMedian.Arguments();

            [Argument(ArgumentType.Multiple, HelpText = "Base predictor type", ShortName = "bp,basePredictorTypes", SortOrder = 1, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureMultiClassClassifierTrainer))]
            public IComponentFactory<ITrainer<TVectorPredictor>>[] BasePredictors;

            public override IComponentFactory<ITrainer<TVectorPredictor>>[] BasePredictorFactories
            {
                get { return BasePredictors; }
                set { BasePredictors = value; }
            }

            public Arguments()
            {
                BasePredictors = new[]
                {
                    ComponentFactoryUtils.CreateFromFunction(
                        env => new MulticlassLogisticRegression(env, new MulticlassLogisticRegression.Arguments()))
                };
            }
        }

        private readonly ISupportMulticlassOutputCombinerFactory _outputCombiner;

        public MulticlassDataPartitionEnsembleTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue)
        {
            SubModelSelector = args.SubModelSelectorType.CreateComponent(Host);
            _outputCombiner = args.OutputCombiner;
            Combiner = args.OutputCombiner.CreateComponent(Host);
        }

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        private protected override EnsembleMultiClassPredictor CreatePredictor(List<FeatureSubsetModel<TVectorPredictor>> models)
        {
            return new EnsembleMultiClassPredictor(Host, CreateModels<TVectorPredictor>(models), Combiner as IMultiClassOutputCombiner);
        }

        public TVectorPredictor CombineModels(IEnumerable<TVectorPredictor> models)
        {
            var predictor = new EnsembleMultiClassPredictor(Host,
                models.Select(k => new FeatureSubsetModel<TVectorPredictor>(k)).ToArray(),
                _outputCombiner.CreateComponent(Host));

            return predictor;
        }
    }
}
