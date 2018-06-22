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
        IMulticlassSubModelSelector, IMultiClassOutputCombiner, SignatureMultiClassClassifierTrainer>,
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

            public Arguments()
            {
                BasePredictors = new[] { new SubComponent<ITrainer<RoleMappedData, TVectorPredictor>, SignatureMultiClassClassifierTrainer>("MultiClassLogisticRegression") };
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

        public override PredictionKind PredictionKind { get { return PredictionKind.MultiClassClassification; } }

        public override EnsembleMultiClassPredictor CreatePredictor()
        {
            var combiner = Combiner;
            return new EnsembleMultiClassPredictor(Host, CreateModels<TVectorPredictor>(), combiner as IMultiClassOutputCombiner);
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
