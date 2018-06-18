// Licensed to the .NET Foundation under one or more agreements.
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
using Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector;
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
        ISubModelSelector<VBuffer<Single>>, IOutputCombiner<VBuffer<Single>>, SignatureMultiClassClassifierTrainer>,
        IModelCombiner<WeightedValue<TVectorPredictor>, TVectorPredictor>
    {
        public const string LoadNameValue = "WeightedEnsembleMulticlass";
        public const string UserNameValue = "Multi-class Parallel Ensemble (bagging, stacking, etc)";
        public const string Summary = "A generic ensemble classifier for multi-class classification.";

        public sealed class Arguments : ArgumentsBase
        {
            public Arguments()
            {
                BasePredictors = new[] { new SubComponent<ITrainer<RoleMappedData, TVectorPredictor>, SignatureMultiClassClassifierTrainer>("MultiClassLogisticRegression") };
                OutputCombiner = new SubComponent<IOutputCombiner<VBuffer<Single>>, SignatureCombiner>(MultiMedian.LoadName);
                SubModelSelectorType = new SubComponent<ISubModelSelector<VBuffer<Single>>, SignatureEnsembleSubModelSelector>(AllSelectorMultiClass.LoadName);
            }
        }

        public MulticlassDataPartitionEnsembleTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue)
        {
        }

        public override PredictionKind PredictionKind { get { return PredictionKind.MultiClassClassification; } }

        public override EnsembleMultiClassPredictor CreatePredictor()
        {
            var combiner = Combiner;
            return new EnsembleMultiClassPredictor(Host, CreateModels<TVectorPredictor>(), combiner);
        }

        public TVectorPredictor CombineModels(IEnumerable<WeightedValue<TVectorPredictor>> models)
        {
            var weights = models.Select(m => m.Weight).ToArray();
            if (weights.All(w => w == 1))
                weights = null;

            var predictor = new EnsembleMultiClassPredictor(Host,
                models.Select(k => new FeatureSubsetModel<TVectorPredictor>(k.Value)).ToArray(),
                Args.OutputCombiner.CreateInstance(Host), weights);

            return predictor;
        }
    }
}
