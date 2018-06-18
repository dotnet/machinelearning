// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.DiversityMeasure;
using Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(BestDiverseSelectorMultiClass), typeof(BestDiverseSelectorMultiClass.Arguments),
    typeof(SignatureEnsembleSubModelSelector), BestDiverseSelectorMultiClass.UserName, BestDiverseSelectorMultiClass.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    using TVectorPredictor = IPredictorProducing<VBuffer<Single>>;
    public sealed class BestDiverseSelectorMultiClass : BaseDiverseSelector<VBuffer<Single>, IDiversityMeasure<VBuffer<Single>>>
    {
        public const string UserName = "Best Diverse Selector";
        public const string LoadName = "BestDiverseSelectorMultiClass";

        public override string DiversityMeasureLoadname
        {
            get { return MultiDisagreementDiversityMeasure.LoadName; }
        }

        [TlcModule.Component(Name = BestDiverseSelectorMultiClass.LoadName, FriendlyName = BestDiverseSelectorMultiClass.UserName)]
        public sealed class Arguments : DiverseSelectorArguments, ISupportSubModelSelectorFactory<VBuffer<Single>>
        {
            public ISubModelSelector<VBuffer<Single>> CreateComponent(IHostEnvironment env) => new BestDiverseSelectorMultiClass(env, this);
        }

        public BestDiverseSelectorMultiClass(IHostEnvironment env, Arguments args)
            : base(env, args, LoadName)
        {
        }

        protected override PredictionKind PredictionKind
        {
            get { return PredictionKind.MultiClassClassification; }
        }

        public override List<ModelDiversityMetric<VBuffer<Single>>> CalculateDiversityMeasure(IList<FeatureSubsetModel<TVectorPredictor>> models,
            ConcurrentDictionary<FeatureSubsetModel<TVectorPredictor>, VBuffer<Single>[]> predictions)
        {
            Host.Assert(models.Count > 1);
            Host.Assert(predictions.Count == models.Count);

            var diversityMetric = CreateDiversityMetric();
            return diversityMetric.CalculateDiversityMeasure(models, predictions);
        }
    }
}
