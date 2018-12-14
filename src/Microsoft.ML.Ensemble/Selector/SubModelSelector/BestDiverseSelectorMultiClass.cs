﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.ML.Ensemble.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.DiversityMeasure;
using Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(typeof(BestDiverseSelectorMultiClass), typeof(BestDiverseSelectorMultiClass.Arguments),
    typeof(SignatureEnsembleSubModelSelector), BestDiverseSelectorMultiClass.UserName, BestDiverseSelectorMultiClass.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    using TVectorPredictor = IPredictorProducing<VBuffer<Single>>;

    internal sealed class BestDiverseSelectorMultiClass : BaseDiverseSelector<VBuffer<Single>, IDiversityMeasure<VBuffer<Single>>>, IMulticlassSubModelSelector
    {
        public const string UserName = "Best Diverse Selector";
        public const string LoadName = "BestDiverseSelectorMultiClass";

        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : DiverseSelectorArguments, ISupportMulticlassSubModelSelectorFactory
        {
            [Argument(ArgumentType.Multiple, HelpText = "The metric type to be used to find the diversity among base learners", ShortName = "dm", SortOrder = 50)]
            [TGUI(Label = "Diversity Measure Type")]
            public ISupportMulticlassDiversityMeasureFactory DiversityMetricType = new MultiDisagreementDiversityFactory();
            public IMulticlassSubModelSelector CreateComponent(IHostEnvironment env) => new BestDiverseSelectorMultiClass(env, this);
        }

        public BestDiverseSelectorMultiClass(IHostEnvironment env, Arguments args)
            : base(env, args, LoadName, args.DiversityMetricType)
        {
        }

        protected override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

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
