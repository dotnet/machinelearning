// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(BestDiverseSelectorMulticlass), typeof(BestDiverseSelectorMulticlass.Arguments),
    typeof(SignatureEnsembleSubModelSelector), BestDiverseSelectorMulticlass.UserName, BestDiverseSelectorMulticlass.LoadName)]

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class BestDiverseSelectorMulticlass : BaseDiverseSelector<VBuffer<Single>, IDiversityMeasure<VBuffer<Single>>>, IMulticlassSubModelSelector
    {
        public const string UserName = "Best Diverse Selector";
        public const string LoadName = "BestDiverseSelectorMultiClass";

        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : DiverseSelectorArguments, ISupportMulticlassSubModelSelectorFactory
        {
            [Argument(ArgumentType.Multiple, HelpText = "The metric type to be used to find the diversity among base learners", ShortName = "dm", SortOrder = 50)]
            [TGUI(Label = "Diversity Measure Type")]
            public ISupportMulticlassDiversityMeasureFactory DiversityMetricType = new MultiDisagreementDiversityFactory();
            public IMulticlassSubModelSelector CreateComponent(IHostEnvironment env) => new BestDiverseSelectorMulticlass(env, this);
        }

        public BestDiverseSelectorMulticlass(IHostEnvironment env, Arguments args)
            : base(env, args, LoadName, args.DiversityMetricType)
        {
        }

        protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        public override List<ModelDiversityMetric<VBuffer<Single>>> CalculateDiversityMeasure(IList<FeatureSubsetModel<VBuffer<float>>> models,
            ConcurrentDictionary<FeatureSubsetModel<VBuffer<float>>, VBuffer<Single>[]> predictions)
        {
            Host.Assert(models.Count > 1);
            Host.Assert(predictions.Count == models.Count);

            var diversityMetric = CreateDiversityMetric();
            return diversityMetric.CalculateDiversityMeasure(models, predictions);
        }
    }
}
