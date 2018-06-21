// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.ML.Ensemble.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.DiversityMeasure;
using Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(BestDiverseSelectorRegression), typeof(BestDiverseSelectorRegression.Arguments),
    typeof(SignatureEnsembleSubModelSelector), BestDiverseSelectorRegression.UserName, BestDiverseSelectorRegression.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    using TScalarPredictor = IPredictorProducing<Single>;

    public sealed class BestDiverseSelectorRegression : BaseDiverseSelector<Single, RegressionDisagreementDiversityMeasure>, IRegressionSubModelSelector
    {
        public const string UserName = "Best Diverse Selector";
        public const string LoadName = "BestDiverseSelectorRegression";

        protected override ISupportDiversityMeasureFactory<float> DefaultDiversityMetricType => new RegressionDisagreementDiversityFactory();

        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : DiverseSelectorArguments, ISupportRegressionSubModelSelectorFactory
        {
            public IRegressionSubModelSelector CreateComponent(IHostEnvironment env) => new BestDiverseSelectorRegression(env, this);
        }

        public BestDiverseSelectorRegression(IHostEnvironment env, Arguments args)
            : base(env, args, LoadName)
        {
        }

        public override List<ModelDiversityMetric<Single>> CalculateDiversityMeasure(IList<FeatureSubsetModel<TScalarPredictor>> models,
            ConcurrentDictionary<FeatureSubsetModel<TScalarPredictor>, Single[]> predictions)
        {
            var diversityMetric = CreateDiversityMetric();
            return diversityMetric.CalculateDiversityMeasure(models, predictions);
        }

        protected override PredictionKind PredictionKind => PredictionKind.Regression;
    }
}
