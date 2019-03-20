// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(BestPerformanceSelector), typeof(BestPerformanceSelector.Arguments),
    typeof(SignatureEnsembleSubModelSelector), BestPerformanceSelector.UserName, BestPerformanceSelector.LoadName)]

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class BestPerformanceSelector : BaseBestPerformanceSelector<Single>, IBinarySubModelSelector
    {
        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportBinarySubModelSelectorFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The metric type to be used to find the best performance", ShortName = "mn", SortOrder = 50)]
            [TGUI(Label = "Metric Name")]
            public BinaryClassifierEvaluator.Metrics MetricName = BinaryClassifierEvaluator.Metrics.Auc;

            public IBinarySubModelSelector CreateComponent(IHostEnvironment env) => new BestPerformanceSelector(env, this);
        }

        public const string UserName = "Best Performance Selector";
        public const string LoadName = "BestPerformanceSelector";

        private readonly BinaryClassifierEvaluator.Metrics _metric;
        private readonly string _metricName;

        public BestPerformanceSelector(IHostEnvironment env, Arguments args)
            : base(args, env, LoadName)
        {
            Host.CheckUserArg(Enum.IsDefined(typeof(BinaryClassifierEvaluator.Metrics), args.MetricName),
                nameof(args.MetricName), "Undefined metric name");
            _metric = args.MetricName;
            _metricName = FindMetricName(typeof(BinaryClassifierEvaluator.Metrics), _metric);
            Host.Assert(!string.IsNullOrEmpty(_metricName));
        }

        protected override string MetricName => _metricName;

        protected override bool IsAscMetric => _metric != BinaryClassifierEvaluator.Metrics.LogLoss;

        protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
    }
}
