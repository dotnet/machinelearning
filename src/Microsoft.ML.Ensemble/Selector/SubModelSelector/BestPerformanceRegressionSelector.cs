// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(typeof(BestPerformanceRegressionSelector), typeof(BestPerformanceRegressionSelector.Arguments),
    typeof(SignatureEnsembleSubModelSelector), BestPerformanceRegressionSelector.UserName, BestPerformanceRegressionSelector.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    public sealed class BestPerformanceRegressionSelector : BaseBestPerformanceSelector<Single>, IRegressionSubModelSelector
    {
        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportRegressionSubModelSelectorFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The metric type to be used to find the best performance", ShortName = "mn", SortOrder = 50)]
            [TGUI(Label = "Metric Name")]
            public RegressionEvaluator.Metrics MetricName = RegressionEvaluator.Metrics.L1;

            public IRegressionSubModelSelector CreateComponent(IHostEnvironment env) => new BestPerformanceRegressionSelector(env, this);
        }

        public const string UserName = "Best Performance Selector";
        public const string LoadName = "BestPerformanceRegressionSelector";

        private readonly RegressionEvaluator.Metrics _metric;

        private readonly string _metricName;

        public BestPerformanceRegressionSelector(IHostEnvironment env, Arguments args)
            : base(args, env, LoadName)
        {
            Host.CheckUserArg(Enum.IsDefined(typeof(RegressionEvaluator.Metrics), args.MetricName), nameof(args.MetricName), "Undefined metric name");
            _metric = args.MetricName;
            _metricName = FindMetricName(typeof(RegressionEvaluator.Metrics), _metric);
            Host.Assert(!string.IsNullOrEmpty(_metricName));
        }

        protected override string MetricName => _metricName;

        protected override bool IsAscMetric => false;

        protected override PredictionKind PredictionKind => PredictionKind.Regression;
    }
}
