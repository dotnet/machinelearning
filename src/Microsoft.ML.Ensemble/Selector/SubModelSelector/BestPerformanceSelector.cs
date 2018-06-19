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

[assembly: LoadableClass(typeof(BestPerformanceSelector), typeof(BestPerformanceSelector.Arguments),
    typeof(SignatureEnsembleSubModelSelector), BestPerformanceSelector.UserName, BestPerformanceSelector.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    public sealed class BestPerformanceSelector : BaseBestPerformanceSelector<Single>, IBinarySubModelSelector
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

        protected override string MetricName
        {
            get { return _metricName; }
        }

        protected override bool IsAscMetric
        {
            get { return _metric != BinaryClassifierEvaluator.Metrics.LogLoss; }
        }

        protected override PredictionKind PredictionKind
        {
            get { return PredictionKind.BinaryClassification; }
        }
    }
}
