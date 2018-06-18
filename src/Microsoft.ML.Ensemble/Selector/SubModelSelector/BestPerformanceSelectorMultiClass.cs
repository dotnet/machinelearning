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

[assembly: LoadableClass(typeof(BestPerformanceSelectorMultiClass), typeof(BestPerformanceSelectorMultiClass.Arguments),
    typeof(SignatureEnsembleSubModelSelector), BestPerformanceSelectorMultiClass.UserName, BestPerformanceSelectorMultiClass.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    public class BestPerformanceSelectorMultiClass : BaseBestPerformanceSelector<VBuffer<Single>>
    {
        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : ArgumentsBase,ISupportSubModelSelectorFactory<VBuffer<Single>>
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The metric type to be used to find the best performance", ShortName = "mn", SortOrder = 50)]
            [TGUI(Label = "Metric Name")]
            public MultiClassClassifierEvaluator.Metrics MetricName = MultiClassClassifierEvaluator.Metrics.AccuracyMicro;

            public ISubModelSelector<VBuffer<Single>> CreateComponent(IHostEnvironment env) => new BestPerformanceSelectorMultiClass(env, this);
        }

        public const string UserName = "Best Performance Selector";
        public const string LoadName = "BestPerformanceSelectorMultiClass";

        private readonly MultiClassClassifierEvaluator.Metrics _metric;
        private readonly string _metricName;

        public BestPerformanceSelectorMultiClass(IHostEnvironment env, Arguments args)
            : base(args, env, LoadName)
        {
            Host.CheckUserArg(Enum.IsDefined(typeof(MultiClassClassifierEvaluator.Metrics), args.MetricName),
                nameof(args.MetricName), "Undefined metric name");
            _metric = args.MetricName;
            _metricName = FindMetricName(typeof(MultiClassClassifierEvaluator.Metrics), _metric);
            Host.Assert(!string.IsNullOrEmpty(_metricName));
        }

        protected override PredictionKind PredictionKind
        {
            get { return PredictionKind.MultiClassClassification; }
        }

        protected override bool IsAscMetric
        {
            get { return _metric != MultiClassClassifierEvaluator.Metrics.LogLoss; }
        }

        protected override string MetricName
        {
            get { return _metricName; }
        }
    }

}
