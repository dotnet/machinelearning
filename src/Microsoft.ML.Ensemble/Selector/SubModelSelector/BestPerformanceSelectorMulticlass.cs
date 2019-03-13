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

[assembly: LoadableClass(typeof(BestPerformanceSelectorMulticlass), typeof(BestPerformanceSelectorMulticlass.Arguments),
    typeof(SignatureEnsembleSubModelSelector), BestPerformanceSelectorMulticlass.UserName, BestPerformanceSelectorMulticlass.LoadName)]

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class BestPerformanceSelectorMulticlass : BaseBestPerformanceSelector<VBuffer<Single>>, IMulticlassSubModelSelector
    {
        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportMulticlassSubModelSelectorFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The metric type to be used to find the best performance", ShortName = "mn", SortOrder = 50)]
            [TGUI(Label = "Metric Name")]
            public MulticlassClassificationEvaluator.Metrics MetricName = MulticlassClassificationEvaluator.Metrics.AccuracyMicro;

            IMulticlassSubModelSelector IComponentFactory<IMulticlassSubModelSelector>.CreateComponent(IHostEnvironment env) => new BestPerformanceSelectorMulticlass(env, this);
        }

        public const string UserName = "Best Performance Selector";
        public const string LoadName = "BestPerformanceSelectorMultiClass";

        private readonly MulticlassClassificationEvaluator.Metrics _metric;
        private readonly string _metricName;

        public BestPerformanceSelectorMulticlass(IHostEnvironment env, Arguments args)
            : base(args, env, LoadName)
        {
            Host.CheckUserArg(Enum.IsDefined(typeof(MulticlassClassificationEvaluator.Metrics), args.MetricName),
                nameof(args.MetricName), "Undefined metric name");
            _metric = args.MetricName;
            _metricName = FindMetricName(typeof(MulticlassClassificationEvaluator.Metrics), _metric);
            Host.Assert(!string.IsNullOrEmpty(_metricName));
        }

        protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        protected override bool IsAscMetric => _metric != MulticlassClassificationEvaluator.Metrics.LogLoss;

        protected override string MetricName => _metricName;
    }
}
