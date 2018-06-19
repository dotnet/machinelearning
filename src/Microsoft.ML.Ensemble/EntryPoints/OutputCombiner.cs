// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Ensemble.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: EntryPointModule(typeof(AverageFactory))]
[assembly: EntryPointModule(typeof(MedianFactory))]
[assembly: EntryPointModule(typeof(MultiAverage))]
[assembly: EntryPointModule(typeof(MultiMedian))]
[assembly: EntryPointModule(typeof(MultiStacking))]
[assembly: EntryPointModule(typeof(MultiVoting))]
[assembly: EntryPointModule(typeof(MultiWeightedAverage))]
[assembly: EntryPointModule(typeof(RegressionStacking))]
[assembly: EntryPointModule(typeof(Stacking))]
[assembly: EntryPointModule(typeof(VotingFactory))]
[assembly: EntryPointModule(typeof(WeightedAverage))]

namespace Microsoft.ML.Ensemble.EntryPoints
{
    [TlcModule.Component(Name = Average.LoadName, FriendlyName = Average.UserName)]
    public sealed class AverageFactory : ISupportBinaryOutputCombinerFactory, ISupportRegressionOutputCombinerFactory
    {
        public IRegressionOutputCombiner CreateComponent(IHostEnvironment env) => new Average(env);

        IBinaryOutputCombiner IComponentFactory<IBinaryOutputCombiner>.CreateComponent(IHostEnvironment env) => new Average(env);
    }

    [TlcModule.Component(Name = Median.LoadName, FriendlyName = Median.UserName)]
    public sealed class MedianFactory : ISupportBinaryOutputCombinerFactory, ISupportRegressionOutputCombinerFactory
    {
        public IRegressionOutputCombiner CreateComponent(IHostEnvironment env) => new Median(env);

        IBinaryOutputCombiner IComponentFactory<IBinaryOutputCombiner>.CreateComponent(IHostEnvironment env) => new Median(env);
    }

    [TlcModule.Component(Name = Voting.LoadName, FriendlyName = Voting.UserName)]
    public sealed class VotingFactory : ISupportBinaryOutputCombinerFactory
    {
        IBinaryOutputCombiner IComponentFactory<IBinaryOutputCombiner>.CreateComponent(IHostEnvironment env) => new Voting(env);
    }
}
