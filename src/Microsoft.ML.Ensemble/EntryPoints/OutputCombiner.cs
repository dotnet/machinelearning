// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Ensemble.EntryPoints
{
    [TlcModule.Component(Name = Average.LoadName, FriendlyName = Average.UserName)]
    public sealed class AverageFactory : ISupportOutputCombinerFactory<Single>
    {
        IOutputCombiner<Single> IComponentFactory<IOutputCombiner<Single>>.CreateComponent(IHostEnvironment env) => new Average(env);
    }

    [TlcModule.Component(Name = Median.LoadName, FriendlyName = Median.UserName)]
    public sealed class MedianFactory : ISupportOutputCombinerFactory<Single>
    {
        IOutputCombiner<Single> IComponentFactory<IOutputCombiner<Single>>.CreateComponent(IHostEnvironment env) => new Median(env);
    }

    [TlcModule.Component(Name = Voting.LoadName, FriendlyName = Voting.UserName)]
    public sealed class VotingFactory : ISupportOutputCombinerFactory<Single>
    {
        IOutputCombiner<Single> IComponentFactory<IOutputCombiner<Single>>.CreateComponent(IHostEnvironment env) => new Voting(env);
    }
}
