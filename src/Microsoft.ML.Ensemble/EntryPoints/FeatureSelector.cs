// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: EntryPointModule(typeof(AllFeatureSelectorFactory))]
[assembly: EntryPointModule(typeof(RandomFeatureSelector))]

namespace Microsoft.ML.Trainers.Ensemble
{
    [TlcModule.Component(Name = AllFeatureSelector.LoadName, FriendlyName = AllFeatureSelector.UserName)]
    internal sealed class AllFeatureSelectorFactory : ISupportFeatureSelectorFactory
    {
        IFeatureSelector IComponentFactory<IFeatureSelector>.CreateComponent(IHostEnvironment env) => new AllFeatureSelector(env);
    }

}
