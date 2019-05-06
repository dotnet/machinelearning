// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: EntryPointModule(typeof(AllSelectorFactory))]
[assembly: EntryPointModule(typeof(AllSelectorMulticlassFactory))]
[assembly: EntryPointModule(typeof(BestDiverseSelectorBinary))]
[assembly: EntryPointModule(typeof(BestDiverseSelectorMulticlass))]
[assembly: EntryPointModule(typeof(BestDiverseSelectorRegression))]
[assembly: EntryPointModule(typeof(BestPerformanceRegressionSelector))]
[assembly: EntryPointModule(typeof(BestPerformanceSelector))]
[assembly: EntryPointModule(typeof(BestPerformanceSelectorMulticlass))]

namespace Microsoft.ML.Trainers.Ensemble
{
    [TlcModule.Component(Name = AllSelector.LoadName, FriendlyName = AllSelector.UserName)]
    internal sealed class AllSelectorFactory : ISupportBinarySubModelSelectorFactory, ISupportRegressionSubModelSelectorFactory
    {
        IBinarySubModelSelector IComponentFactory<IBinarySubModelSelector>.CreateComponent(IHostEnvironment env) => new AllSelector(env);

        IRegressionSubModelSelector IComponentFactory<IRegressionSubModelSelector>.CreateComponent(IHostEnvironment env) => new AllSelector(env);
    }

    [TlcModule.Component(Name = AllSelectorMulticlass.LoadName, FriendlyName = AllSelectorMulticlass.UserName)]
    internal sealed class AllSelectorMulticlassFactory : ISupportMulticlassSubModelSelectorFactory
    {
        IMulticlassSubModelSelector IComponentFactory<IMulticlassSubModelSelector>.CreateComponent(IHostEnvironment env) => new AllSelectorMulticlass(env);
    }
}
