// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Ensemble.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: EntryPointModule(typeof(AllSelectorFactory))]
[assembly: EntryPointModule(typeof(AllSelectorMultiClassFactory))]
[assembly: EntryPointModule(typeof(BestDiverseSelectorBinary))]
[assembly: EntryPointModule(typeof(BestDiverseSelectorMultiClass))]
[assembly: EntryPointModule(typeof(BestDiverseSelectorRegression))]
[assembly: EntryPointModule(typeof(BestPerformanceRegressionSelector))]
[assembly: EntryPointModule(typeof(BestPerformanceSelector))]
[assembly: EntryPointModule(typeof(BestPerformanceSelectorMultiClass))]

namespace Microsoft.ML.Ensemble.EntryPoints
{
    [TlcModule.Component(Name = AllSelector.LoadName, FriendlyName = AllSelector.UserName)]
    public sealed class AllSelectorFactory : ISupportBinarySubModelSelectorFactory, ISupportRegressionSubModelSelectorFactory
    {
        IBinarySubModelSelector IComponentFactory<IBinarySubModelSelector>.CreateComponent(IHostEnvironment env) => new AllSelector(env);

        IRegressionSubModelSelector IComponentFactory<IRegressionSubModelSelector>.CreateComponent(IHostEnvironment env) => new AllSelector(env);
    }

    [TlcModule.Component(Name = AllSelectorMultiClass.LoadName, FriendlyName = AllSelectorMultiClass.UserName)]
    public sealed class AllSelectorMultiClassFactory : ISupportMulticlassSubModelSelectorFactory
    {
        IMulticlassSubModelSelector IComponentFactory<IMulticlassSubModelSelector>.CreateComponent(IHostEnvironment env) => new AllSelectorMultiClass(env);
    }
}
