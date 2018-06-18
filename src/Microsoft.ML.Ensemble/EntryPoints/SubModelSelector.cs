// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Ensemble.EntryPoints
{
    [TlcModule.Component(Name = AllSelector.LoadName, FriendlyName = AllSelector.UserName)]
    public sealed class AllSelectorFactory : ISupportSubModelSelectorFactory<Single>
    {
        public ISubModelSelector<Single> CreateComponent(IHostEnvironment env) => new AllSelector(env);
    }

    [TlcModule.Component(Name = AllSelectorMultiClass.LoadName, FriendlyName = AllSelectorMultiClass.UserName)]
    public sealed class AllSelectorMultiClassFactory : ISupportSubModelSelectorFactory<VBuffer<Single>>
    {
        public ISubModelSelector<VBuffer<Single>> CreateComponent(IHostEnvironment env) => new AllSelectorMultiClass(env);
    }
}
