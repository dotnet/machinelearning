// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector;

[assembly: LoadableClass(typeof(AllSelectorMultiClass), null, typeof(SignatureEnsembleSubModelSelector),
    AllSelectorMultiClass.UserName, AllSelectorMultiClass.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    public class AllSelectorMultiClass : BaseSubModelSelector<VBuffer<Single>>, IMulticlassSubModelSelector
    {
        public const string UserName = "All Selector";
        public const string LoadName = "AllSelectorMultiClass";

        public override Single ValidationDatasetProportion => 0;

        protected override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        public AllSelectorMultiClass(IHostEnvironment env)
            : base(env, LoadName)
        {
        }
    }
}
