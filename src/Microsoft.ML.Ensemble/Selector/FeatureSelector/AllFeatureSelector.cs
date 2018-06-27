// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.FeatureSelector;

[assembly: LoadableClass(typeof(AllFeatureSelector), null, typeof(SignatureEnsembleFeatureSelector),
    AllFeatureSelector.UserName, AllFeatureSelector.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.Selector.FeatureSelector
{
    public sealed class AllFeatureSelector : IFeatureSelector
    {
        public const string UserName = "All Feature Selector";
        public const string LoadName = "AllFeatureSelector";

        public AllFeatureSelector(IHostEnvironment env)
        {
        }

        public Subset SelectFeatures(RoleMappedData data, IRandom rand)
        {
            return new Subset(data);
        }
    }
}
