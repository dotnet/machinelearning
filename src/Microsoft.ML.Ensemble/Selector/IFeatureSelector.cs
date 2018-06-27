// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Runtime.Ensemble.Selector
{
    public interface IFeatureSelector
    {
        Subset SelectFeatures(RoleMappedData data, IRandom rand);
    }

    public delegate void SignatureEnsembleFeatureSelector();

    [TlcModule.ComponentKind("EnsembleFeatureSelector")]
    public interface ISupportFeatureSelectorFactory : IComponentFactory<IFeatureSelector>
    {
    }
}
