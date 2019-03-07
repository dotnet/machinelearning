// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.Ensemble
{
    internal interface IFeatureSelector
    {
        Subset SelectFeatures(RoleMappedData data, Random rand);
    }

    internal delegate void SignatureEnsembleFeatureSelector();

    [TlcModule.ComponentKind("EnsembleFeatureSelector")]
    internal interface ISupportFeatureSelectorFactory : IComponentFactory<IFeatureSelector>
    {
    }
}
