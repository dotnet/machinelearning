// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.Ensemble
{
    internal interface ISubModelSelector<TOutput>
    {
        IList<FeatureSubsetModel<TOutput>> Prune(IList<FeatureSubsetModel<TOutput>> models);

        void CalculateMetrics(FeatureSubsetModel<TOutput> model, ISubsetSelector subsetSelector, Subset subset,
            Batch batch, bool needMetrics);

        Single ValidationDatasetProportion { get; }
    }

    internal interface IRegressionSubModelSelector : ISubModelSelector<Single>
    {
    }

    internal interface IBinarySubModelSelector : ISubModelSelector<Single>
    {
    }

    internal interface IMulticlassSubModelSelector : ISubModelSelector<VBuffer<Single>>
    {
    }

    internal delegate void SignatureEnsembleSubModelSelector();

    [TlcModule.ComponentKind("EnsembleMulticlassSubModelSelector")]
    internal interface ISupportMulticlassSubModelSelectorFactory : IComponentFactory<IMulticlassSubModelSelector>
    {
    }

    [TlcModule.ComponentKind("EnsembleBinarySubModelSelector")]
    internal interface ISupportBinarySubModelSelectorFactory : IComponentFactory<IBinarySubModelSelector>
    {

    }

    [TlcModule.ComponentKind("EnsembleRegressionSubModelSelector")]
    internal interface ISupportRegressionSubModelSelectorFactory : IComponentFactory<IRegressionSubModelSelector>
    {

    }
}
