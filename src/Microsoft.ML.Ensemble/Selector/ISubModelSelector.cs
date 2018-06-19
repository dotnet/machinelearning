// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.Ensemble.Selector
{
    public interface ISubModelSelector<TOutput>
    {
        IList<FeatureSubsetModel<IPredictorProducing<TOutput>>> Prune(IList<FeatureSubsetModel<IPredictorProducing<TOutput>>> models);

        void CalculateMetrics(FeatureSubsetModel<IPredictorProducing<TOutput>> model, ISubsetSelector subsetSelector, Subset subset,
            Batch batch, bool needMetrics);

        Single ValidationDatasetProportion { get; }
    }

    public interface IRegressionSubModelSelector : ISubModelSelector<Single>
    {
    }

    public interface IBinarySubModelSelector : ISubModelSelector<Single>
    {
    }

    public interface IMulticlassSubModelSelector : ISubModelSelector<VBuffer<Single>>
    {
    }

    public delegate void SignatureEnsembleSubModelSelector();

    [TlcModule.ComponentKind("EnsembleMulticlassSubModelSelector")]
    public interface ISupportMulticlassSubModelSelectorFactory : IComponentFactory<IMulticlassSubModelSelector>
    {
    }

    [TlcModule.ComponentKind("EnsembleBinarySubModelSelector")]
    public interface ISupportBinarySubModelSelectorFactory: IComponentFactory<IBinarySubModelSelector>
    {

    }

    [TlcModule.ComponentKind("EnsembleRegressionSubModelSelector")]
    public interface ISupportRegressionSubModelSelectorFactory : IComponentFactory<IRegressionSubModelSelector>
    {

    }
}
