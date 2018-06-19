// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    /// <summary>
    /// Signature for combiners.
    /// </summary>
    public delegate void SignatureCombiner();

    public delegate void Combiner<TOutput>(ref TOutput dst, TOutput[] src, Single[] weights);

    public interface IOutputCombiner
    {
    }

    /// <summary>
    /// Generic interface for combining outputs of multiple models
    /// </summary>
    public interface IOutputCombiner<TOutput> : IOutputCombiner
    {
        Combiner<TOutput> GetCombiner();
    }

    public interface IStackingTrainer<TOutput>
    {
        void Train(List<FeatureSubsetModel<IPredictorProducing<TOutput>>> models, RoleMappedData data, IHostEnvironment env);
        Single ValidationDatasetProportion { get; }
    }

    public interface IRegressionOutputCombiner : IOutputCombiner<Single>
    {
    }

    public interface IBinaryOutputCombiner : IOutputCombiner<Single>
    {
    }

    [TlcModule.ComponentKind("EnsembleOutputCombiner")]
    public interface ISupportOutputCombinerFactory<TOutput> : IComponentFactory<IOutputCombiner<TOutput>>
    {
    }

    public interface IWeightedAverager
    {
        string WeightageMetricName { get; }
    }

}