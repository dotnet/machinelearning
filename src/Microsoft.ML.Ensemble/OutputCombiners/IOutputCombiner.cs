// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;

namespace Microsoft.ML.Trainers.Ensemble
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

    internal interface IStackingTrainer<TOutput>
    {
        void Train(List<FeatureSubsetModel<TOutput>> models, RoleMappedData data, IHostEnvironment env);
        Single ValidationDatasetProportion { get; }
    }

    public interface IRegressionOutputCombiner : IOutputCombiner<Single>
    {
    }

    public interface IBinaryOutputCombiner : IOutputCombiner<Single>
    {
    }

    public interface IMultiClassOutputCombiner : IOutputCombiner<VBuffer<Single>>
    {
    }

    [TlcModule.ComponentKind("EnsembleMulticlassOutputCombiner")]
    public interface ISupportMulticlassOutputCombinerFactory : IComponentFactory<IMultiClassOutputCombiner>
    {
    }

    [TlcModule.ComponentKind("EnsembleBinaryOutputCombiner")]
    public interface ISupportBinaryOutputCombinerFactory : IComponentFactory<IBinaryOutputCombiner>
    {

    }

    [TlcModule.ComponentKind("EnsembleRegressionOutputCombiner")]
    public interface ISupportRegressionOutputCombinerFactory : IComponentFactory<IRegressionOutputCombiner>
    {

    }

    public interface IWeightedAverager
    {
        string WeightageMetricName { get; }
    }

}