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
    /// <summary>
    /// Signature for combiners.
    /// </summary>
    internal delegate void SignatureCombiner();

    internal delegate void Combiner<TOutput>(ref TOutput dst, TOutput[] src, Single[] weights);

    internal interface IOutputCombiner
    {
    }

    /// <summary>
    /// Generic interface for combining outputs of multiple models
    /// </summary>
    internal interface IOutputCombiner<TOutput> : IOutputCombiner
    {
        Combiner<TOutput> GetCombiner();
    }

    internal interface IStackingTrainer<TOutput>
    {
        void Train(List<FeatureSubsetModel<TOutput>> models, RoleMappedData data, IHostEnvironment env);
        Single ValidationDatasetProportion { get; }
    }

    internal interface IRegressionOutputCombiner : IOutputCombiner<Single>
    {
    }

    internal interface IBinaryOutputCombiner : IOutputCombiner<Single>
    {
    }

    internal interface IMulticlassOutputCombiner : IOutputCombiner<VBuffer<Single>>
    {
    }

    [TlcModule.ComponentKind("EnsembleMulticlassOutputCombiner")]
    internal interface ISupportMulticlassOutputCombinerFactory : IComponentFactory<IMulticlassOutputCombiner>
    {
    }

    [TlcModule.ComponentKind("EnsembleBinaryOutputCombiner")]
    internal interface ISupportBinaryOutputCombinerFactory : IComponentFactory<IBinaryOutputCombiner>
    {

    }

    [TlcModule.ComponentKind("EnsembleRegressionOutputCombiner")]
    internal interface ISupportRegressionOutputCombinerFactory : IComponentFactory<IRegressionOutputCombiner>
    {

    }

    internal interface IWeightedAverager
    {
        string WeightageMetricName { get; }
    }

}