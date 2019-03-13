// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.Ensemble
{
    internal interface IDiversityMeasure<TOutput>
    {
        List<ModelDiversityMetric<TOutput>> CalculateDiversityMeasure(IList<FeatureSubsetModel<TOutput>> models,
            ConcurrentDictionary<FeatureSubsetModel<TOutput>, TOutput[]> predictions);
    }

    internal delegate void SignatureEnsembleDiversityMeasure();

    internal interface IBinaryDiversityMeasure : IDiversityMeasure<Single>
    { }
    internal interface IRegressionDiversityMeasure : IDiversityMeasure<Single>
    { }
    internal interface IMulticlassDiversityMeasure : IDiversityMeasure<VBuffer<Single>>
    { }

    [TlcModule.ComponentKind("EnsembleBinaryDiversityMeasure")]
    internal interface ISupportBinaryDiversityMeasureFactory : IComponentFactory<IBinaryDiversityMeasure>
    {
    }

    [TlcModule.ComponentKind("EnsembleRegressionDiversityMeasure")]
    internal interface ISupportRegressionDiversityMeasureFactory : IComponentFactory<IRegressionDiversityMeasure>
    {
    }

    [TlcModule.ComponentKind("EnsembleMulticlassDiversityMeasure")]
    internal interface ISupportMulticlassDiversityMeasureFactory : IComponentFactory<IMulticlassDiversityMeasure>
    {
    }
}
