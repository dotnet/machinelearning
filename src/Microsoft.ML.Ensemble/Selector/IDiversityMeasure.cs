// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Ensemble.Selector.DiversityMeasure;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Runtime.Ensemble.Selector
{
    public interface IDiversityMeasure<TOutput>
    {
        List<ModelDiversityMetric<TOutput>> CalculateDiversityMeasure(IList<FeatureSubsetModel<IPredictorProducing<TOutput>>> models,
            ConcurrentDictionary<FeatureSubsetModel<IPredictorProducing<TOutput>>, TOutput[]> predictions);
    }

    public delegate void SignatureEnsembleDiversityMeasure();

    [TlcModule.ComponentKind("EnsembleDiversityMeasure")]
    public interface ISupportDiversityMeasureFactory<TOutput> : IComponentFactory<IDiversityMeasure<TOutput>>
    {
    }
}
