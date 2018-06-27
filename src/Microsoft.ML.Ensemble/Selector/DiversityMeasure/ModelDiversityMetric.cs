// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Ensemble.Selector.DiversityMeasure
{
    public class ModelDiversityMetric<TOutput>
    {
        public FeatureSubsetModel<IPredictorProducing<TOutput>> ModelX { get; set; }
        public FeatureSubsetModel<IPredictorProducing<TOutput>> ModelY { get; set; }
        public Single DiversityNumber { get; set; }
    }
}
