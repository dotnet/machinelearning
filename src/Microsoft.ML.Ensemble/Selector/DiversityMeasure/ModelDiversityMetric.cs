// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class ModelDiversityMetric<TOutput>
    {
        public FeatureSubsetModel<TOutput> ModelX { get; set; }
        public FeatureSubsetModel<TOutput> ModelY { get; set; }
        public Single DiversityNumber { get; set; }
    }
}
