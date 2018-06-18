// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Ensemble
{
    public sealed class FeatureSubsetModel<TPredictor>
       where TPredictor : IPredictor
    {
        public readonly TPredictor Predictor;
        public readonly BitArray SelectedFeatures;
        public readonly int Cardinality;

        public KeyValuePair<string, double>[] Metrics { get; set; }

        public FeatureSubsetModel(TPredictor predictor, BitArray features = null,
            KeyValuePair<string, double>[] metrics = null)
        {
            Predictor = predictor;
            int card;
            if (features != null && (card = Utils.GetCardinality(features)) < features.Count)
            {
                SelectedFeatures = features;
                Cardinality = card;
            }
            Metrics = metrics;
        }
    }
}
