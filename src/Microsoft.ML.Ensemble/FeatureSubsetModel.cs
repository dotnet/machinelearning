// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class FeatureSubsetModel<TOutput>
    {
        public readonly IPredictorProducing<TOutput> Predictor;
        public readonly BitArray SelectedFeatures;
        public readonly int Cardinality;

        public KeyValuePair<string, double>[] Metrics { get; set; }

        public FeatureSubsetModel(IPredictorProducing<TOutput> predictor, BitArray features = null,
            KeyValuePair<string, double>[] metrics = null)
        {
            if (!(predictor is IPredictorProducing<TOutput> predictorProducing))
            {
                throw Contracts.ExceptParam(nameof(predictor),
                    $"Input predictor did not have the expected output type {typeof(TOutput).Name}.");
            }
            Predictor = predictorProducing;
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
