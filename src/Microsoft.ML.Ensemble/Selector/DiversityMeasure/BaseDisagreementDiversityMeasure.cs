﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.Ensemble
{
    internal abstract class BaseDisagreementDiversityMeasure<TOutput> : IDiversityMeasure<TOutput>
    {
        public List<ModelDiversityMetric<TOutput>> CalculateDiversityMeasure(IList<FeatureSubsetModel<TOutput>> models,
            ConcurrentDictionary<FeatureSubsetModel<TOutput>, TOutput[]> predictions)
        {
            Contracts.Assert(models.Count > 1);
            Contracts.Assert(predictions.Count == models.Count);

            var diversityValues = new List<ModelDiversityMetric<TOutput>>();

            for (int i = 0; i < (models.Count - 1); i++)
            {
                for (int j = i + 1; j < models.Count; j++)
                {
                    Single differencesCount = 0;
                    var modelXOutputs = predictions[models[i]];
                    var modelYOutputs = predictions[models[j]];
                    for (int k = 0; k < modelXOutputs.Length; k++)
                    {
                        differencesCount += GetDifference(in modelXOutputs[k], in modelYOutputs[k]);
                    }
                    diversityValues.Add(new ModelDiversityMetric<TOutput>()
                    {
                        DiversityNumber = differencesCount,
                        ModelX = models[i],
                        ModelY = models[j]
                    });
                }
            }
            return diversityValues;
        }

        protected abstract Single GetDifference(in TOutput tOutput1, in TOutput tOutput2);
    }
}
