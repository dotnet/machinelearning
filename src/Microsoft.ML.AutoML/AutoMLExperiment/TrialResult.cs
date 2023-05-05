// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.Data;
using static Microsoft.ML.TrainCatalogBase;

namespace Microsoft.ML.AutoML
{
    public class TrialResult : IEqualityComparer<TrialResult>
    {
        public TrialSettings TrialSettings { get; set; }

        public ITransformer Model { get; set; }

        /// <summary>
        /// the loss for current trial, which is smaller the better. This value will be used to fit smart tuners in <see cref="AutoMLExperiment"/>.
        /// </summary>
        public double Loss { get; set; }

        /// <summary>
        /// Evaluation result.
        /// </summary>
        public double Metric { get; set; }

        public double DurationInMilliseconds { get; set; }

        public double? PeakCpu { get; set; }

        public double? PeakMemoryInMegaByte { get; set; }

        public bool Equals(TrialResult x, TrialResult y)
        {
            return GetHashCode(x) == GetHashCode(y);
        }

        /// <summary>
        /// compute hash code based on trial ID only.
        /// </summary>
        public int GetHashCode(TrialResult obj)
        {
            return obj?.TrialSettings?.TrialId.GetHashCode() ?? 0;
        }
    }

    /// <summary>
    /// TrialResult with Metrics
    /// </summary>
    internal class TrialResult<TMetric> : TrialResult
        where TMetric : class
    {
        public TMetric Metrics { get; set; }

        public IEnumerable<CrossValidationResult<TMetric>> CrossValidationMetrics { get; set; }

        public Exception Exception { get; set; }

        public bool IsSucceed { get => Exception == null; }

        public bool IsCrossValidation { get => CrossValidationMetrics == null; }

        public EstimatorChain<ITransformer> Pipeline { get; set; }
    }
}
