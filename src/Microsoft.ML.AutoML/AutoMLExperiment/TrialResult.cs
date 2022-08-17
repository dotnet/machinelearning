// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Data;
using static Microsoft.ML.TrainCatalogBase;

namespace Microsoft.ML.AutoML
{
    public class TrialResult
    {
        public TrialSettings TrialSettings { get; set; }

        public ITransformer Model { get; set; }

        public double Metric { get; set; }

        public double DurationInMilliseconds { get; set; }
    }

    /// <summary>
    /// TrialResult with Binary Classification Metrics
    /// </summary>
    internal class BinaryClassificationTrialResult : TrialResult
    {
        public BinaryClassificationMetrics BinaryClassificationMetrics { get; set; }

        public IEnumerable<CrossValidationResult<BinaryClassificationMetrics>> CrossValidationMetrics { get; set; }

        public Exception Exception { get; set; }

        public bool IsSucceed { get => Exception == null; }

        public bool IsCrossValidation { get => CrossValidationMetrics == null; }
    }
}
