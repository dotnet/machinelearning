// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.Serialization;
using System.Text.Json.Serialization;
using System.Threading;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Settings used for the trial
    /// </summary>
    public class TrialSettings
    {
        /// <summary>
        /// Identifier of the trial
        /// </summary>
        public int TrialId { get; set; }
        /// <summary>
        /// Parameters for the pipeline used in this trial
        /// </summary>
        public Parameter Parameter { get; set; }
        /// <summary>
        /// Cancellation token source to have the ability to cancel the trial
        /// </summary>
        [JsonIgnore]
        public CancellationTokenSource CancellationTokenSource { get; set; }
        /// <summary>
        /// Performance metrics of the trial
        /// </summary>
        public TrialPerformanceMetrics PerformanceMetrics { get; internal set; }
        /// <summary>
        /// The time when the trial started (UTC)
        /// </summary>
        public DateTime StartedAtUtc { get; internal set; }
    }
}
