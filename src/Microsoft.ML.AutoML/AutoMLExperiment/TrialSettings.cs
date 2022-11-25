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
    public class TrialSettings
    {
        public int TrialId { get; set; }
        public Parameter Parameter { get; set; }
        [JsonIgnore]
        public CancellationTokenSource CancellationTokenSource { get; set; }
        public TrialPerformanceMetrics PerformanceMetrics { get; internal set; }
        public DateTime StartedAtUtc { get; internal set; }
    }
}
