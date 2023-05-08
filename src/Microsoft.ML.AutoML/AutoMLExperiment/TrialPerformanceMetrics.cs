// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Performance metrics for a trial.
    /// </summary>
    public class TrialPerformanceMetrics
    {
        /// <summary>
        /// Peak memory usage during the trial in megabytes
        /// </summary>
        public double? PeakMemoryUsage { get; set; }
        /// <summary>
        /// Peak CPU usage during the trial
        /// </summary>
        public double? PeakCpuUsage { get; set; }
        /// <summary>
        /// Current CPU usage of the runner process
        /// </summary>
        public double CpuUsage { get; internal set; }
        /// <summary>
        /// Current memory usage of the runner process in megabytes
        /// </summary>
        public double MemoryUsage { get; internal set; }
    }
}
