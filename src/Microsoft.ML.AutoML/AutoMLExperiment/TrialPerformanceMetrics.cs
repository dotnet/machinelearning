// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoML
{
    public class TrialPerformanceMetrics
    {
        public double? PeakMemoryUsage { get; set; }
        public double? PeakCpuUsage { get; set; }
        public double CpuUsage { get; internal set; }
        public double MemoryUsage { get; internal set; }
        public float[] FreeSpaceOnDrives { get; internal set; }
    }
}
