// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// Class containing some information about an exectuted pipeline.
    /// These are analogous to IRunResult for smart sweepers.
    /// </summary>
    public sealed class PipelineSweeperRunSummary
    {
        public double MetricValue { get; }
        public double TrainingMetricValue { get; }
        public int NumRowsInTraining { get; }
        public long RunTimeMilliseconds { get; }

        public PipelineSweeperRunSummary(double metricValue, int numRows, long runTimeMilliseconds, double trainingMetricValue)
        {
            MetricValue = metricValue;
            TrainingMetricValue = trainingMetricValue;
            NumRowsInTraining = numRows;
            RunTimeMilliseconds = runTimeMilliseconds;
        }
    }
}
