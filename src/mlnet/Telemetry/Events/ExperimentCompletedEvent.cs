// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.AutoML;

namespace Microsoft.ML.CLI.Telemetry.Events
{
    /// <summary>
    /// Telemetry event for AutoML experiment completion.
    /// </summary>
    internal static class ExperimentCompletedEvent
    {
        public static void TrackEvent<TMetrics>(RunDetail<TMetrics> bestRun,
            List<RunDetail<TMetrics>> allRuns,
            TaskKind machineLearningTask,
            TimeSpan duration)
        {
            Telemetry.TrackEvent("experiment-completed",
                new Dictionary<string, string>()
                {
                    { "BestIterationNum", (allRuns.IndexOf(bestRun) + 1).ToString() },
                    { "BestPipeline", Telemetry.GetSanitizedPipelineStr(bestRun.Pipeline) },
                    { "BestTrainer", bestRun.TrainerName },
                    { "MachineLearningTask", machineLearningTask.ToString() },
                    { "NumIterations", allRuns.Count().ToString() },
                    { "PeakMemory", Process.GetCurrentProcess().PeakWorkingSet64.ToString() },
                },
                duration);
        }
    }
}
