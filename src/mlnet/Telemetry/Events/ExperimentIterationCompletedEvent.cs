// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML.AutoML;
using Newtonsoft.Json;

namespace Microsoft.ML.CLI.Telemetry.Events
{
    /// <summary>
    /// Telemetry event for completion of experiment iteration.
    /// </summary>
    internal static class ExperimentIterationCompletedEvent
    {
        public static void TrackEvent<TMetrics>(int iterationNum,
            RunDetail<TMetrics> runDetail,
            double score,
            TaskKind machineLearningTask)
        {
            Telemetry.TrackEvent("experiment-iteration-completed",
                new Dictionary<string, string>()
                {
                    { "IterationNum", iterationNum.ToString() },
                    { "MachineLearningTask", machineLearningTask.ToString() },
                    { "Metrics", GetMetricsStr(runDetail.ValidationMetrics) },
                    { "PeakMemory", Process.GetCurrentProcess().PeakWorkingSet64.ToString() },
                    { "Pipeline", Telemetry.GetSanitizedPipelineStr(runDetail.Pipeline) },
                    { "PipelineInferenceTimeInSeconds", runDetail.PipelineInferenceTimeInSeconds.ToString() },
                    { "Score", score.ToString() },
                    { "TrainerName", runDetail.TrainerName },
                },
                TimeSpan.FromSeconds(runDetail.RuntimeInSeconds),
                runDetail.Exception);
        }

        private static string GetMetricsStr<TMetrics>(TMetrics metrics)
        {
            if (metrics == null)
            {
                return null;
            }
            return JsonConvert.SerializeObject(metrics);
        }
    }
}
