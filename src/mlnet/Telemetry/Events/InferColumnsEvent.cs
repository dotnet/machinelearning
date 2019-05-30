// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML.AutoML;

namespace Microsoft.ML.CLI.Telemetry.Events
{
    /// <summary>
    /// Telemetry event for AutoML column inferencing.
    /// </summary>
    internal static class InferColumnsEvent
    {
        public static void TrackEvent(ColumnInformation inferredColumns,
            TimeSpan duration)
        {
            var properties = new Dictionary<string, string>();

            // Include count of each column type present as a property
            var columnsByPurpose = ColumnInformationUtil.CountColumnsByPurpose(inferredColumns);
            var totalColumnCount = 0;
            foreach (var kvp in columnsByPurpose)
            {
                totalColumnCount += kvp.Value;
                if (kvp.Key == ColumnPurpose.Label)
                {
                    continue;
                }
                properties[kvp.Key + "ColumnCount"] = kvp.Value.ToString();
            }

            properties["ColumnCount"] = totalColumnCount.ToString();
            properties["PeakMemory"] = Process.GetCurrentProcess().PeakWorkingSet64.ToString();

            Telemetry.TrackEvent("infer-columns", properties, duration);
        }
    }
}
