// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.CLI.Telemetry.Events
{
    /// <summary>
    /// System info telemetry event.
    /// </summary>
    internal class SystemInfoEvent
    {
        public static void TrackEvent()
        {
            Telemetry.TrackEvent("system-info",
                new Dictionary<string, string>
                {
                    { "LogicalCores", Environment.ProcessorCount.ToString() },
                });
        }
    }
}
