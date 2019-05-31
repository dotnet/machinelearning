// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Microsoft.ML.CLI.Telemetry.Events
{
    /// <summary>
    /// Telemetry event for CLI application exit.
    /// </summary>
    internal class ApplicationExitEvent
    {
        public static void TrackEvent(int exitCode, bool commandParseSucceeded, TimeSpan duration, Exception ex)
        {
            Telemetry.TrackEvent("application-exit",
                new Dictionary<string, string>
                {
                    { "CommandParseSucceeded", commandParseSucceeded.ToString() },
                    { "ExitCode", exitCode.ToString() },
                    { "PeakMemory", Process.GetCurrentProcess().PeakWorkingSet64.ToString() },
                },
                duration, ex);
        }
    }
}
