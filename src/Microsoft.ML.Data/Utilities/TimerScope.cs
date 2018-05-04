// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    using Stopwatch = System.Diagnostics.Stopwatch;

    /// <summary>
    /// A timer scope class that starts a Stopwatch when created, calculates and prints elapsed time, physical and virtual memory usages before sending these to the telemetry when disposed.
    /// </summary>
    public sealed class TimerScope : IDisposable
    {
        // Note that this class does not own nor dispose of this channel.
        private readonly IChannel _ch;
        private readonly Stopwatch _watch;
        private readonly IHost _host;

        public TimerScope(IHost host, IChannel ch)
        {
            Contracts.AssertValue(ch);

            _ch = ch;
            _host = host;
            _watch = Stopwatch.StartNew();
        }

        public void Dispose()
        {
            _watch.Stop();

            long physicalMemoryUsageInMB = System.Diagnostics.Process.GetCurrentProcess().PeakWorkingSet64 / 1024 / 1024;
            _ch.Info("Physical memory usage(MB): {0}", physicalMemoryUsageInMB);

            long virtualMemoryUsageInMB = System.Diagnostics.Process.GetCurrentProcess().PeakVirtualMemorySize64 / 1024 / 1024;
            _ch.Info("Virtual memory usage(MB): {0}", virtualMemoryUsageInMB);

            // Print the fractions of seconds if elapsed time is small enough that fractions matter
            Double elapsedSeconds = (Double)_watch.ElapsedMilliseconds / 1000;
            if (elapsedSeconds > 99)
                elapsedSeconds = Math.Round(elapsedSeconds);

            // REVIEW: This is \n\n is to prevent changes across bunch of baseline files.
            // Ideally we should change our comparison method to ignore empty lines.
            _ch.Info("{0}\t Time elapsed(s): {1}\n\n", DateTime.Now, elapsedSeconds);

            using (var pipe = _host.StartPipe<TelemetryMessage>("TelemetryPipe"))
            {
                _ch.AssertValue(pipe);

                pipe.Send(TelemetryMessage.CreateMetric("TLC_RunTime", elapsedSeconds));
                pipe.Send(TelemetryMessage.CreateMetric("TLC_PhysicalMemoryUsageInMB", physicalMemoryUsageInMB));
                pipe.Send(TelemetryMessage.CreateMetric("TLC_VirtualMemoryUsageInMB", virtualMemoryUsageInMB));
                pipe.Done();
            }
        }
    }
}
