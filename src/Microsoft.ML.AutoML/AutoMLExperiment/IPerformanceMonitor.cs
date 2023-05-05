// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Timers;
using Microsoft.ML.Runtime;
using Timer = System.Timers.Timer;

namespace Microsoft.ML.AutoML
{
    public interface IPerformanceMonitor : IDisposable
    {
        void Start();

        void Pause();

        void Stop();

        double? GetPeakMemoryUsageInMegaByte();

        double? GetPeakCpuUsage();

        /// <summary>
        /// The handler function every time <see cref="PerformanceMetricsUpdated"/> get fired.
        /// </summary>
        void OnPerformanceMetricsUpdatedHandler(TrialSettings trialSettings, TrialPerformanceMetrics metrics, CancellationTokenSource trialCancellationTokenSource);


        public event EventHandler<TrialPerformanceMetrics> PerformanceMetricsUpdated;
    }

    public class DefaultPerformanceMonitor : IPerformanceMonitor
    {
        private readonly IChannel _logger;
        private readonly AutoMLExperiment.AutoMLExperimentSettings _settings;
        private Timer _timer;
        private double? _peakCpuUsage;
        private double? _peakMemoryUsage;
        private readonly int _checkIntervalInMilliseconds;
        private TimeSpan _totalCpuProcessorTime;

        public DefaultPerformanceMonitor(AutoMLExperiment.AutoMLExperimentSettings settings, IChannel logger, int checkIntervalInMilliseconds)
        {
            _settings = settings;
            _logger = logger;
            _checkIntervalInMilliseconds = checkIntervalInMilliseconds;
        }


        public event EventHandler<TrialPerformanceMetrics> PerformanceMetricsUpdated;


        public void Dispose()
        {
            Stop();
        }

        public double? GetPeakCpuUsage()
        {
            return _peakCpuUsage;
        }

        public double? GetPeakMemoryUsageInMegaByte()
        {
            return _peakMemoryUsage;
        }

        public void Start()
        {
            if (_timer == null)
            {
                _timer = new Timer(_checkIntervalInMilliseconds);
                _totalCpuProcessorTime = Process.GetCurrentProcess().TotalProcessorTime;
                _timer.Elapsed += OnCheckCpuAndMemoryUsage;
                _timer.AutoReset = true;
                _logger?.Trace($"{typeof(DefaultPerformanceMonitor)} has been started");
            }

            // trigger the PerformanceMetricsUpdated event and (re)start the timer
            _timer.Enabled = false;
            SampleCpuAndMemoryUsage();
            _timer.Enabled = true;
        }

        public void Pause()
        {
            _timer.Enabled = false;
        }

        public void Stop()
        {
            _timer?.Stop();
            _timer?.Dispose();
            _timer = null;
            _peakCpuUsage = null;
            _peakMemoryUsage = null;
        }

        private void OnCheckCpuAndMemoryUsage(object source, ElapsedEventArgs e)
        {
            SampleCpuAndMemoryUsage();
        }

        private void SampleCpuAndMemoryUsage()
        {
            // calculate CPU usage in %
            // the % of CPU usage is calculating in the following way
            // for every _totalCpuProcessorTime
            // total CPU time is _totalCpuProcessorTime * ProcessorCount
            // total CPU time used by current process is currentCpuProcessorTime
            // the % of CPU usage by current process is simply currentCpuProcessorTime / total CPU time.
            using (var process = Process.GetCurrentProcess())
            {
                var currentCpuProcessorTime = Process.GetCurrentProcess().TotalProcessorTime;
                var elapseCpuProcessorTime = currentCpuProcessorTime - _totalCpuProcessorTime;
                var cpuUsedMs = elapseCpuProcessorTime.TotalMilliseconds;
                var cpuUsageInTotal = cpuUsedMs / (Environment.ProcessorCount * _checkIntervalInMilliseconds);
                _totalCpuProcessorTime = currentCpuProcessorTime;
                _peakCpuUsage = Math.Max(cpuUsageInTotal, _peakCpuUsage ?? 0);

                // calculate Memory Usage in MB
                var memoryUsage = process.WorkingSet64 * 1.0 / (1024 * 1024);
                _peakMemoryUsage = Math.Max(memoryUsage, _peakMemoryUsage ?? 0);

                var metrics = new TrialPerformanceMetrics()
                {
                    CpuUsage = cpuUsageInTotal,
                    MemoryUsage = memoryUsage,
                    PeakCpuUsage = _peakCpuUsage,
                    PeakMemoryUsage = _peakMemoryUsage
                };

                _logger?.Trace($"current CPU: {cpuUsageInTotal}, current Memory(mb): {memoryUsage}");

                PerformanceMetricsUpdated?.Invoke(this, metrics);
            }
        }

        public virtual void OnPerformanceMetricsUpdatedHandler(TrialSettings trialSettings, TrialPerformanceMetrics metrics, CancellationTokenSource trialCancellationTokenSource)
        {
            _logger.Trace($"maximum memory usage: {_settings.MaximumMemoryUsageInMegaByte}, PeakMemoryUsage: {metrics.PeakMemoryUsage} trialIsCancelled: {trialCancellationTokenSource.IsCancellationRequested}");
            if (_settings.MaximumMemoryUsageInMegaByte is double d && metrics.PeakMemoryUsage > d && !trialCancellationTokenSource.IsCancellationRequested)
            {
                _logger.Trace($"cancel current trial {trialSettings.TrialId} because it uses {metrics.PeakMemoryUsage} mb memory and the maximum memory usage is {d}");
                trialCancellationTokenSource.Cancel();

                GC.AddMemoryPressure(Convert.ToInt64(metrics.PeakMemoryUsage) * 1024 * 1024);
                GC.Collect();
            }
        }
    }
}
