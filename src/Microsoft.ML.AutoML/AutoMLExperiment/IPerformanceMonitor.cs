// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using System.Timers;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    internal interface IPerformanceMonitor : IDisposable
    {
        void Start();

        void Stop();

        double? GetPeakMemoryUsageInMegaByte();

        double? GetPeakCpuUsage();

        public event EventHandler<double> CpuUsage;

        public event EventHandler<double> MemoryUsageInMegaByte;
    }

    internal class DefaultPerformanceMonitor : IPerformanceMonitor
    {
        private readonly IChannel _logger;
        private Timer _timer;
        private double? _peakCpuUsage;
        private double? _peakMemoryUsage;
        private readonly int _checkIntervalInMilliseconds;
        private TimeSpan _totalCpuProcessorTime;

        public DefaultPerformanceMonitor(IChannel logger, int checkIntervalInMilliseconds)
        {
            _logger = logger;
            _checkIntervalInMilliseconds = checkIntervalInMilliseconds;
        }


        public event EventHandler<double> CpuUsage;

        public event EventHandler<double> MemoryUsageInMegaByte;


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
                _timer.Enabled = true;
                _logger?.Trace($"{typeof(DefaultPerformanceMonitor)} has been started");
            }
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
                var memoryUsage = process.PrivateMemorySize64 * 1.0 / (1024 * 1024);
                _peakMemoryUsage = Math.Max(memoryUsage, _peakMemoryUsage ?? 0);
                _logger?.Trace($"current CPU: {cpuUsageInTotal}, current Memory(mb): {memoryUsage}");
                MemoryUsageInMegaByte?.Invoke(this, memoryUsage);
                CpuUsage?.Invoke(this, cpuUsageInTotal);
            }
        }
    }
}
