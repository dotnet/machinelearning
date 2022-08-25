// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoML
{
    public interface IPerformanceMonitor : IDisposable
    {
        void Start();

        void Stop();

        float GetPeakMemoryUsageInMegaByte();

        float GetPeakCpuUsage();

        //event EventHandler<float> CpuUsage;

        //event EventHandler<float> MemoryUsageInMegaByte;
    }

    internal class DefaultPerformanceMonitor : IPerformanceMonitor
    {
        //public event EventHandler<float> CpuUsage;

        //public event EventHandler<float> MemoryUsageInMegaByte;
        public void Dispose()
        {
            throw new NotImplementedException();
        }

        public float GetPeakCpuUsage()
        {
            throw new NotImplementedException();
        }

        public float GetPeakMemoryUsageInMegaByte()
        {
            throw new NotImplementedException();
        }

        public void Start()
        {
            throw new NotImplementedException();
        }

        public void Stop()
        {
            throw new NotImplementedException();
        }
    }
}
