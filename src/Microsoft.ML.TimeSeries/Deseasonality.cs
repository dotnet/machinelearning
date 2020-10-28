// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.TimeSeries
{
    internal interface IDeseasonality
    {
        /// <summary>
        /// Remove the seasonality component from the given time-series.
        /// </summary>
        /// <param name="values">An array representing the input time-series.</param>
        /// <param name="period">The period value of the time-series.</param>
        /// <param name="results">The de-seasonalized time-series.</param>
        public abstract void Deseasonality(ref double[] values, int period, ref double[] results);
    }

    internal sealed class MeanDeseasonality : IDeseasonality
    {
        private double[] _circularComponent;

        public void Deseasonality(ref double[] values, int period, ref double[] results)
        {
            Array.Resize(ref _circularComponent, period);

            var length = values.Length;

            // Initialize the circular component to 0.
            for (int i = 0; i < period; ++i)
            {
                _circularComponent[i] = 0;
            }

            // Sum up values that locate at the same position in one period.
            for (int i = 0; i < length; ++i)
            {
                var indexInPeriod = i % period;
                _circularComponent[indexInPeriod] += values[i];
            }

            // Calculate the mean value as circular component.
            var cnt = (length - 1) / period;
            var rest = (length - 1) % period;
            for (int i = 0; i < period; ++i)
            {
                var lastCircle = i <= rest ? 1 : 0;
                _circularComponent[i] = _circularComponent[i] / (cnt + lastCircle);
            }

            // Substract the circular component from the original series.
            for (int i = 0; i < length; ++i)
            {
                var indexInPeriod = i % period;
                results[i] -= _circularComponent[indexInPeriod];
            }
        }
    }

    internal sealed class MedianDeseasonality : IDeseasonality
    {
        private List<double>[] _subSeries;
        private double[] _circularComponent;

        public void Deseasonality(ref double[] values, int period, ref double[] results)
        {
            Array.Resize(ref _circularComponent, period);
            Array.Resize(ref _subSeries, period);

            var length = values.Length;

            for (int i = 0; i < period; ++i)
            {
                _subSeries[i] = new List<double>();
            }

            // Split the original series into #period subseries.
            for (int i = 0; i < length; ++i)
            {
                var indexInPeriod = i % period;
                _subSeries[indexInPeriod].Add(values[i]);
            }

            // Calculate the median value as circular component.
            for (int i = 0; i < period; ++i)
            {
                _circularComponent[i] = MathUtility.QuickMedian(_subSeries[i]);
            }

            // Substract the circular component from the original series.
            for (int i = 0; i < length; ++i)
            {
                var indexInPeriod = i % period;
                results[i] -= _circularComponent[indexInPeriod];
            }
        }
    }

    /// <summary>
    /// This class takes the residual component of stl decompose as the deseasonality result.
    /// </summary>
    internal sealed class StlDeseasonality : IDeseasonality
    {
        private readonly InnerStl _stl;
        private readonly IDeseasonality _backupFunc;

        public StlDeseasonality()
        {
            _stl = new InnerStl(true);
            _backupFunc = new MedianDeseasonality();
        }

        public void Deseasonality(ref double[] values, int period, ref double[] results)
        {
            bool success = _stl.Decomposition(values, period);
            if (success)
            {
                for (int i = 0; i < _stl.Residual.Count; ++i)
                {
                    results[i] = _stl.Residual[i];
                }
            }
            // invoke the back up deseasonality method if stl decompose fails.
            else
            {
                _backupFunc.Deseasonality(ref values, period, ref results);
            }
        }
    }
}
