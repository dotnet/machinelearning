﻿using System;
using System.Collections.Generic;

namespace Microsoft.ML.TimeSeries
{
    internal sealed class MedianDeseasonality : DeseasonalityBase
    {
        private List<double>[] _subSeries;
        private double[] _circularComponent;

        public override void Deseasonality(ref double[] values, int period, ref double[] results)
        {
            Array.Resize(ref _circularComponent, period);
            Array.Resize(ref _subSeries, period);

            var length = values.Length;

            for (int i = 0; i < period; ++i)
            {
                _subSeries[i] = new List<double>();
            }

            // split the original series into #period subseries.
            for (int i = 0; i < length; ++i)
            {
                var indexInPeriod = i % period;
                _subSeries[indexInPeriod].Add(values[i]);
            }

            // calculate the median value as circular component.
            for (int i = 0; i < period; ++i)
            {
                _circularComponent[i] = MathUtility.QuickMedian(_subSeries[i]);
            }

            // substract the circular component from the original series.
            for (int i = 0; i < length; ++i)
            {
                var indexInPeriod = i % period;
                results[i] -= _circularComponent[indexInPeriod];
            }
        }
    }
}
