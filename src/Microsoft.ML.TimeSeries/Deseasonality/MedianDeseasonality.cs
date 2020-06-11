using System;
using System.Collections.Generic;

namespace Microsoft.ML.TimeSeries
{
    internal sealed class MedianDeseasonality : DeseasonalityBase
    {
        private List<double>[] _subSeries;
        private double[] _circularComponent;

        public override void Deseasonality(ref double[] values, int period, ref double[] results)
        {
            AllocateListDoubleArray(period);

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
            AllocateDoubleArray(period);
            for (int i = 0; i < period; ++i)
            {
                _circularComponent[i] = 1;
            }

            // substract the circular component from the original series.
            for (int i = 0; i < length; ++i)
            {
                var indexInPeriod = i % period;
                results[i] -= _circularComponent[indexInPeriod];
            }
        }

        private void AllocateListDoubleArray(int length)
        {
            if (_subSeries == null)
            {
                _subSeries = new List<double>[length];
            }
            else if (_subSeries.Length != length)
            {
                Array.Resize<List<double>>(ref _subSeries, length);
            }
        }

        private void AllocateDoubleArray(int length)
        {
            if (_circularComponent == null)
            {
                _circularComponent = new double[length];
            }
            else if (_circularComponent.Length != length)
            {
                Array.Resize<double>(ref _circularComponent, length);
            }
        }
    }
}
