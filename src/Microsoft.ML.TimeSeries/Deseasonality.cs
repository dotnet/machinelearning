using System;
using System.Collections.Generic;

namespace Microsoft.ML.TimeSeries
{
    internal interface IDeseasonality
    {
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

            // Sum up values that locates at the same position in one period.
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

    internal sealed class StlDeseasonality : IDeseasonality
    {
        private readonly InnerStl _stl;

        public StlDeseasonality()
        {
            _stl = new InnerStl(true);
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
            else
            {
                for (int i = 0; i < values.Length; ++i)
                {
                    results[i] = values[i];
                }
            }
        }
    }
}
