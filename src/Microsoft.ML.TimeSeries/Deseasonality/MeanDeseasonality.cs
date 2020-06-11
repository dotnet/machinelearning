using System;

namespace Microsoft.ML.TimeSeries.Deseasonality
{
    internal sealed class MeanDeseasonality : DeseasonalityBase
    {
        private double[] _circularComponent;

        public override void Deseasonality(ref double[] values, int period, ref double[] results)
        {
            AllocateDoubleArray(period);
            var length = values.Length;

            // initialize the circurlar component to 0.
            for (int i = 0; i < period; ++i)
            {
                _circularComponent[i] = 0;
            }

            // sum up values that locates at the same position in one period.
            for (int i = 0; i < length; ++i)
            {
                var indexInPeriod = i % period;
                _circularComponent[indexInPeriod] += values[i];
            }

            // calculate the mean value as circular component.
            var cnt = (length - 1) / period;
            var rest = (length - 1) % period;
            for (int i = 0; i < period; ++i)
            {
                var lastCircle = i <= rest ? 1 : 0;
                _circularComponent[i] = _circularComponent[i] / (cnt + lastCircle);
            }

            // substract the circular component from the original series.
            for (int i = 0; i < length; ++i)
            {
                var indexInPeriod = i % period;
                results[i] -= _circularComponent[indexInPeriod];
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
