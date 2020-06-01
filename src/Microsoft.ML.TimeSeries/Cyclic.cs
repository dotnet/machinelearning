using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    public class Cyclic
    {
        private readonly IReadOnlyList<double> _y;

        public Cyclic(IReadOnlyList<double> yValues)
        {
            //ExtendedDiagnostics.EnsureArgumentNotNull(yValues, nameof(yValues));

            if (yValues.Count < BasicParameters.MinTimeSeriesLength)
                throw new Exception("input data structure cannot be 0-length: cyclic");

            _y = yValues;
        }

        /// <summary>
        /// detect the cyclic length by given the input time series.
        /// if not exist (the cyclic pattern is not significant), then return -1
        /// </summary>
        public int DetectCyclic(out double confidence)
        {
            return SerialCorrelation.Period(_y, out confidence);
        }
    }
}
