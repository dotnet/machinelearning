using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    internal class StlConfiguration
    {
        /// <summary>
        /// the smoothing parameter for the seasonal component.
        /// should be odd, and at least 7.
        /// </summary>
        public const int Ns = 9;

        /// <summary>
        /// the number of passes through the inner loop. /ref this value is set to 2, which works for many cases
        /// </summary>
        public const int Ni = 2;

        /// <summary>
        /// the number of robustness iterations of the outer loop
        /// </summary>
        public const int No = 10;

        public StlConfiguration()
        {
            Np = -1;
        }

        public StlConfiguration(int np)
        {
            Np = np;
        }

        /// <summary>
        /// the number of observations in each cycle of the seasonal component
        /// </summary>
        public int Np { get; }

        /// <summary>
        /// the smoothing parameter for the low-pass filter.
        /// /ref: should be the least odd integer greater than or equal to np.
        /// it will preventing the trend and seasonal components from competing for the same variation in the data.
        /// </summary>
        public int Nl
        {
            get
            {
                if (Np % 2 == 0)
                    return Np + 1;
                return Np;
            }
        }

        /// <summary>
        /// the smoothing parameter for the trend component.
        /// /ref: in order to avoid the trend ans seasonal components compete for variation in the data, the nt should be chosen
        /// s.t., satisty the following inequality.
        /// </summary>
        public int Nt
        {
            get
            {
                double value = 1.5 * Np / (1.0 - 1.5 / StlConfiguration.Ns);
                int result = (int)value + 1;
                if (result % 2 == 0)
                    result++;
                return result;
            }
        }
    }
}
