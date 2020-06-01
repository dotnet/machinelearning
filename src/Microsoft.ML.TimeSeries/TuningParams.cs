using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    public class TuningParams
    {
        public const int ShortTimeseriesLength = 40;
        private const double MagnitudeLower = 0.7;
        private const double MagnitudeUpper = 1.3;

        /// <summary>
        /// dynamically identify a set of outlier categorization params by checking the input time series length
        /// </summary>
        /// <param name="length">the input time series length</param>
        /// <param name="outlierConnectWindowSize">when we determine if an outlier is dense or not, we use the outliers in its neighbors. this value indicates the window size of the neighbors</param>
        /// <param name="denseOutlierThreshold">the threshold of the total outlier count in the window size</param>
        /// <param name="glueGapThreshold">finally, we glue the outliers near each other into a group. this is the threshold to determine "nearby"</param>
        public static void ParamsForCategorizeOutliers(
            int length,
            out int outlierConnectWindowSize,
            out int denseOutlierThreshold,
            out int glueGapThreshold)
        {
            // for typical long time series, we use a set of parameters which are propotional to the entire time series length
            if (length > ShortTimeseriesLength)
            {
                int windowSize = ChangeDetectionUtility.GetWindowSize(length);

                // at least 5% of data points are outliers, which forms reasonable change region. this is one-side window size.
                outlierConnectWindowSize = Math.Max(1, (int)(windowSize * 0.05));

                // the total length is 2 times of outlierConnectWindowSize plus the checking point itself
                int checkSize = (2 * outlierConnectWindowSize) + 1;

                // in the checkSize region, at least 70% points are outliers, then these points formed a dense outlier region
                denseOutlierThreshold = (int)(checkSize * 0.7);

                // the maximum gap between two outliers that we accept gluing them together.
                glueGapThreshold = Math.Max(1, checkSize - denseOutlierThreshold);
            }
            else
            {
                outlierConnectWindowSize = 0;
                denseOutlierThreshold = 1;
                glueGapThreshold = 1;
            }
        }

        /// <summary>
        /// especially for short time series, we don't necessarily output two many outliers.
        /// since there usually exist only few outliers for short time series.
        /// </summary>
        public static int OutputOutlierMaxCount(int length)
        {
            /*all these numbers are tunable magic numbers*/
            return 5;
        }

        /// <summary>
        /// check whether two positive numbers are with similar magnitude or not.
        /// </summary>
        /// <param name="absValue1">the first positive number</param>
        /// <param name="absValue2">the second positive number</param>
        /// <returns>return true if they are with similar magnitude. otherwise, return false</returns>
        public static bool IsSimilarMagnitude(double absValue1, double absValue2)
        {
            if (absValue1 > absValue2)
            {
                // using smaller or equal to handle when both input values are 0
                return absValue1 <= MagnitudeUpper * absValue2;
            }
            else
            {
                // using smaller or equal to handle when both input values are 0
                return absValue1 >= MagnitudeLower * absValue2;
            }
        }
    }
    internal class ChangeDetectionUtility
    {
        /// <summary>
        /// outlier detection for change points. we use this value instead of 6 (recommended by paper) to reduce false-negative
        /// </summary>
        internal const double Severity = 3.5;

        /// <summary>
        /// according to the theory, the window size A should be
        /// lim A/n ->0, lim (logn)^2/A -> 0
        /// </summary>
        private const double Power = 0.6;

        /// <summary>
        /// calculate the proper window size for change point detection, given the total time series length.
        /// stay tuned
        /// </summary>
        /// <param name="n">the total length of time series</param>
        public static int GetWindowSize(int n)
        {
            // at least 0.1n as the window size, make sure at most 5 change points.
            double value = Math.Max(Math.Pow(n, Power), n * 0.1);
            return (int)value;
        }
    }
}
