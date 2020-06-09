namespace Microsoft.ML.TimeSeries
{
    public class BasicParameters
    {
        /// <summary>
        /// the minimum length of a valid time series. a time series with length equals 2 is so trivial. when less than 2, meaningless.
        /// </summary>
        public const int MinTimeSeriesLength = 3;

        /// <summary>
        /// the maximum length of a valid time series. when there are too many data points, the chart will look so dense that details are lost.
        /// this number is tuned so that the bird strike data can still preserve results.
        /// </summary>
        public const int MaxTimeSeriesLength = 4000;

        /// <summary>
        /// the minimum count of repeated periods. this is used for determining a noticeable seasonal signal.
        /// </summary>
        public const int MinPeriodRepeatCount = 3;

        /// <summary>
        /// the minimum count of regular gaps. when there are too few gaps, the time series will look odd, which will impact the seasonality analysis
        /// </summary>
        public const int MinRegularGap = 5;
    }
}
