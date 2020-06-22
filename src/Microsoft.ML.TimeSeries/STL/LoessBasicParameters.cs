namespace Microsoft.ML.TimeSeries
{
    internal class LoessBasicParameters
    {
        /// <summary>
        /// The minimum length of a valid time series. A time series with length equals 2 is so trivial and meaningless less than 2.
        /// </summary>
        public const int MinTimeSeriesLength = 3;
    }
}
