namespace Microsoft.ML.TimeSeries
{
    internal class LoessBasicParameters
    {
        /// <summary>
        /// the minimum length of a valid time series. a time series with length equals 2 is so trivial. when less than 2, meaningless.
        /// </summary>
        public const int MinTimeSeriesLength = 3;
    }
}
