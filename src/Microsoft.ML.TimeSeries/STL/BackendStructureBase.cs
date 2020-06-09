using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    public enum TimeSeriesInfoKind
    {
        /// <summary>
        /// (single-time series) the outliers of a single time series
        /// </summary>
        Outlier,

        /// <summary>
        /// (single-time series) the seasonal signal of a single time series
        /// </summary>
        Seasonal,

        /// <summary>
        /// (single-time series) the trend curve of a single time series
        /// </summary>
        Trend,

        /// <summary>
        /// (two-time series) the lead/lag correlation between two time series
        /// </summary>
        CrossCorrelation,

        /// <summary>
        /// (two-time series) the correlation of outliers from two time series
        /// </summary>
        OutlierCorrelation,

        /// <summary>
        /// (two-time series) the correlation of trends from two time series
        /// </summary>
        TrendCorrelation,
    }

    public abstract class TimeSeriesInfoBase
    {
        /// <summary>
        /// each insight should be ranked, so that it can be compared with other insights.
        /// </summary>
        public abstract double Rank { get; set; }

        /// <summary>
        /// the description of this particular insight
        /// </summary>
        public abstract string Description { get; protected set; }

        /// <summary>
        /// indicate the kind of insight.
        /// </summary>
        public abstract TimeSeriesInfoKind Kind { get; protected set; }

        /// <summary>
        /// basic comparison function, used for quick sort.
        /// </summary>
        /// <param name="left">the left element</param>
        /// <param name="right">the right element</param>
        public static int Compare(TimeSeriesInfoBase left, TimeSeriesInfoBase right)
        {
            if (object.ReferenceEquals(left, right))
                return 0;
            if (left == null)
                return -1;
            if (right == null)
                return 1;
            return left.Rank.CompareTo(right.Rank);
        }
    }

    /// <summary>
    /// the characteristic of single time series.
    /// </summary>
    public abstract class SingleSeriesInfo : TimeSeriesInfoBase
    {
        /// <summary>
        /// x-axis values of original curve
        /// </summary>
        public IReadOnlyList<double> X { get; protected set; }

        /// <summary>
        /// y-axis values of original curve
        /// </summary>
        public IReadOnlyList<double> Y { get; protected set; }
    }
}
