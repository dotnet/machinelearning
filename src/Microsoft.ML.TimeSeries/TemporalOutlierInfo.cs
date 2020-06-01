using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// the outliers of time series. A default rank is available when this instance is constructed.
    /// </summary>
    public class TemporalOutlierInfo : SingleSeriesInfo
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TemporalOutlierInfo"/> class.
        /// the corresponding properties of outlier component
        /// </summary>
        /// <param name="x">x axis values</param>
        /// <param name="y">y axis values</param>
        /// <param name="residual">the residual of the curve. which is the data source for outlier identification</param>
        /// <param name="index">the indication for outliers</param>
        /// <param name="severity">severity</param>
        /// <param name="confidence">confidence</param>
        public TemporalOutlierInfo(
            IReadOnlyList<double> x,
            IReadOnlyList<double> y,
            IReadOnlyList<double> residual,
            IReadOnlyList<int> index,
            IReadOnlyList<double> severity,
            double confidence)
        {
            X = x;
            Y = y;
            Residual = residual;
            OutlierIndex = index;
            OutlierSeverity = severity;
            Rank = confidence;

            Count = 0;
            foreach (int indicator in index)
            {
                if (indicator == 1)
                    Count++;
            }

            Description = String.Format("there exists {0} outliers. Rank:{1}", Count, Rank);

            Kind = TimeSeriesInfoKind.Outlier;
        }

        /// <summary>
        /// 0/1 value for each data point, to indicate whether it is an outlier or not.
        /// 1 means outlier, 0 means not.
        /// </summary>
        public IReadOnlyList<int> OutlierIndex { get; private set; }

        /// <summary>
        /// equal length with OutlierIndex, indicate the severity of each outlier. 0 for non-outlier epochs.
        /// </summary>
        public IReadOnlyList<double> OutlierSeverity { get; private set; }

        /// <summary>
        /// the residual values, which is the signal after decompose trend and seasonal signal.
        /// </summary>
        public IReadOnlyList<double> Residual { get; private set; }

        /// <summary>
        /// total count of outliers
        /// </summary>
        public int Count { get; private set; }

        public override string Description
        {
            get;
            protected set;
        }

        public override double Rank
        {
            get;
            set;
        }

        public override TimeSeriesInfoKind Kind
        {
            get;
            protected set;
        }
    }
}
