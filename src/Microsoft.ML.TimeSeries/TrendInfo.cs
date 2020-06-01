using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// the trend component of time series. and corresponding mathematical properties of the trend. A default rank is available when this instance is constructed.
    /// </summary>
    public class TrendInfo : SingleSeriesInfo
    {
        private const double _slopeThreshold = 0.25;
        private const double _msrThreshold = 0.1;

        /// <summary>
        /// Initializes a new instance of the <see cref="TrendInfo"/> class.
        /// insight item for trend component
        /// </summary>
        /// <param name="x">x-axis values of original curve</param>
        /// <param name="y">y-axis values of original curve</param>
        /// <param name="trend">the curve of trend</param>
        /// <param name="mrs">mean residual squares. which measure the quality of regression model fitting to original curve.</param>
        public TrendInfo(IReadOnlyList<double> x, IReadOnlyList<double> y, IReadOnlyList<double> trend, double mrs)
        {
            X = x;
            Y = y;
            Trend = trend;

            int length = Trend.Count;

            // double slope = (this.Trend[length - 1] - this.Trend[0]) / (length - 1);
            double increase = Trend[length - 1] - Trend[0];

            // get relative slope, which is unit-invariant
            double y0 = Y[0];
            if (Math.Abs(y0) < 0.01)
                y0 = 0.01;

            // this is so called relative slope.
            Slope = increase / y0;
            IsIncrease = Slope > 0;

            // get relative mrs, which is unit-invariant. now the mrs is caclulated directly from original signal rather than trend. which is compliant with user perception.
            double average = 0;
            foreach (double value in Y)
                average += value;
            average /= Y.Count;
            if (Math.Abs(average) < 0.01)
                average = 0.01;

            // conducted normalization
            Mrs = mrs / average / average;
            IncreaseRatio = CalculateIncreasingRatio(Trend);

            // 0.15 is a magic number
            double slopeSignificance = MathUtility.Sigmoid(50 * (Math.Abs(Slope) - 0.15));

            // the lower the variance, the better the trend. 0.10 is a magic number.
            double mesSignificance = 1 - MathUtility.Sigmoid(50 * (mrs - 0.10));
            double consistencySignificance = 1.0;
            if (IsIncrease)
            {
                if (IncreaseRatio <= 0.5)
                    consistencySignificance = 0;

                // 0.667 is a magic number
                consistencySignificance = MathUtility.Sigmoid(50 * (IncreaseRatio - 0.667));
            }
            else
            {
                if (IncreaseRatio >= 0.5)
                    consistencySignificance = 0;

                // 0.667 is a magic number
                consistencySignificance = MathUtility.Sigmoid(50 * (1.0 - IncreaseRatio - 0.667));
            }
            Rank = slopeSignificance * mesSignificance * consistencySignificance;

            Description = String.Format("this is a trend, with slope = {0}, Mrs:{1}. rank:{2}", Slope, Mrs, Rank);

            Kind = TimeSeriesInfoKind.Trend;
        }

        /// <summary>
        /// the curve of trend
        /// </summary>
        public IReadOnlyList<double> Trend { get; private set; }

        /// <summary>
        /// the estimated relative slope of the trend.
        /// </summary>
        public double Slope { get; private set; }

        /// <summary>
        /// relative mean residual squares. which measure the quality of regression model fitting to original curve.
        /// this is already normalized
        /// </summary>
        public double Mrs { get; private set; }

        /// <summary>
        /// indicate whether the curve is overall increasing or not.
        /// this is directly derived from "Slope". if slope is positive, this is true.
        /// </summary>
        public bool IsIncrease { get; set; }

        /// <summary>
        /// a percentage to indicate in the trend, what's the ratio of local increasing, i.e., delta is positive means increasing.
        /// </summary>
        public double IncreaseRatio { get; set; }

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

        public static double CalcSignificance(
            IReadOnlyList<double> trend,
            IReadOnlyList<double> rawSignal,
            double mrs,
            out double slope,
            out double increaseRatio,
            out double decreaseRatio)
        {
            // we must assume the length of x, and y are same.
            int length = trend.Count;

            // calculate the slope. which is unit-invariant
            double increase = trend[length - 1] - trend[0];

            // let's try to use this value
            double y0 = rawSignal[0];
            if (Math.Abs(y0) <= 0.01)
                y0 = 0.01;
            slope = increase / y0;
            bool isIncrease = slope > 0;

            // when the time series is short, we should apply the consistency checking on the original signal, so that the result will be compliant with user perception.
            if (trend.Count < TuningParams.ShortTimeseriesLength)
            {
                CalculateRatio(rawSignal, out increaseRatio, out decreaseRatio);
            }
            else
            {
                CalculateRatio(trend, out increaseRatio, out decreaseRatio);
            }

            double slopeSignificance = MathUtility.Sigmoid(50 * (Math.Abs(slope) - _slopeThreshold));

            double mesSignificance = 1 - MathUtility.Sigmoid(50 * (mrs - _msrThreshold));
            double consistencySignificance = 1.0;
            double consistencyThreshold = ConsistencyThreshold(trend.Count);
            if (isIncrease)
            {
                if (increaseRatio <= 0.5)
                    consistencySignificance = 0;
                else
                    consistencySignificance = MathUtility.Sigmoid(50 * (increaseRatio - consistencyThreshold));
            }
            else
            {
                if (decreaseRatio <= 0.5)
                    consistencySignificance = 0;
                else
                    consistencySignificance = MathUtility.Sigmoid(50 * (decreaseRatio - consistencyThreshold));
            }
            return slopeSignificance * mesSignificance * consistencySignificance;
        }

        /// <summary>
        /// calculate the increasing ratio. total number of positive deltas divided by total deltas
        /// </summary>
        private static double CalculateIncreasingRatio(IReadOnlyList<double> trend)
        {
            if (trend.Count <= 1)
                return 0;
            int count = 0;
            for (int i = 0; i < trend.Count - 1; i++)
            {
                if (trend[i + 1] - trend[i] > 0)
                    count++;
            }
            return count * 1.0 / (trend.Count - 1);
        }

        /// <summary>
        /// calculate the increasing/decreasing ratio.
        /// </summary>
        private static void CalculateRatio(IReadOnlyList<double> signal, out double increaseRatio, out double decreaseRatio)
        {
            increaseRatio = 0;
            decreaseRatio = 0;
            if (signal.Count <= 1)
                return;
            int increaseCount = 0;
            int decreaseCount = 0;
            for (int i = 0; i < signal.Count - 1; i++)
            {
                if (signal[i + 1] - signal[i] > 0)
                    increaseCount++;
                else if (signal[i + 1] - signal[i] < 0)
                    decreaseCount++;
            }
            increaseRatio = increaseCount * 1.0 / (signal.Count - 1);
            decreaseRatio = decreaseCount * 1.0 / (signal.Count - 1);
        }

        private static double ConsistencyThreshold(int length)
        {
            if (length < TuningParams.ShortTimeseriesLength)
                return 0.8;
            else
                return 0.7;
        }
    }
}
