using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    public class SeasonalInfo : SingleSeriesInfo
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SeasonalInfo"/> class.
        /// the corresponding properties of seasonal component
        /// </summary>
        /// <param name="x">x</param>
        /// <param name="y">y</param>
        /// <param name="seasonal">the seasonal component</param>
        /// <param name="period">period of seasonal component</param>
        /// <param name="amplitude">the average amplitude of the seasonal component</param>
        /// <param name="confidence">the statistical confidence</param>
        public SeasonalInfo(
            IReadOnlyList<double> x,
            IReadOnlyList<double> y,
            IReadOnlyList<double> seasonal,
            int period,
            double amplitude,
            double confidence)
        {
            X = x;
            Y = y;
            SeasonalSignal = seasonal;
            Period = period;
            Amplitude = amplitude;

            // since seasonal component has sound statistical modeling and tests, so its rank is just the statistical confidence
            Rank = confidence;

            Description = String.Format(
                "this is a seasonal component, with period = {0}, amplitude = {1}. rank:{2}",
                Period,
                Amplitude,
                Rank);

            Kind = TimeSeriesInfoKind.Seasonal;
        }

        /// <summary>
        /// the curve of seasonal
        /// </summary>
        public IReadOnlyList<double> SeasonalSignal { get; private set; }

        /// <summary>
        /// the period of the seasonal component.
        /// </summary>
        public int Period { get; private set; }

        /// <summary>
        /// the average amplitude of the seasonal component
        /// </summary>
        public double Amplitude { get; private set; }

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

        public override string ToString()
        {
            return Description;
        }
    }
}
