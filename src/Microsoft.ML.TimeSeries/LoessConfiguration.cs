using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    public enum RegressionModelType
    {
        /// <summary>
        /// the 1-order model, i.e., linear model
        /// </summary>
        One,

        /// <summary>
        /// the 2-order model, i.e., square polynomial model.
        /// </summary>
        Two,
    }

    /// <summary>
    /// this class is used to store the parameters which are needed for lowess algorithm.
    /// the name of these constansts are compliant with the original terms in paper.
    /// </summary>
    public class LoessConfiguration
    {
        /// <summary>
        /// this value is used for performance concern. when the length of the series goes large, a ratio of neighbors will be significant,
        /// which leads to unsatisfied slow. so this value is used to bound the maximum # of neighbors one epoch can have.
        /// </summary>
        public const int MaximumNeighborCount = 100;

        /// <summary>
        /// minumum number of neighbor counts, to apply underlying regression analysis.
        /// this number should be even, so that neighbors on left/right side of a given data point is balanced. unbalanced neighbors would make the local-weighted regression biased noticeably at corner cases.
        /// </summary>
        public const int MinimumNeighborCount = 4;

        /// <summary>
        /// (0, 1], a smooth range ratio. let fn be the number of neighbors of a specific point.
        /// </summary>
        public static readonly double F = 0.3;

        /// <summary>
        /// this is used to indicate which regression model is used.
        /// </summary>
        public static readonly RegressionModelType ModelType = RegressionModelType.One;

        /// <summary>
        /// the number of iterations for robust regression.
        /// </summary>
        public static readonly int T = 2;
    }
}
