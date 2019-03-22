// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This class represents one data point on Precision-Recall curve for binary classification.
    /// </summary>
    public sealed class BinaryPrecisionRecallDataPoint
    {
        /// <summary>
        /// Gets the threshold for this data point.
        /// </summary>
        public double Threshold { get; }
        /// <summary>
        /// Gets the precision for the current threshold.
        /// </summary>
        public double Precision { get; }
        /// <summary>
        /// Gets the recall for the current threshold.
        /// </summary>
        public double Recall { get; }

        /// <summary>
        /// Gets the true positive rate for the current threshold.
        /// </summary>
        public double TruePositiveRate => Recall;

        /// <summary>
        /// Gets the false positive rate for the given threshold.
        /// </summary>
        public double FalsePositiveRate { get; }

        internal BinaryPrecisionRecallDataPoint(ValueGetter<float> thresholdGetter, ValueGetter<double> precisionGetter, ValueGetter<double> recallGetter, ValueGetter<double> fprGetter)
        {
            float threshold = default;
            double precision = default;
            double recall = default;
            double fpr = default;

            thresholdGetter(ref threshold);
            precisionGetter(ref precision);
            recallGetter(ref recall);
            fprGetter(ref fpr);

            Threshold = threshold;
            Precision = precision;
            Recall = recall;
            FalsePositiveRate = fpr;
        }
    }

}
