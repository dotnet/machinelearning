// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This class represents one data point on Precision-Recall curve.
    /// </summary>
    public sealed class PrecisionRecallMetrics
    {
        /// <summary>
        /// Gets the threshold for this data point
        /// </summary>
        public double Threshold { get; }
        /// <summary>
        /// Gets the precision for the current threshold
        /// </summary>
        public double Precision { get; }
        /// <summary>
        /// Gets the recall for the current threshold
        /// </summary>
        public double Recall { get; }
        /// <summary>
        /// Gets the fpr rate for the given threshold
        /// </summary>
        public double FalsePositiveRate { get; }

        internal PrecisionRecallMetrics(IExceptionContext ectx, DataViewRow overallResult)
        {
            double FetchDouble(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);
            double FetchFloat(string name) => RowCursorUtils.Fetch<float>(ectx, overallResult, name);
            Threshold = FetchFloat(BinaryClassifierEvaluator.Threshold);
            Precision = FetchDouble(BinaryClassifierEvaluator.Precision);
            Recall = FetchDouble(BinaryClassifierEvaluator.Recall);
            FalsePositiveRate = FetchDouble(BinaryClassifierEvaluator.FalsePositiveRate);
        }
    }
}
