// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using static Microsoft.ML.Runtime.Data.MetricKinds;

namespace Microsoft.ML.Models
{
    /// <summary>
    /// This class contains the overall metrics computed by regression evaluators.
    /// </summary>
    public sealed class RegressionMetrics
    {
        private RegressionMetrics()
        {
        }

        internal static List<RegressionMetrics> FromOverallMetrics(IHostEnvironment env, IDataView overallMetrics)
        {
            Contracts.AssertValue(env);
            env.AssertValue(overallMetrics);

            var metricsEnumerable = overallMetrics.AsEnumerable<SerializationClass>(env, true, ignoreMissingColumns: true);
            if (!metricsEnumerable.GetEnumerator().MoveNext())
            {
                throw env.Except("The overall RegressionMetrics didn't have any rows.");
            }

            List<RegressionMetrics> metrics = new List<RegressionMetrics>();
            foreach (var metric in metricsEnumerable)
            {
                metrics.Add(new RegressionMetrics()
                {
                    L1 = metric.L1,
                    L2 = metric.L2,
                    Rms = metric.Rms,
                    LossFn = metric.LossFn,
                    RSquared = metric.RSquared,
                    FoldIndex = metric.FoldIndex,
                });
            }

            return metrics;
        }

        /// <summary>
        /// Gets the absolute loss of the model.
        /// </summary>
        /// <remarks>
        /// The absolute loss is defined as
        /// L1 = (1/m) * sum( abs( yi - y'i))
        /// where m is the number of instances in the test set.
        /// y'i are the predicted labels for each instance.
        /// yi are the correct labels of each instance.
        /// </remarks>
        public double L1 { get; private set; }

        /// <summary>
        /// Gets the squared loss of the model.
        /// </summary>
        /// <remarks>
        /// The squared loss is defined as
        /// L2 = (1/m) * sum(( yi - y'i)^2)
        /// where m is the number of instances in the test set.
        /// y'i are the predicted labels for each instance.
        /// yi are the correct labels of each instance.
        /// </remarks>
        public double L2 { get; private set; }

        /// <summary>
        /// Gets the root mean square loss (or RMS) which is the square root of the L2 loss.
        /// </summary>
        public double Rms { get; private set; }

        /// <summary>
        /// Gets the user defined loss function.
        /// </summary>
        /// <remarks>
        /// This is the average of a loss function defined by the user,
        /// computed over all the instances in the test set.
        /// </remarks>
        public double LossFn { get; private set; }

        /// <summary>
        /// Gets the R squared value of the model, which is also known as
        /// the coefficient of determination​.
        /// </summary>
        public double RSquared { get; private set; }

        /// <summary>
        /// For cross validation gives fold number or standard deviation or average across all metrics.
        /// In other cases equal to null.
        /// </summary>
        public string FoldIndex { get; private set; }

        /// <summary>
        /// This class contains the public fields necessary to deserialize from IDataView.
        /// </summary>
        private sealed class SerializationClass
        {
#pragma warning disable 649 // never assigned
            [ColumnName(Runtime.Data.RegressionEvaluator.L1)]
            public Double L1;

            [ColumnName(Runtime.Data.RegressionEvaluator.L2)]
            public Double L2;

            [ColumnName(Runtime.Data.RegressionEvaluator.Rms)]
            public Double Rms;

            [ColumnName(Runtime.Data.RegressionEvaluator.Loss)]
            public Double LossFn;

            [ColumnName(Runtime.Data.RegressionEvaluator.RSquared)]
            public Double RSquared;

            [ColumnName(ColumnNames.FoldIndex)]
            public string FoldIndex;
#pragma warning restore 649 // never assigned
        }
    }
}
