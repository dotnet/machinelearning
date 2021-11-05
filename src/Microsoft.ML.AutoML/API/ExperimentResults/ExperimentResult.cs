// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Result of an AutoML experiment.
    /// </summary>
    /// <typeparam name="TMetrics">Metrics type for the experiment (like <see cref="BinaryClassificationMetrics"/>).</typeparam>
    public class ExperimentResult<TMetrics>
    {
        /// <summary>
        /// Details of the runs in this experiment.
        /// </summary>
        /// <remarks>
        /// See <see cref="RunDetail{TMetrics}"/> for more information.
        /// </remarks>
        public readonly IEnumerable<RunDetail<TMetrics>> RunDetails;

        /// <summary>
        /// Best run in this experiment.
        /// </summary>
        /// <remarks>
        /// AutoML considers the optimizing metric (like <see cref="BinaryExperimentSettings.OptimizingMetric"/>)
        /// when determining the best run.
        /// </remarks>
        public readonly RunDetail<TMetrics> BestRun;

        internal ExperimentResult(IEnumerable<RunDetail<TMetrics>> runDetails,
            RunDetail<TMetrics> bestRun)
        {
            RunDetails = runDetails;
            BestRun = bestRun;
        }
    }
}
