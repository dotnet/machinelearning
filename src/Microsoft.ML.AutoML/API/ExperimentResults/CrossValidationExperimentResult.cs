// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Result of an AutoML experiment that includes cross validation details.
    /// </summary>
    /// <typeparam name="TMetrics">Metrics type for the experiment (like <see cref="BinaryClassificationMetrics"/>).</typeparam>
    public class CrossValidationExperimentResult<TMetrics> : IDisposable
    {
        /// <summary>
        /// Details of the cross validation runs in this experiment.
        /// </summary>
        /// <remarks>
        /// See <see cref="CrossValidationRunDetail{TMetrics}"/> for more information.
        /// </remarks>
        public readonly IEnumerable<CrossValidationRunDetail<TMetrics>> RunDetails;

        /// <summary>
        /// Best run in this experiment.
        /// </summary>
        /// <remarks>
        /// AutoML considers the optimizing metric (like <see cref="BinaryExperimentSettings.OptimizingMetric"/>)
        /// when determining the best run.
        /// </remarks>
        public readonly CrossValidationRunDetail<TMetrics> BestRun;

        internal CrossValidationExperimentResult(IEnumerable<CrossValidationRunDetail<TMetrics>> runDetails,
            CrossValidationRunDetail<TMetrics> bestRun)
        {
            RunDetails = runDetails;
            BestRun = bestRun;
        }

        #region IDisposable Support
        private bool _disposed;
        private bool _disposedRunDetails;

        /// <summary>
        /// Releases unmanaged Tensor objects in models stored in RunDetail and BestRun instances
        /// </summary>
        /// <remarks>
        /// Invocation of Dispose() is necessary to clean up remaining C library Tensor objects and
        /// avoid a memory leak
        /// </remarks>
        public void Dispose()
        {
            if (_disposed)
                return;
            if (!_disposedRunDetails)
                foreach (var runDetail in RunDetails)
                    foreach(var result in runDetail.Results)
                        (result.Model as IDisposable)?.Dispose();
            foreach (var result in BestRun.Results)
                (result.Model as IDisposable)?.Dispose();
            (RunDetails as IDisposable)?.Dispose();
            (BestRun as IDisposable)?.Dispose();
            _disposed = true;
            _disposedRunDetails = true;
        }

        /// <summary>
        /// Releases unmanaged Tensor objects in models stored in RunDetail instances
        /// </summary>
        /// <remarks>
        /// Invocation of DisposeRunDetails() is necessary to clean up remaining C library
        /// Tensor objects and avoid a memory leak. Compared to Dispose(), DisposeRunDetails()
        /// only disposes of RunDetails so that the best determined model in BestRun can have
        /// a different lifetime than models in other experimental runs.
        /// </remarks>
        public void DisposeRunDetails()
        {
            if (_disposedRunDetails || _disposed)
                return;
            foreach (var runDetail in RunDetails)
                if (!runDetail.Equals(BestRun))
                    foreach (var result in runDetail.Results)
                        (result.Model as IDisposable)?.Dispose();
            (RunDetails as IDisposable)?.Dispose();
            _disposedRunDetails = true;
        }
        #endregion
    }
}
