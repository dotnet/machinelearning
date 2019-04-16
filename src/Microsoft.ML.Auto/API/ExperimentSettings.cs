// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Threading;

namespace Microsoft.ML.Auto
{
    /// <summary>
    /// Base class for experiment settings. All task-specific AutoML experiment settings
    /// (like <see cref="BinaryExperimentSettings"/>) inherit from this class.
    /// </summary>
    public abstract class ExperimentSettings
    {
        /// <summary>
        /// Maximum time in seconds the experiment is allowed to run.
        /// </summary>
        /// <remarks>
        /// An experiment may run for longer than <see name="MaxExperimentTimeInSeconds"/>.
        /// This is because once AutoML starts training an ML.NET model, AutoML lets the
        /// model train to completion. For instance, if the first model 
        /// AutoML trains takes 4 hours, and the second model trained takes 5 hours,
        /// but <see name="MaxExperimentTimeInSeconds"/> was the number of seconds in 6 hours, 
        /// the experiment will run for 4 + 5 = 9 hours (not 6 hours).
        /// </remarks>
        public uint MaxExperimentTimeInSeconds { get; set; } = 24 * 60 * 60;

        /// <summary>
        /// Cancellation token for the AutoML experiment. It propagates the notification
        /// that the experiment should be canceled.
        /// </summary>
        /// <remarks>
        /// An experiment may not immediately stop after cancellation.
        /// This is because once AutoML starts training an ML.NET model, AutoML lets the
        /// model train to completion. For instance, if the first model 
        /// AutoML trains takes 4 hours, and the second model trained takes 5 hours,
        /// but cancellation is requested after 6 hours, 
        /// the experiment will stop after 4 + 5 = 9 hours (not 6 hours).
        /// </remarks>
        public CancellationToken CancellationToken { get; set; } = default;

        /// <summary>
        /// This is a pointer to a directory where all models trained during the AutoML experiment will be saved.
        /// If <see langword="null"/>, models will be kept in memory instead of written to disk.
        /// (Please note: for an experiment with high runtime operating on a large dataset, opting to keep models in 
        /// memory could cause a system to run out of memory.)
        /// </summary>
        public DirectoryInfo CacheDirectory { get; set; } = new DirectoryInfo(Path.Combine(Path.GetTempPath(), "Microsoft.ML.Auto"));

        /// <summary>
        /// This setting controls whether or not an AutoML experiment will make use of ML.NET-provided caching.
        /// If set to true, caching will be forced on for all pipelines. If set to false, caching will be forced off.
        /// If set to <see langword="null"/> (default value), AutoML will decide whether to enable caching for each model.
        /// </summary>
        public bool? CacheBeforeTrainer = null;
        
        internal int MaxModels = int.MaxValue;
        internal IDebugLogger DebugLogger;
    }
}
