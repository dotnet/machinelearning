// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Threading;

namespace Microsoft.ML.AutoML
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
        /// <value>The default value is 86,400, the number of seconds in one day.</value>
        /// <remarks>
        /// An experiment may run for longer than <see name="MaxExperimentTimeInSeconds"/>.
        /// This is because once AutoML starts training an ML.NET model, AutoML lets the
        /// model train to completion. For instance, if the first model
        /// AutoML trains takes 4 hours, and the second model trained takes 5 hours,
        /// but <see name="MaxExperimentTimeInSeconds"/> was the number of seconds in 6 hours,
        /// the experiment will run for 4 + 5 = 9 hours (not 6 hours).
        /// </remarks>
        public uint MaxExperimentTimeInSeconds { get; set; }

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
        public CancellationToken CancellationToken { get; set; }

        /// <summary>
        /// This is the name of the directory where all models trained during the AutoML experiment will be saved.
        /// If <see langword="null"/>, models will be kept in memory instead of written to disk.
        /// (Please note: for an experiment with high runtime operating on a large dataset, opting to keep models in
        /// memory could cause a system to run out of memory.)
        /// </summary>
        /// <value>The default value is the directory named "Microsoft.ML.AutoML" in the in the location specified by the <see cref="MLContext.TempFilePath"/>.</value>
        public string CacheDirectoryName { get; set; }

        /// <summary>
        /// Whether AutoML should cache before ML.NET trainers.
        /// See <see cref="TrainerInfo.WantCaching"/> for more information on caching.
        /// </summary>
        /// <value>The default value is <see cref="CacheBeforeTrainer.Auto"/>.</value>
        public CacheBeforeTrainer CacheBeforeTrainer { get; set; }

        internal int MaxModels;

        /// <summary>
        /// Initializes a new instance of <see cref="ExperimentSettings"/>.
        /// </summary>
        public ExperimentSettings()
        {
            MaxExperimentTimeInSeconds = 24 * 60 * 60;
            CancellationToken = default;
            CacheDirectoryName = "Microsoft.ML.AutoML";
            CacheBeforeTrainer = CacheBeforeTrainer.Auto;
            MaxModels = int.MaxValue;
        }

    }

    /// <summary>
    /// Whether AutoML should cache before ML.NET trainers.
    /// See <see cref="TrainerInfo.WantCaching"/> for more information on caching.
    /// </summary>
    public enum CacheBeforeTrainer
    {
        /// <summary>
        /// Dynamically determine whether to cache before each trainer.
        /// </summary>
        Auto,

        /// <summary>
        /// Always force caching on.
        /// </summary>
        On,

        /// <summary>
        /// Always force caching off.
        /// </summary>
        Off,
    }
}
