// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Threading;

namespace Microsoft.ML.Auto
{
    public class ExperimentSettings
    {
        public uint MaxExperimentTimeInSeconds { get; set; } = 24 * 60 * 60;
        public CancellationToken CancellationToken { get; set; } = default;

        /// <summary>
        /// This is a pointer to a directory where all models trained during the AutoML experiment will be saved.
        /// If null, models will be kept in memory instead of written to disk.
        /// (Please note: for an experiment with high runtime operating on a large dataset, opting to keep models in 
        /// memory could cause a system to run out of memory.)
        /// </summary>
        public DirectoryInfo ModelDirectory { get; set; } = null;

        /// This setting controls whether or not an AutoML experiment will make use of ML.NET-provided caching.
        /// If set to true, caching will be forced on for all pipelines. If set to false, caching will be forced off.
        /// If set to null (default value), AutoML will decide whether to enable caching for each model.
        /// </summary>
        public bool? EnableCaching = null;

        internal int MaxModels = int.MaxValue;
        internal IDebugLogger DebugLogger;
    }
}
