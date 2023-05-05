// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.Serialization;
using System.Text.Json.Serialization;
using System.Threading;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Settings used for the trial
    /// </summary>
    public class TrialSettings
    {
        /// <summary>
        /// Identifier of the trial
        /// </summary>
        public int TrialId { get; set; }
        /// <summary>
        /// UTC time when the trial started
        /// </summary>
        public DateTime StartedAtUtc { get; set; }
        /// <summary>
        /// UTC time when the trial ended, null if it's still running
        /// </summary>
        public DateTime? EndedAtUtc { get; set; }
        /// <summary>
        /// Parameters for the pipeline used in this trial
        /// </summary>
        public Parameter Parameter { get; set; }
    }
}
