// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Tokenizers
{
    public delegate void ReportProgress(Progress progress);

    /// <summary>
    /// Represent the state of the reported progress.
    /// </summary>
    public enum ProgressState
    {
        /// <summary>
        /// The progress is started. The reported value in the Progress structure will have the max number progressing toward.
        /// </summary>
        Start,

        /// <summary>
        /// The progress is ended. The reported value in the Progress structure will have the final max processed number.
        /// </summary>
        End,

        /// <summary>
        /// The progress is incremented. The reported value in increment value in the progress.
        /// </summary>
        Increment
    }

    public readonly struct Progress
    {
        /// <summary>
        /// Construct the Progress object using the progress state, message and the value.
        /// </summary>
        public Progress(ProgressState state, string? message, int value)
        {
            State = state;
            Message = message;
            Value = value;
        }

        /// <summary>
        /// The progress state.
        /// </summary>
        public ProgressState State { get; }

        /// <summary>
        /// The message of the progress.
        /// </summary>
        public string? Message { get; }

        /// <summary>
        /// The Value of the progress.
        /// </summary>
        /// <remarks>
        /// The value is the max number the progress can reach if the progress state is `Start`.
        /// The value is the max number the progress reached if the progress state is `End`.
        /// The value is the incremented value in the progress if the progress state is `Increment`.
        /// </remarks>
        public int Value { get; }
    }
}