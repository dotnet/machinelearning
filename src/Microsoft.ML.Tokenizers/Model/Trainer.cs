// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// A `Trainer` has the responsibility to train a model. We feed it with lines/sentences
    /// and then it can train the given `Model`.
    /// </summary>
    public abstract class Trainer
    {
        /// <summary>
        /// Set when need to report the progress during the training.
        /// </summary>
        public ReportProgress? Progress { get; set; }

        /// <summary>
        /// Perform the actual training and update the input model with the new vocabularies and merges data.
        /// </summary>
        /// <param name="model">The model to train.</param>
        /// <returns>Special tokens to be added directly to the tokenizer along with the model.</returns>
        public abstract IReadOnlyList<AddedToken>? Train(Model model);

        /// <summary>
        /// Process the input sequences and feed the result to the model.
        /// </summary>
        /// <param name="sequences">The list of sequences to feed the trainer.</param>
        /// <param name="process">Optional process callback for reporting the training progress update.</param>
        public abstract void Feed(IEnumerable<string> sequences, Func<string, IEnumerable<string>> process);
    }
}
