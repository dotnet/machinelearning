// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// Instances of this class are meant to be constructed and passed to trainers.
    /// </summary>
    public sealed class TrainContext
    {
        /// <summary>
        /// The training set. Cannot be <c>null</c>.
        /// </summary>
        public RoleMappedData Train { get; }

        /// <summary>
        /// The validation set. Can be <c>null</c>.
        /// </summary>
        public RoleMappedData Validation { get; }

        /// <summary>
        /// The initial 
        /// </summary>
        public IPredictor InitialPredictor { get; }


        /// <summary>
        /// Constructor, given a training set and optional other arguments.
        /// </summary>
        /// <param name="train">Will be set to <see cref="Train"/>, must be specified</param>
        /// <param name="valid">Will be set to <see cref="Validation"/> if specified</param>
        /// <param name="initPredictor">Will be set to <see cref="InitialPredictor"/> if specified</param>
        public TrainContext(RoleMappedData train, RoleMappedData valid = null, IPredictor initPredictor = null)
        {
            Contracts.CheckValue(train, nameof(train));
            Contracts.CheckValueOrNull(valid);
            Contracts.CheckValueOrNull(initPredictor);

            // REVIEW: Should there be code here to ensure that the role mappings between the two are compatible?
            // That is, all the role mappings are the same and the columns between them have identical types?

            Train = train;
            Validation = valid;
            InitialPredictor = initPredictor;
        }
    }
}
