// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// Holds information relevant to trainers. Instances of this class are meant to be constructed and passed
    /// into <see cref="ITrainer{TPredictor}.Train(TrainContext)"/> or <see cref="ITrainer.Train(TrainContext)"/>.
    /// This holds at least a training set, as well as optioonally a predictor.
    /// </summary>
    public sealed class TrainContext
    {
        /// <summary>
        /// The training set. Cannot be <c>null</c>.
        /// </summary>
        public RoleMappedData TrainingSet { get; }

        /// <summary>
        /// The validation set. Can be <c>null</c>. Note that passing a non-<c>null</c> validation set into
        /// a trainer that does not support validation sets should not be considered an error condition. It
        /// should simply be ignored in that case.
        /// </summary>
        public RoleMappedData ValidationSet { get; }

        /// <summary>
        /// The initial predictor, for incremental training. Note that if a <see cref="ITrainer"/> implementor
        /// does not support incremental training, then it can ignore it similarly to how one would ignore
        /// <see cref="ValidationSet"/>. However, if the trainer does support incremental training and there
        /// is something wrong with a non-<c>null</c> value of this, then the trainer ought to throw an exception.
        /// </summary>
        public IPredictor InitialPredictor { get; }


        /// <summary>
        /// Constructor, given a training set and optional other arguments.
        /// </summary>
        /// <param name="train">Will set <see cref="TrainingSet"/> to this value. This must be specified</param>
        /// <param name="valid">Will set <see cref="ValidationSet"/> to this value if specified</param>
        /// <param name="initPredictor">Will set <see cref="InitialPredictor"/> to this value if specified</param>
        public TrainContext(RoleMappedData train, RoleMappedData valid = null, IPredictor initPredictor = null)
        {
            Contracts.CheckValue(train, nameof(train));
            Contracts.CheckValueOrNull(valid);
            Contracts.CheckValueOrNull(initPredictor);

            // REVIEW: Should there be code here to ensure that the role mappings between the two are compatible?
            // That is, all the role mappings are the same and the columns between them have identical types?

            TrainingSet = train;
            ValidationSet = valid;
            InitialPredictor = initPredictor;
        }
    }
}
