// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// Instances of this class posses information about trainers, in terms of their requirements and capabilities.
    /// The intended usage is as the value for <see cref="ITrainer.Info"/>.
    /// </summary>
    public sealed class TrainerInfo
    {
        // REVIEW: Ideally trainers should be able to communicate
        // something about the type of data they are capable of being trained
        // on, for example, what ColumnKinds they want, how many of each, of what type,
        // etc. This interface seems like the most natural conduit for that sort
        // of extra information.

        /// <summary>
        /// Whether the trainer needs to see data in normalized form. Only non-parametric learners will tend to produce
        /// normalization here.
        /// </summary>
        public bool NeedNormalization { get; }

        /// <summary>
        /// Whether the trainer needs calibration to produce probabilities. As a general rule only trainers that produce
        /// binary classifier predictors that also do not have a natural probabilistic interpretation should have a
        /// <c>true</c> value here.
        /// </summary>
        public bool NeedCalibration { get; }

        /// <summary>
        /// Whether this trainer could benefit from a cached view of the data. Trainers that have few passes over the
        /// data, or that need to build their own custom data structure over the data, will have a <c>false</c> here.
        /// </summary>
        public bool WantCaching { get; }

        /// <summary>
        /// Whether the trainer supports validation set via <see cref="TrainContext.ValidationSet"/>. Not implementing
        /// this interface and returning <c>false</c> from this property is an indication the trainer does not support
        /// that.
        /// </summary>
        public bool SupportsValidation { get; }

        /// <summary>
        /// Whether the trainer can use test set via <see cref="TrainContext.TestSet"/>. Not implementing
        /// this interface and returning <c>false</c> from this property is an indication the trainer does not support
        /// that.
        /// </summary>
        public bool SupportsTest { get; }

        /// <summary>
        /// Whether the trainer can support incremental trainers via <see cref="TrainContext.InitialPredictor"/>. Not
        /// implementing this interface and returning <c>true</c> from this property is an indication the trainer does
        /// not support that.
        /// </summary>
        public bool SupportsIncrementalTraining { get; }

        /// <summary>
        /// Initializes with the given parameters. The parameters have default values for the most typical values
        /// for most classical trainers.
        /// </summary>
        /// <param name="normalization">The value for the property <see cref="NeedNormalization"/></param>
        /// <param name="calibration">The value for the property <see cref="NeedCalibration"/></param>
        /// <param name="caching">The value for the property <see cref="WantCaching"/></param>
        /// <param name="supportValid">The value for the property <see cref="SupportsValidation"/></param>
        /// <param name="supportIncrementalTrain">The value for the property <see cref="SupportsIncrementalTraining"/></param>
        /// <param name="supportTest">The value for the property <see cref="SupportsTest"/></param>
        public TrainerInfo(bool normalization = true, bool calibration = false, bool caching = true,
            bool supportValid = false, bool supportIncrementalTrain = false, bool supportTest = false)
        {
            NeedNormalization = normalization;
            NeedCalibration = calibration;
            WantCaching = caching;
            SupportsValidation = supportValid;
            SupportsIncrementalTraining = supportIncrementalTrain;
            SupportsTest = supportTest;
        }
    }
}
