// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Training;

namespace Microsoft.ML.Core.Prediction
{
    /// <summary>
    /// Holds information relevant to trainers. It is passed to the constructor of the<see cref="ITrainerEstimator{IPredictionTransformer, IPredictor}"/>
    /// holding additional data needed to fit the estimator. The additional data can be a validation set or an initial model.
    /// This holds at least a training set, as well as optioonally a predictor.
    /// </summary>
    public class TrainerEstimatorContext
    {
        /// <summary>
        /// The validation set. Can be <c>null</c>. Note that passing a non-<c>null</c> validation set into
        /// a trainer that does not support validation sets should not be considered an error condition. It
        /// should simply be ignored in that case.
        /// </summary>
        public IDataView ValidationSet { get; }

        /// <summary>
        /// The initial predictor, for incremental training. Note that if a <see cref="ITrainerEstimator{IPredictionTransformer, IPredictor}"/> implementor
        /// does not support incremental training, then it can ignore it similarly to how one would ignore
        /// <see cref="ValidationSet"/>. However, if the trainer does support incremental training and there
        /// is something wrong with a non-<c>null</c> value of this, then the trainer ought to throw an exception.
        /// </summary>
        public IPredictor InitialPredictor { get; }

        /// <summary>
        /// Initializes a new instance of <see cref="TrainerEstimatorContext"/>, given a training set and optional other arguments.
        /// </summary>
        /// <param name="validationSet">Will set <see cref="ValidationSet"/> to this value if specified</param>
        /// <param name="initialPredictor">Will set <see cref="InitialPredictor"/> to this value if specified</param>
        public TrainerEstimatorContext(IDataView validationSet = null, IPredictor initialPredictor = null)
        {
            Contracts.CheckValueOrNull(validationSet);
            Contracts.CheckValueOrNull(initialPredictor);

            ValidationSet = validationSet;
            InitialPredictor = initialPredictor;
        }
    }
}
