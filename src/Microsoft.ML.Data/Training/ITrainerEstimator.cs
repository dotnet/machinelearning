// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// Interface for the Trainer Estimator.
    /// </summary>
    /// <typeparam name="TTransformer"></typeparam>
    /// <typeparam name="TPredictor"></typeparam>
    public interface ITrainerEstimator<out TTransformer, out TPredictor> : IEstimator<TTransformer>
        where TTransformer : ISingleFeaturePredictionTransformer<TPredictor>
        where TPredictor : class
    {
        /// <summary>
        /// Gets the <see cref="TrainerInfo"/> information about the trainer.
        /// </summary>
        TrainerInfo Info { get; }
    }
}
