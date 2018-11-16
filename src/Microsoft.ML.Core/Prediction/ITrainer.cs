// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime
{
    // REVIEW: Would be nice if the registration under SignatureTrainer were automatic
    // given registration for one of the "sub-class" signatures.

    /// <summary>
    /// Loadable class signatures for trainers. Typically each trainer should register with
    /// both SignatureTrainer and SignatureXxxTrainer where Xxx is the prediction kind.
    /// </summary>
    [BestFriend]
    internal delegate void SignatureTrainer();

    [BestFriend]
    internal delegate void SignatureBinaryClassifierTrainer();
    [BestFriend]
    internal delegate void SignatureMultiClassClassifierTrainer();
    [BestFriend]
    internal delegate void SignatureRegressorTrainer();
    [BestFriend]
    internal delegate void SignatureMultiOutputRegressorTrainer();
    [BestFriend]
    internal delegate void SignatureRankerTrainer();
    [BestFriend]
    internal delegate void SignatureAnomalyDetectorTrainer();
    [BestFriend]
    internal delegate void SignatureClusteringTrainer();
    [BestFriend]
    internal delegate void SignatureSequenceTrainer();
    [BestFriend]
    internal delegate void SignatureMatrixRecommendingTrainer();

    /// <summary>
    /// The base interface for a trainers. Implementors should not implement this interface directly,
    /// but rather implement the more specific <see cref="ITrainer{TPredictor}"/>.
    /// </summary>
    public interface ITrainer
    {
        /// <summary>
        /// Auxiliary information about the trainer in terms of its capabilities
        /// and requirements.
        /// </summary>
        TrainerInfo Info { get; }

        /// <summary>
        /// Return the type of prediction task for the produced predictor.
        /// </summary>
        PredictionKind PredictionKind { get; }

        /// <summary>
        ///  Trains a predictor.
        /// </summary>
        /// <param name="context">A context containing at least the training data</param>
        /// <returns>The trained predictor</returns>
        /// <seealso cref="ITrainer{TPredictor}.Train(TrainContext)"/>
        IPredictor Train(TrainContext context);
    }

    /// <summary>
    /// Strongly typed generic interface for a trainer. A trainer object takes training data
    /// and produces a predictor.
    /// </summary>
    /// <typeparam name="TPredictor"> Type of predictor produced</typeparam>
    public interface ITrainer<out TPredictor> : ITrainer
        where TPredictor : IPredictor
    {
        /// <summary>
        ///  Trains a predictor.
        /// </summary>
        /// <param name="context">A context containing at least the training data</param>
        /// <returns>The trained predictor</returns>
        new TPredictor Train(TrainContext context);
    }

    public static class TrainerExtensions
    {
        /// <summary>
        /// Convenience train extension for the case where one has only a training set with no auxiliary information.
        /// Equivalent to calling <see cref="ITrainer.Train(TrainContext)"/>
        /// on a <see cref="TrainContext"/> constructed with <paramref name="trainData"/>.
        /// </summary>
        /// <param name="trainer">The trainer</param>
        /// <param name="trainData">The training data.</param>
        /// <returns>The trained predictor</returns>
        public static IPredictor Train(this ITrainer trainer, RoleMappedData trainData)
            => trainer.Train(new TrainContext(trainData));

        /// <summary>
        /// Convenience train extension for the case where one has only a training set with no auxiliary information.
        /// Equivalent to calling <see cref="ITrainer{TPredictor}.Train(TrainContext)"/>
        /// on a <see cref="TrainContext"/> constructed with <paramref name="trainData"/>.
        /// </summary>
        /// <param name="trainer">The trainer</param>
        /// <param name="trainData">The training data.</param>
        /// <returns>The trained predictor</returns>
        public static TPredictor Train<TPredictor>(this ITrainer<TPredictor> trainer, RoleMappedData trainData) where TPredictor : IPredictor
            => trainer.Train(new TrainContext(trainData));
    }
}
