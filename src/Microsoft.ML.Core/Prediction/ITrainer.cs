// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime
{
    // REVIEW: Would be nice if the registration under SignatureTrainer were automatic
    // given registration for one of the "sub-class" signatures.

    /// <summary>
    /// Loadable class signatures for trainers. Typically each trainer should register with
    /// both SignatureTrainer and SignatureXxxTrainer where Xxx is the prediction kind.
    /// </summary>
    public delegate void SignatureTrainer();

    public delegate void SignatureBinaryClassifierTrainer();
    public delegate void SignatureMultiClassClassifierTrainer();
    public delegate void SignatureRegressorTrainer();
    public delegate void SignatureMultiOutputRegressorTrainer();
    public delegate void SignatureRankerTrainer();
    public delegate void SignatureAnomalyDetectorTrainer();
    public delegate void SignatureClusteringTrainer();
    public delegate void SignatureSequenceTrainer();
    public delegate void SignatureMatrixRecommendingTrainer();

    public delegate void SignatureModelCombiner(PredictionKind kind);

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

    // A trainer can optionally implement this to indicate it can combine multiple models into a single predictor.
    public interface IModelCombiner<TModel, TPredictor>
        where TPredictor : IPredictor
    {
        TPredictor CombineModels(IEnumerable<TModel> models);
    }

    /// <summary>
    /// Interface implemented by the MetalinearLearners base class.
    /// Used to distinguish the MetaLinear Learners from the other learners
    /// </summary>
    public interface IMetaLinearTrainer
    {
    }
}
