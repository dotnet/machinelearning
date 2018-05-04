// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;

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

    /// <summary>
    /// Interface to provide extra information about a trainer.
    /// </summary>
    public interface ITrainerEx : ITrainer
    {
        // REVIEW: Ideally trainers should be able to communicate
        // something about the type of data they are capable of being trained
        // on, e.g., what ColumnKinds they want, how many of each, of what type,
        // etc. This interface seems like the most natural conduit for that sort
        // of extra information.

        // REVIEW: Can we please have consistent naming here?
        // 'Need' vs. 'Want' looks arbitrary to me, and it's grammatically more correct to
        // be 'Needs' / 'Wants' anyway.

        /// <summary>
        /// Whether the trainer needs to see data in normalized form.
        /// </summary>
        bool NeedNormalization { get; }

        /// <summary>
        /// Whether the trainer needs calibration to produce probabilities.
        /// </summary>
        bool NeedCalibration { get; }

        /// <summary>
        /// Whether this trainer could benefit from a cached view of the data.
        /// </summary>
        bool WantCaching { get; }
    }

    public interface ITrainerHost
    {
        Random Rand { get; }
        int Verbosity { get; }

        TextWriter StdOut { get; }
        TextWriter StdErr { get; }
    }

    // The Trainer (of Factory) can optionally implement this.
    public interface IModelCombiner<TModel, TPredictor>
        where TPredictor : IPredictor
    {
        TPredictor CombineModels(IEnumerable<TModel> models);
    }

    /// <summary>
    /// Weakly typed interface for a trainer "session" that produces a predictor.
    /// </summary>
    public interface ITrainer
    {
        /// <summary>
        /// Return the type of prediction task for the produced predictor.
        /// </summary>
        PredictionKind PredictionKind { get; }

        /// <summary>
        ///  Returns the trained predictor.
        ///  REVIEW: Consider removing this.
        /// </summary>
        IPredictor CreatePredictor();
    }

    /// <summary>
    /// Interface implemented by the MetalinearLearners base class.
    /// Used to distinguish the MetaLinear Learners from the other learners
    /// </summary>
    public interface IMetaLinearTrainer
    {

    }

    public interface ITrainer<in TDataSet> : ITrainer
    {
        /// <summary>
        /// Trains a predictor using the specified dataset.
        /// </summary>
        /// <param name="data"> Training dataset </param>
        void Train(TDataSet data);
    }

    /// <summary>
    /// Strongly typed generic interface for a trainer. A trainer object takes
    /// supervision data and produces a predictor.
    /// </summary>
    /// <typeparam name="TDataSet"> Type of the training dataset</typeparam>
    /// <typeparam name="TPredictor"> Type of predictor produced</typeparam>
    public interface ITrainer<in TDataSet, out TPredictor> : ITrainer<TDataSet>
        where TPredictor : IPredictor
    {
        /// <summary>
        ///  Returns the trained predictor.
        /// </summary>
        /// <returns>Trained predictor ready to make predictions</returns>
        new TPredictor CreatePredictor();
    }

    /// <summary>
    /// Trainers that want data to do their own validation implement this interface.
    /// </summary>
    public interface IValidatingTrainer<in TDataSet> : ITrainer<TDataSet>
    {
        /// <summary>
        /// Trains a predictor using the specified dataset.
        /// </summary>
        /// <param name="data">Training dataset</param>
        /// <param name="validData">Validation dataset</param>
        void Train(TDataSet data, TDataSet validData);
    }

    public interface IIncrementalTrainer<in TDataSet, in TPredictor> : ITrainer<TDataSet>
    {
        /// <summary>
        /// Trains a predictor using the specified dataset and a trained predictor.
        /// </summary>
        /// <param name="data">Training dataset</param>
        /// <param name="predictor">A trained predictor</param>
        void Train(TDataSet data, TPredictor predictor);
    }

    public interface IIncrementalValidatingTrainer<in TDataSet, in TPredictor> : ITrainer<TDataSet>
    {
        /// <summary>
        /// Trains a predictor using the specified dataset and a trained predictor.
        /// </summary>
        /// <param name="data">Training dataset</param>
        /// <param name="validData">Validation dataset</param>
        /// <param name="predictor">A trained predictor</param>
        void Train(TDataSet data, TDataSet validData, TPredictor predictor);
    }

#if FUTURE
    public interface IMultiTrainer<in TDataSet, in TFeatures, out TResult> :
        IMultiTrainer<TDataSet, TDataSet, TFeatures, TResult>
    {
    }

    public interface IMultiTrainer<in TDataSet, in TDataBatch, in TFeatures, out TResult> :
        ITrainer<TDataSet, TFeatures, TResult>
    {
        void UpdatePredictor(TDataBatch trainInstance);
        IPredictor<TFeatures, TResult> GetCurrentPredictor();
    }
#endif
}
