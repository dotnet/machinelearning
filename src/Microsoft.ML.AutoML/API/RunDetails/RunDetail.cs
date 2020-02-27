// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Details about an AutoML experiment run.
    /// </summary>
    /// <remarks>
    /// Over the course of an experiment, many models are evaluated on a dataset.
    /// This object contains information about each model evaluated during
    /// the AutoML experiment.
    /// </remarks>
    /// <typeparam name="TMetrics">Metrics type for the experiment (like <see cref="BinaryClassificationMetrics"/>).</typeparam>
    public sealed class RunDetail<TMetrics> : RunDetail
    {
        /// <summary>
        /// Metrics of how the trained model performed on the validation data during
        /// the run.
        /// </summary>
        /// <remarks>
        /// Internally, each run has train data and validation data. Model trained on the
        /// run's training is evaluated against the validation data,
        /// and the metrics for that calculation are emitted here.
        /// </remarks>
        public TMetrics ValidationMetrics { get; private set; }

        /// <summary>
        /// Model trained during the run.
        /// </summary>
        /// <remarks>
        /// You can use the trained model to obtain predictions on input data.
        /// </remarks>
        public ITransformer Model { get { return _modelContainer?.GetModel(); } }

        /// <summary>
        /// Exception encountered during the run. This property is <see langword="null"/> if
        /// no exception was encountered.
        /// </summary>
        /// <remarks>
        /// If an exception occurred, it's possible some properties in this object
        /// (like <see cref="Model"/>) could be <see langword="null"/>.
        /// </remarks>
        public Exception Exception { get; private set; }

        private readonly ModelContainer _modelContainer;

        internal RunDetail(string trainerName,
            IEstimator<ITransformer> estimator,
            Pipeline pipeline,
            ModelContainer modelContainer,
            TMetrics metrics,
            Exception exception) : base(trainerName, estimator, pipeline)
        {
            _modelContainer = modelContainer;
            ValidationMetrics = metrics;
            Exception = exception;
        }
    }

    /// <summary>
    /// Details about an AutoML experiment run.
    /// </summary>
    /// <remarks>
    /// In trying to produce the best model, an AutoML experiment evaluates the quality of many models
    /// on a dataset. This object contains information about each model tried during the AutoML experiment.
    /// </remarks>
    public abstract class RunDetail
    {
        /// <summary>
        /// String name of the trainer used in this run. (For instance, <c>"LightGbm"</c>.)
        /// </summary>
        public string TrainerName { get; private set; }

        /// <summary>
        /// Runtime in seconds.
        /// </summary>
        /// <remarks>
        /// Runtime includes model training time. Depending on the size of the data,
        /// the runtime may be quite long.
        /// </remarks>
        public double RuntimeInSeconds { get; internal set; }

        /// <summary>
        /// An ML.NET <see cref="IEstimator{TTransformer}"/> that represents the pipeline in this run.
        /// </summary>
        /// <remarks>
        /// You can call <see cref="IEstimator{TTransformer}.Fit(IDataView)" /> on
        /// this estimator to re-train your pipeline on any <see cref="IEstimator{TTransformer}" />.
        /// </remarks>
        public IEstimator<ITransformer> Estimator { get; private set; }

        internal Pipeline Pipeline { get; private set; }
        internal double PipelineInferenceTimeInSeconds { get; set; }

        internal RunDetail(string trainerName,
            IEstimator<ITransformer> estimator,
            Pipeline pipeline)
        {
            TrainerName = trainerName;
            Estimator = estimator;
            Pipeline = pipeline;
        }
    }
}
