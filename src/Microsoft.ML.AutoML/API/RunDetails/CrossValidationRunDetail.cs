// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Details about a cross validation run in an AutoML experiment.
    /// </summary>
    /// <typeparam name="TMetrics">Metrics type for the run.</typeparam>
    /// <remarks>
    /// Over the course of an experiment, many models are evaluated on a dataset
    /// using cross validation. This object contains information about each model
    /// evaluated during the AutoML experiment.
    /// </remarks>
    public sealed class CrossValidationRunDetail<TMetrics> : RunDetail
    {
        /// <summary>
        /// Results for each of the cross validation folds.
        /// </summary>
        public IEnumerable<TrainResult<TMetrics>> Results { get; private set; }

        internal CrossValidationRunDetail(string trainerName,
            IEstimator<ITransformer> estimator,
            Pipeline pipeline,
            IEnumerable<TrainResult<TMetrics>> results) : base(trainerName, estimator, pipeline)
        {
            Results = results;
        }
    }

    /// <summary>
    /// Result of a pipeline trained on a cross validation fold.
    /// </summary>
    /// <typeparam name="TMetrics">Metrics type for the run.</typeparam>
    public sealed class TrainResult<TMetrics>
    {
        /// <summary>
        /// Each fold has training data and validation data. A model trained on the
        /// folds's training data is evaluated against the validation data,
        /// and the metrics for that calculation are emitted here.
        /// </summary>
        public TMetrics ValidationMetrics { get; private set; }

        /// <summary>
        /// Model trained on the fold during the run.
        /// </summary>
        /// <remarks>
        /// You can use the trained model to obtain predictions on input data.
        /// </remarks>
        public ITransformer Model { get { return _modelContainer.GetModel(); } }

        /// <summary>
        /// Exception encountered while training the fold. This property is
        /// <see langword="null"/> if no exception was encountered.
        /// </summary>
        /// <remarks>
        /// If an exception occurred, it's possible some properties in this object
        /// (like <see cref="Model"/>) could be <see langword="null"/>.
        /// </remarks>
        public Exception Exception { get; private set; }

        private readonly ModelContainer _modelContainer;

        internal TrainResult(ModelContainer modelContainer,
            TMetrics metrics,
            Exception exception)
        {
            _modelContainer = modelContainer;
            ValidationMetrics = metrics;
            Exception = exception;
        }
    }

}
