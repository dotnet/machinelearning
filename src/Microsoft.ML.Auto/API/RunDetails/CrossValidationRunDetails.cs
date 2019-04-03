// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Auto
{
    public sealed class CrossValidationRunDetails<TMetrics> : RunDetails
    {
        public IEnumerable<TrainResult<TMetrics>> Results { get; private set; }

        internal CrossValidationRunDetails(string trainerName,
            IEstimator<ITransformer> estimator,
            Pipeline pipeline,
            IEnumerable<TrainResult<TMetrics>> results) : base(trainerName, estimator, pipeline)
        {
            Results = results;
        }
    }

    public sealed class TrainResult<TMetrics>
    {
        public TMetrics ValidationMetrics { get; private set; }
        public ITransformer Model { get { return _modelContainer.GetModel(); } }
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
