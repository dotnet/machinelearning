// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.ObjectPool;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    public class PredictionEnginePoolPolicy<TData, TPrediction> : IPooledObjectPolicy<PredictionEngine<TData, TPrediction>>
                    where TData : class
                    where TPrediction : class, new()
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        private readonly List<WeakReference> _references;

        public PredictionEnginePoolPolicy(MLContext mlContext, ITransformer model)
        {
            _mlContext = mlContext;
            _model = model;
            _references = new List<WeakReference>();
        }

        public PredictionEngine<TData, TPrediction> Create()
        {
            var engine = _mlContext.Model.CreatePredictionEngine<TData, TPrediction>(_model);
            _references.Add(new WeakReference(engine));
            return engine;
        }

        public bool Return(PredictionEngine<TData, TPrediction> obj)
        {
            return _references.Any(x => x.Target == obj);
        }
    }
}