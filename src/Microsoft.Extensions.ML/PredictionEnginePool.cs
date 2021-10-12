// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.Extensions.Options;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// Provides a pool of <see cref="PredictionEngine{TSrc, TDst}"/> objects
    /// that can be used to make predictions.
    /// </summary>
    public class PredictionEnginePool<TData, TPrediction>
        where TData : class
        where TPrediction : class, new()
    {
        private readonly MLOptions _mlContextOptions;
        private readonly IOptionsFactory<PredictionEnginePoolOptions<TData, TPrediction>> _predictionEngineOptions;
        private readonly IServiceProvider _serviceProvider;
        private readonly PoolLoader<TData, TPrediction> _defaultEnginePool;
        private readonly ConcurrentDictionary<string, PoolLoader<TData, TPrediction>> _namedPools;

        public PredictionEnginePool(IServiceProvider serviceProvider,
                                    IOptions<MLOptions> mlContextOptions,
                                    IOptionsFactory<PredictionEnginePoolOptions<TData, TPrediction>> predictionEngineOptions)
        {
            _mlContextOptions = mlContextOptions.Value;
            _predictionEngineOptions = predictionEngineOptions;
            _serviceProvider = serviceProvider;

            var defaultOptions = _predictionEngineOptions.Create(string.Empty);

            if (defaultOptions.ModelLoader != null)
            {
                _defaultEnginePool = new PoolLoader<TData, TPrediction>(_serviceProvider, defaultOptions);
            }

            _namedPools = new ConcurrentDictionary<string, PoolLoader<TData, TPrediction>>();
        }

        /// <summary>
        /// Get the Model used to create the pooled PredictionEngine.
        /// </summary>
        /// <param name="modelName">
        /// The name of the model. Used when there are multiple models with the same input/output.
        /// </param>
        public ITransformer GetModel(string modelName)
        {
            if (!_namedPools.ContainsKey(modelName))
            {
                AddPool(modelName);
            }

            return _namedPools[modelName].Loader.GetModel();
        }

        /// <summary>
        /// Get the Model used to create the pooled PredictionEngine.
        /// </summary>
        public ITransformer GetModel()
        {
            return _defaultEnginePool.Loader.GetModel();
        }

        /// <summary>
        /// Gets a PredictionEngine that can be used to make predictions using
        /// <typeparamref name="TData"/> and <typeparamref name="TPrediction"/>.
        /// </summary>
        public PredictionEngine<TData, TPrediction> GetPredictionEngine()
        {
            return GetPredictionEngine(string.Empty);
        }

        /// <summary>
        /// Gets a PredictionEngine for a named model.
        /// </summary>
        /// <param name="modelName">
        /// The name of the model which allows for uniquely identifying the model when
        /// multiple models have the same <typeparamref name="TData"/> and
        /// <typeparamref name="TPrediction"/> types.
        /// </param>
        public PredictionEngine<TData, TPrediction> GetPredictionEngine(string modelName)
        {
            if (_namedPools.TryGetValue(modelName, out var existingPool))
            {
                return existingPool.PredictionEnginePool.Get();
            }

            //This is the case where someone has used string.Empty to get the default model.
            //We can throw all the time, but it seems reasonable that we would just do what
            //they are expecting if they know that an empty string means default.
            if (string.IsNullOrEmpty(modelName))
            {
                if (_defaultEnginePool == null)
                {
                    throw new ArgumentException("You need to configure a default, not named, model before you use this method.");
                }

                return _defaultEnginePool.PredictionEnginePool.Get();
            }

            var pool = AddPool(modelName);
            return pool.PredictionEnginePool.Get();
        }

        private PoolLoader<TData, TPrediction> AddPool(string modelName)
        {
            //Here we are in the world of named models where the model hasn't been built yet.
            var options = _predictionEngineOptions.Create(modelName);
            var pool = new PoolLoader<TData, TPrediction>(_serviceProvider, options);
            pool = _namedPools.GetOrAdd(modelName, pool);
            return pool;
        }

        /// <summary>
        /// Returns a rented PredictionEngine to the pool.
        /// </summary>
        /// <param name="engine">The rented PredictionEngine.</param>
        public void ReturnPredictionEngine(PredictionEngine<TData, TPrediction> engine)
        {
            ReturnPredictionEngine(string.Empty, engine);
        }

        /// <summary>
        /// Returns a rented PredictionEngine to the pool.
        /// </summary>
        /// <param name="modelName">
        /// The name of the model which allows for uniquely identifying the model when
        /// multiple models have the same <typeparamref name="TData"/> and
        /// <typeparamref name="TPrediction"/> types.
        /// </param>
        /// <param name="engine">The rented PredictionEngine.</param>
        public void ReturnPredictionEngine(string modelName, PredictionEngine<TData, TPrediction> engine)
        {
            if (engine == null)
            {
                throw new ArgumentNullException(nameof(engine));
            }

            if (string.IsNullOrEmpty(modelName))
            {
                _defaultEnginePool.PredictionEnginePool.Return(engine);
            }
            else
            {
                _namedPools[modelName].PredictionEnginePool.Return(engine);
            }
        }
    }
}
