// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.ObjectPool;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Primitives;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// Encapsulates the data and logic required for loading and reloading PredictionEngine object pools.
    /// </summary>
    internal class PoolLoader<TData, TPrediction> : IDisposable
        where TData : class
        where TPrediction : class, new()
    {
        private static readonly ObjectPoolProvider _poolProvider = new DefaultObjectPoolProvider();

        private ObjectPool<PredictionEngine<TData, TPrediction>> _pool;
        private ITransformer _model;
        private readonly IDisposable _changeTokenRegistration;

        private readonly ConditionalWeakTable<PredictionEngine<TData, TPrediction>, ObjectPool<PredictionEngine<TData, TPrediction>>> _rentedEngines;
        private bool _disposed;

        public PoolLoader(IServiceProvider sp, PredictionEnginePoolOptions<TData, TPrediction> poolOptions)
        {
            var contextOptions = sp.GetRequiredService<IOptions<MLOptions>>();
            Context = contextOptions.Value.MLContext ?? throw new ArgumentNullException(nameof(contextOptions));
            Loader = poolOptions.ModelLoader ?? throw new ArgumentNullException(nameof(poolOptions));

            _rentedEngines = new ConditionalWeakTable<PredictionEngine<TData, TPrediction>, ObjectPool<PredictionEngine<TData, TPrediction>>>();

            LoadPool();

            _changeTokenRegistration = ChangeToken.OnChange(
                () => Loader.GetReloadToken(),
                () => LoadPool());
        }

        public ModelLoader Loader { get; }
        private MLContext Context { get; }

        /// <summary>
        /// The active pool generation. Exposed for compatibility; prefer <see cref="Get"/> and
        /// <see cref="Return"/>, which route an engine back to the generation that created it.
        /// </summary>
        public ObjectPool<PredictionEngine<TData, TPrediction>> PredictionEnginePool { get { return Volatile.Read(ref _pool); } }

        /// <summary>
        /// Rents an engine from the current pool generation, recording its origin so it can be
        /// returned to the correct generation later.
        /// </summary>
        public PredictionEngine<TData, TPrediction> Get()
        {
            var pool = Volatile.Read(ref _pool);
            if (_disposed || pool == null)
            {
                throw new ObjectDisposedException(nameof(PoolLoader<TData, TPrediction>));
            }

            var engine = pool.Get();

            _rentedEngines.Remove(engine);
            _rentedEngines.Add(engine, pool);
            return engine;
        }

        /// <summary>
        /// Returns an engine to the generation it was rented from. If that generation has already
        /// been disposed by a hot-swap, the pool disposes the engine instead of retaining it.
        /// </summary>
        public void Return(PredictionEngine<TData, TPrediction> engine)
        {
            if (engine == null)
            {
                throw new ArgumentNullException(nameof(engine));
            }

            if (_rentedEngines.TryGetValue(engine, out var origin))
            {
                _rentedEngines.Remove(engine);
                origin.Return(engine);
            }
            else
            {
                engine.Dispose();
            }
        }

        private void LoadPool()
        {
            if (_disposed)
            {
                return;
            }

            var model = Loader.GetModel();
            var policy = new PredictionEnginePoolPolicy<TData, TPrediction>(Context, model);
            var newPool = _poolProvider.Create(policy);

            var oldPool = Interlocked.Exchange(ref _pool, newPool);
            var oldModel = Interlocked.Exchange(ref _model, model);

            DisposeGeneration(oldPool, oldModel, model);

            if (_disposed && Interlocked.CompareExchange(ref _pool, null, newPool) == newPool)
            {
                DisposeGeneration(newPool, model, null);
            }
        }

        private static void DisposeGeneration(
            ObjectPool<PredictionEngine<TData, TPrediction>> pool,
            ITransformer model,
            ITransformer survivingModel)
        {
            (pool as IDisposable)?.Dispose();

            if (model != null && !ReferenceEquals(model, survivingModel))
            {
                (model as IDisposable)?.Dispose();
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _changeTokenRegistration?.Dispose();

            var pool = Interlocked.Exchange(ref _pool, null);
            var model = Interlocked.Exchange(ref _model, null);
            DisposeGeneration(pool, model, null);

            (Loader as IDisposable)?.Dispose();
        }
    }
}
