// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
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
        private DefaultObjectPool<PredictionEngine<TData, TPrediction>> _pool;
        private readonly IDisposable _changeTokenRegistration;

        public PoolLoader(IServiceProvider sp, PredictionEnginePoolOptions<TData, TPrediction> poolOptions)
        {
            var contextOptions = sp.GetRequiredService<IOptions<MLOptions>>();
            Context = contextOptions.Value.MLContext ?? throw new ArgumentNullException(nameof(contextOptions));
            Loader = poolOptions.ModelLoader ?? throw new ArgumentNullException(nameof(poolOptions));

            LoadPool();

            _changeTokenRegistration = ChangeToken.OnChange(
                () => Loader.GetReloadToken(),
                () => LoadPool());
        }

        public ModelLoader Loader { get; }
        private MLContext Context { get; }
        public ObjectPool<PredictionEngine<TData, TPrediction>> PredictionEnginePool { get { return _pool; } }

        private void LoadPool()
        {
            var predictionEnginePolicy = new PredictionEnginePoolPolicy<TData, TPrediction>(Context, Loader.GetModel());
            Interlocked.Exchange(ref _pool, new DefaultObjectPool<PredictionEngine<TData, TPrediction>>(predictionEnginePolicy));
        }

        public void Dispose()
        {
            _changeTokenRegistration?.Dispose();
        }
    }
}
