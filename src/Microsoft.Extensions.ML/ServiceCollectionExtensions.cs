// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// Extension methods for <see cref="IServiceCollection"/>.
    /// </summary>
    public static class ServiceCollectionExtensions
    {
        /// <summary>
        /// Adds a <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/> to the service collection.
        /// </summary>
        /// <param name="services">
        /// The <see cref="IServiceCollection "/> to add services to.
        /// </param>
        /// <returns>
        /// The <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/> that was added to the collection.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> AddPredictionEnginePool<TData, TPrediction>(
            this IServiceCollection services)
            where TData : class
            where TPrediction : class, new()
        {
            services.AddPredictionEngineServices<TData, TPrediction>();
            return new PredictionEnginePoolBuilder<TData, TPrediction>(services);
        }

        internal static IServiceCollection AddPredictionEngineServices<TData, TPrediction>(
            this IServiceCollection services)
            where TData : class
            where TPrediction : class, new()
        {
            services.AddLogging();
            services.AddOptions();
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IPostConfigureOptions<MLOptions>, PostMLContextOptionsConfiguration>());
            services.AddSingleton<PredictionEnginePool<TData, TPrediction>, PredictionEnginePool<TData, TPrediction>>();
            return services;
        }
    }
}
