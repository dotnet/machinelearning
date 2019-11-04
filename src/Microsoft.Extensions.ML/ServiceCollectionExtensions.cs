// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
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
        /// <param name="implementationFactory">
        /// The factory that creates the <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </param>
        /// <returns>
        /// The <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/> that was added to the collection.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> AddPredictionEnginePool<TData, TPrediction>(
            this IServiceCollection services,
            Func<IServiceProvider, PredictionEnginePool<TData, TPrediction>> implementationFactory)
            where TData : class
            where TPrediction : class, new()
        {
            services
                .AddPrerequisiteServices()
                .AddSingleton<PredictionEnginePool<TData, TPrediction>, PredictionEnginePool<TData, TPrediction>>(implementationFactory);
            return new PredictionEnginePoolBuilder<TData, TPrediction>(services);
        }

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
            services
                .AddPrerequisiteServices()
                .AddSingleton<PredictionEnginePool<TData, TPrediction>, PredictionEnginePool<TData, TPrediction>>();
            return new PredictionEnginePoolBuilder<TData, TPrediction>(services);
        }

        public static IServiceCollection AddPrerequisiteServices(this IServiceCollection services)
        {
            services
                .AddLogging()
                .AddOptions()
                .TryAddEnumerable(ServiceDescriptor.Singleton<IPostConfigureOptions<MLOptions>, PostMLContextOptionsConfiguration>());
            return services;
        }
    }
}
