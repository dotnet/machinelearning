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
        /// Adds a <see cref="PredictionEnginePool{TData, TPrediction}"/> and required config services to the service collection.
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
            return services.AddPredictionEnginePool<TData, TPrediction>(() =>
                services.AddSingleton<PredictionEnginePool<TData, TPrediction>>());
        }

        /// <summary>
        /// Adds a <see cref="PredictionEnginePool{TData, TPrediction}"/> and required config services to the service collection.
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
            return services.AddPredictionEnginePool<TData, TPrediction>(() =>
                services.AddSingleton(implementationFactory));
        }

        private static PredictionEnginePoolBuilder<TData, TPrediction> AddPredictionEnginePool<TData, TPrediction>(
            this IServiceCollection services,
            Action callback)
            where TData : class
            where TPrediction : class, new()
        {
            services.AddRequiredPredictionEnginePoolServices();
            callback();

            return new PredictionEnginePoolBuilder<TData, TPrediction>(services);
        }

        /// <summary>
        /// Adds only the required config services for <see cref="PredictionEnginePool{TData, TPrediction}"/> to the service collection.
        /// </summary>
        /// <param name="services">
        /// The <see cref="IServiceCollection "/> to add services to.
        /// </param>
        /// <returns>
        /// A reference to this instance after the operation has completed.
        /// </returns>
        public static IServiceCollection AddRequiredPredictionEnginePoolServices(this IServiceCollection services)
        {
            services
                .AddLogging()
                .AddOptions()
                .TryAddEnumerable(ServiceDescriptor.Singleton<IPostConfigureOptions<MLOptions>, PostMLContextOptionsConfiguration>());

            return services;
        }
    }
}
