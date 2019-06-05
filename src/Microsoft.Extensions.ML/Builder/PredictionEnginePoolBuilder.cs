// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// A class that provides the mechanisms to configure a pool
    /// of ML.NET <see cref="PredictionEngine{TData, TPrediction}"/> objects.
    /// </summary>
    public class PredictionEnginePoolBuilder<TData, TPrediction>
        where TData : class
        where TPrediction : class, new()
    {
        /// <summary>
        /// Initializes a new instance of <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </summary>
        /// <param name="services">The <see cref="IServiceCollection"/> to add services to.</param>
        public PredictionEnginePoolBuilder(IServiceCollection services)
        {
            Services = services ?? throw new ArgumentException(nameof(services));
        }

        /// <summary>
        /// The <see cref="IServiceCollection"/> to add services to.
        /// </summary>
        public IServiceCollection Services { get; private set; }
    }
}
