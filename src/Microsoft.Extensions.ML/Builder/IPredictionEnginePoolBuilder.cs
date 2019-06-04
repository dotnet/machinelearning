// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Extensions.DependencyInjection;

namespace Microsoft.Extensions.ML
{
    public interface IPredictionEnginePoolBuilder<TData, TPrediction> where TData : class where TPrediction : class, new()
    {
        IServiceCollection Services { get; }

        string ModelName { get; }
    }

    public class DefaultPredictionEnginePoolBuilder<TData, TPrediction> : IPredictionEnginePoolBuilder<TData, TPrediction> where TData : class where TPrediction : class, new()
    {
        public DefaultPredictionEnginePoolBuilder(IServiceCollection services, string modelName)
        {
            Services = services ?? throw new ArgumentException(nameof(services));
            ModelName = modelName;
        }
        public IServiceCollection Services { get; private set; }

        public string ModelName { get; private set; }
    }
}
