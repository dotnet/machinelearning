// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    /// <summary>
    /// The trivial wrapper for a <see cref="IDataLoader{TSource}"/> that acts as an estimator and ignores the source.
    /// </summary>
    internal sealed class TrivialLoaderEstimator<TSource, TLoader> : IDataLoaderEstimator<TSource, TLoader>
        where TLoader : IDataLoader<TSource>
    {
        public TLoader Loader { get; }

        public TrivialLoaderEstimator(TLoader loader)
        {
            Loader = loader;
        }

        public TLoader Fit(TSource input) => Loader;

        public SchemaShape GetOutputSchema() => SchemaShape.Create(Loader.GetOutputSchema());
    }
}
