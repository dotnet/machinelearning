// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// The trivial implementation of <see cref="IEstimator{TTransformer}"/> that already has
    /// the transformer and returns it on every call to <see cref="Fit(IDataView)"/>.
    ///
    /// Concrete implementations still have to provide the schema propagation mechanism, since
    /// there is no easy way to infer it from the transformer.
    /// </summary>
    public abstract class TrivialEstimator<TTransformer> : IEstimator<TTransformer>
        where TTransformer : class, ITransformer
    {
        protected readonly IHost Host;
        protected readonly TTransformer Transformer;

        protected TrivialEstimator(IHost host, TTransformer transformer)
        {
            Contracts.AssertValue(host);

            Host = host;
            Host.CheckValue(transformer, nameof(transformer));
            Transformer = transformer;
        }

        public TTransformer Fit(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            // Validate input schema.
            Transformer.GetOutputSchema(input.Schema);
            return Transformer;
        }

        public abstract SchemaShape GetOutputSchema(SchemaShape inputSchema);
    }
}
