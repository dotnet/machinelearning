// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Extension methods that allow chaining estimator and transformer pipes together.
    /// </summary>
    public static class LearningPipelineExtensions
    {
        public static CompositeReaderEstimator<TSource, TTrans> Append<TSource, TTrans>(
            this IDataReaderEstimator<TSource, IDataReader<TSource>> start, IEstimator<TTrans> estimator)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(start, nameof(start));
            Contracts.CheckValue(estimator, nameof(estimator));

            return new CompositeReaderEstimator<TSource, ITransformer>(start).Append(estimator);
        }

        public static EstimatorChain<TTrans> Append<TTrans>(
            this IEstimator<ITransformer> start, IEstimator<TTrans> estimator,
            TransformerScope scope = TransformerScope.Everything)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(start, nameof(start));
            Contracts.CheckValue(estimator, nameof(estimator));

            return new EstimatorChain<ITransformer>().Append(start).Append(estimator, scope);
        }

        public static CompositeDataReader<TSource, TTrans> Append<TSource, TTrans>(this IDataReader<TSource> reader, TTrans transformer)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(reader, nameof(reader));
            Contracts.CheckValue(transformer, nameof(transformer));

            return new CompositeDataReader<TSource, ITransformer>(reader).AppendTransformer(transformer);
        }

        public static TransformerChain<TTrans> Append<TTrans>(this ITransformer start, TTrans transformer)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(start, nameof(start));
            Contracts.CheckValue(transformer, nameof(transformer));

            return new TransformerChain<TTrans>(start, transformer);
        }
    }
}
