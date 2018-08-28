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
        /// <summary>
        /// Create a composite reader estimator by appending an estimator to a reader estimator.
        /// </summary>
        public static CompositeReaderEstimator<TSource, TTrans> Append<TSource, TTrans>(
            this IDataReaderEstimator<TSource, IDataReader<TSource>> start, IEstimator<TTrans> estimator)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(start, nameof(start));
            Contracts.CheckValue(estimator, nameof(estimator));

            return new CompositeReaderEstimator<TSource, ITransformer>(start).Append(estimator);
        }

        /// <summary>
        /// Create a composite reader estimator by appending an estimator to a reader.
        /// </summary>
        public static CompositeReaderEstimator<TSource, TTrans> Append<TSource, TTrans>(
            this IDataReader<TSource> start, IEstimator<TTrans> estimator)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(start, nameof(start));
            Contracts.CheckValue(estimator, nameof(estimator));

            return new TrivialReaderEstimator<TSource, IDataReader<TSource>>(start).Append(estimator);
        }

        /// <summary>
        /// Create an estimator chain by appending an estimator to an estimator.
        /// </summary>
        public static EstimatorChain<TTrans> Append<TTrans>(
            this IEstimator<ITransformer> start, IEstimator<TTrans> estimator,
            TransformerScope scope = TransformerScope.Everything)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(start, nameof(start));
            Contracts.CheckValue(estimator, nameof(estimator));

            return new EstimatorChain<ITransformer>().Append(start).Append(estimator, scope);
        }

        /// <summary>
        /// Create a composite reader by appending a transformer to a data reader.
        /// </summary>
        public static CompositeDataReader<TSource, TTrans> Append<TSource, TTrans>(this IDataReader<TSource> reader, TTrans transformer)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(reader, nameof(reader));
            Contracts.CheckValue(transformer, nameof(transformer));

            return new CompositeDataReader<TSource, ITransformer>(reader).AppendTransformer(transformer);
        }

        /// <summary>
        /// Create a transformer chain by appending a transformer to a transformer.
        /// </summary>
        public static TransformerChain<TTrans> Append<TTrans>(this ITransformer start, TTrans transformer)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(start, nameof(start));
            Contracts.CheckValue(transformer, nameof(transformer));

            return new TransformerChain<TTrans>(start, transformer);
        }
    }
}
