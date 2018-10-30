// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

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
        /// Create an estimator chain by mappering same column names only.
        /// </summary>
        public static EstimatorChain<TTrans> AppendNull<TTrans>(
            this IEstimator<ITransformer> start, IEstimator<TTrans> estimator,
            TransformerScope scope = TransformerScope.Everything)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(start, nameof(start));
            Contracts.Assert(estimator == null);

            return new EstimatorChain<ITransformer>().Append(start).Append<TTrans>();
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

        private sealed class DelegateEstimator<TTransformer> : IEstimator<TTransformer>
            where TTransformer : class, ITransformer
        {
            private readonly IEstimator<TTransformer> _est;
            private readonly Action<TTransformer> _onFit;

            public DelegateEstimator(IEstimator<TTransformer> estimator, Action<TTransformer> onFit)
            {
                Contracts.AssertValue(estimator);
                Contracts.AssertValue(onFit);
                _est = estimator;
                _onFit = onFit;
            }

            public TTransformer Fit(IDataView input)
            {
                var trans = _est.Fit(input);
                _onFit(trans);
                return trans;
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
                => _est.GetOutputSchema(inputSchema);
        }

        /// <summary>
        /// Given an estimator, return a wrapping object that will call a delegate once <see cref="IEstimator{TTransformer}.Fit(IDataView)"/>
        /// is called. It is often important for an estimator to return information about what was fit, which is why the
        /// <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> method returns a specifically typed object, rather than just a general
        /// <see cref="ITransformer"/>. However, at the same time, <see cref="IEstimator{TTransformer}"/> are often formed into pipelines
        /// with many objects, so we may need to build a chain of estimators via <see cref="EstimatorChain{TLastTransformer}"/> where the
        /// estimator for which we want to get the transformer is buried somewhere in this chain. For that scenario, we can through this
        /// method attach a delegate that will be called once fit is called.
        /// </summary>
        /// <typeparam name="TTransformer">The type of <see cref="ITransformer"/> returned by <paramref name="estimator"/></typeparam>
        /// <param name="estimator">The estimator to wrap</param>
        /// <param name="onFit">The delegate that is called with the resulting <typeparamref name="TTransformer"/> instances once
        /// <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> is called. Because <see cref="IEstimator{TTransformer}.Fit(IDataView)"/>
        /// may be called multiple times, this delegate may also be called multiple times.</param>
        /// <returns>A wrapping estimator that calls the indicated delegate whenever fit is called</returns>
        public static IEstimator<TTransformer> WithOnFitDelegate<TTransformer>(this IEstimator<TTransformer> estimator, Action<TTransformer> onFit)
            where TTransformer : class, ITransformer
        {
            Contracts.CheckValue(estimator, nameof(estimator));
            Contracts.CheckValue(onFit, nameof(onFit));
            return new DelegateEstimator<TTransformer>(estimator, onFit);
        }

        internal static T[] AppendElement<T>(this T[] array, T element)
        {
            T[] result = new T[Utils.Size(array) + 1];
            Array.Copy(array, result, result.Length - 1);
            result[result.Length - 1] = element;
            return result;
        }
    }
}
