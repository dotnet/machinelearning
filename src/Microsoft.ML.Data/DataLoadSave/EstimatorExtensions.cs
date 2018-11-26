﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;

namespace Microsoft.ML
{
    /// <summary>
    /// Extension methods that allow chaining estimator and transformer pipes together.
    /// </summary>
    public static class LearningPipelineExtensions
    {
        /// <summary>
        /// Create a new composite reader estimator, by appending another estimator to the end of this data reader estimator.
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
        /// Create a new composite reader estimator, by appending an estimator to this data reader.
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
        /// Create a new estimator chain, by appending another estimator to the end of this estimator.
        /// </summary>
        public static EstimatorChain<TTrans> Append<TTrans>(
            this IEstimator<ITransformer> start, IEstimator<TTrans> estimator,
            TransformerScope scope = TransformerScope.Everything)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(start, nameof(start));
            Contracts.CheckValue(estimator, nameof(estimator));

            if (start is EstimatorChain<ITransformer> est)
                return est.Append(estimator, scope);

            return new EstimatorChain<ITransformer>().Append(start).Append(estimator, scope);
        }

        /// <summary>
        /// Append a 'caching checkpoint' to the estimator chain. This will ensure that the downstream estimators will be trained against
        /// cached data. It is helpful to have a caching checkpoint before trainers that take multiple data passes.
        /// </summary>
        /// <param name="start">The starting estimator</param>
        /// <param name="env">The host environment to use for caching.</param>

        public static EstimatorChain<TTrans> AppendCacheCheckpoint<TTrans>(this IEstimator<TTrans> start, IHostEnvironment env)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(start, nameof(start));
            return new EstimatorChain<ITransformer>().Append(start).AppendCacheCheckpoint(env);
        }

        /// <summary>
        /// Create a new composite reader, by appending a transformer to this data reader.
        /// </summary>
        public static CompositeDataReader<TSource, TTrans> Append<TSource, TTrans>(this IDataReader<TSource> reader, TTrans transformer)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(reader, nameof(reader));
            Contracts.CheckValue(transformer, nameof(transformer));

            return new CompositeDataReader<TSource, ITransformer>(reader).AppendTransformer(transformer);
        }

        /// <summary>
        /// Create a new transformer chain, by appending another transformer to the end of this transformer chain.
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
