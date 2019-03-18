// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
namespace Microsoft.ML.StaticPipe
{
    public static class StaticPipeExtensions
    {
        /// <summary>
        /// Asserts that a given data view has the indicated schema. If this method returns without
        /// throwing then the view has been validated to have columns with the indicated names and types.
        /// </summary>
        /// <typeparam name="T">The type representing the view's schema shape</typeparam>
        /// <param name="view">The view to assert the static schema on</param>
        /// <param name="env">The host environment to keep in the statically typed variant</param>
        /// <param name="outputDecl">The delegate through which we declare the schema, which ought to
        /// use the input <see cref="SchemaAssertionContext"/> to declare a <see cref="ValueTuple"/>
        /// of the <see cref="PipelineColumn"/> indices, properly named</param>
        /// <returns>A statically typed wrapping of the input view</returns>
        public static DataView<T> AssertStatic<[IsShape] T>(this IDataView view, IHostEnvironment env,
            Func<SchemaAssertionContext, T> outputDecl)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(view, nameof(view));
            env.CheckValue(outputDecl, nameof(outputDecl));

            // We don't actually need to call the method, it's just there to give the declaration.
#if DEBUG
            outputDecl(SchemaAssertionContext.Inst);
#endif

            var schema = StaticSchemaShape.Make<T>(outputDecl.Method.ReturnParameter);
            return new DataView<T>(env, view, schema);
        }

        public static DataLoader<TIn, T> AssertStatic<TIn, [IsShape] T>(this IDataLoader<TIn> loader, IHostEnvironment env,
            Func<SchemaAssertionContext, T> outputDecl)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(loader, nameof(loader));
            env.CheckValue(outputDecl, nameof(outputDecl));

            var schema = StaticSchemaShape.Make<T>(outputDecl.Method.ReturnParameter);
            return new DataLoader<TIn, T>(env, loader, schema);
        }

        public static DataLoaderEstimator<TIn, T, TLoader> AssertStatic<TIn, [IsShape] T, TLoader>(
            this IDataLoaderEstimator<TIn, TLoader> loaderEstimator, IHostEnvironment env,
            Func<SchemaAssertionContext, T> outputDecl)
            where TLoader : class, IDataLoader<TIn>
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(loaderEstimator, nameof(loaderEstimator));
            env.CheckValue(outputDecl, nameof(outputDecl));

            var schema = StaticSchemaShape.Make<T>(outputDecl.Method.ReturnParameter);
            return new DataLoaderEstimator<TIn, T, TLoader>(env, loaderEstimator, schema);
        }

        public static Transformer<TIn, TOut, TTrans> AssertStatic<[IsShape] TIn, [IsShape] TOut, TTrans>(
            this TTrans transformer, IHostEnvironment env,
            Func<SchemaAssertionContext, TIn> inputDecl,
            Func<SchemaAssertionContext, TOut> outputDecl)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformer, nameof(transformer));
            env.CheckValue(inputDecl, nameof(inputDecl));
            env.CheckValue(outputDecl, nameof(outputDecl));

            var inSchema = StaticSchemaShape.Make<TIn>(inputDecl.Method.ReturnParameter);
            var outSchema = StaticSchemaShape.Make<TOut>(outputDecl.Method.ReturnParameter);
            return new Transformer<TIn, TOut, TTrans>(env, transformer, inSchema, outSchema);
        }

        public static Estimator<TIn, TOut, TTrans> AssertStatic<[IsShape] TIn, [IsShape] TOut, TTrans>(
            this IEstimator<TTrans> estimator, IHostEnvironment env,
            Func<SchemaAssertionContext, TIn> inputDecl,
            Func<SchemaAssertionContext, TOut> outputDecl)
            where TTrans : class, ITransformer
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(estimator, nameof(estimator));
            env.CheckValue(inputDecl, nameof(inputDecl));
            env.CheckValue(outputDecl, nameof(outputDecl));

            var inSchema = StaticSchemaShape.Make<TIn>(inputDecl.Method.ReturnParameter);
            var outSchema = StaticSchemaShape.Make<TOut>(outputDecl.Method.ReturnParameter);
            return new Estimator<TIn, TOut, TTrans>(env, estimator, inSchema, outSchema);
        }
    }
}
