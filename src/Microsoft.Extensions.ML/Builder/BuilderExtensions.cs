// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Extensions.DependencyInjection;

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// Extension methods for <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
    /// </summary>
    public static class BuilderExtensions
    {
        /// <summary>
        /// Adds the model at the specified location to the builder.
        /// </summary>
        /// <param name="builder">The builder to which to add the model.</param>
        /// <param name="uri">The location of the model.</param>
        /// <returns>
        /// The updated <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> FromUri<TData, TPrediction>(
            this PredictionEnginePoolBuilder<TData, TPrediction> builder, string uri)
            where TData : class
            where TPrediction : class, new()
        {
            return builder.FromUri(string.Empty, new Uri(uri));
        }

        /// <summary>
        /// Adds the named model at the specified location to the builder.
        /// </summary>
        /// <param name="builder">The builder to which to add the model.</param>
        /// <param name="modelName">
        /// The name of the model which allows for uniquely identifying the model when
        /// multiple models have the same <typeparamref name="TData"/> and
        /// <typeparamref name="TPrediction"/> types.
        /// </param>
        /// <param name="uri">The location of the model.</param>
        /// <returns>
        /// The updated <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> FromUri<TData, TPrediction>(
            this PredictionEnginePoolBuilder<TData, TPrediction> builder, string modelName, string uri)
            where TData : class
            where TPrediction : class, new()
        {
            return builder.FromUri(modelName, new Uri(uri));
        }

        /// <summary>
        /// Adds the named model at the specified location to the builder.
        /// </summary>
        /// <param name="builder">The builder to which to add the model.</param>
        /// <param name="modelName">
        /// The name of the model which allows for uniquely identifying the model when
        /// multiple models have the same <typeparamref name="TData"/> and
        /// <typeparamref name="TPrediction"/> types.
        /// </param>
        /// <param name="uri">The location of the model.</param>
        /// <returns>
        /// The updated <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> FromUri<TData, TPrediction>(
            this PredictionEnginePoolBuilder<TData, TPrediction> builder, string modelName, Uri uri)
            where TData : class where TPrediction : class, new()
        {
            return builder.FromUri(modelName, uri, TimeSpan.FromMinutes(5));
        }

        /// <summary>
        /// Adds the model at the specified location to the builder.
        /// </summary>
        /// <param name="builder">The builder to which to add the model.</param>
        /// <param name="uri">The location of the model.</param>
        /// <param name="period">
        /// How often to query if the model has been updated at the specified location.
        /// </param>
        /// <returns>
        /// The updated <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> FromUri<TData, TPrediction>(
            this PredictionEnginePoolBuilder<TData, TPrediction> builder, string uri, TimeSpan period)
            where TData : class where TPrediction : class, new()
        {
            return builder.FromUri(string.Empty, new Uri(uri), period);
        }

        /// <summary>
        /// Adds the named model at the specified location to the builder.
        /// </summary>
        /// <param name="builder">The builder to which to add the model.</param>
        /// <param name="modelName">
        /// The name of the model which allows for uniquely identifying the model when
        /// multiple models have the same <typeparamref name="TData"/> and
        /// <typeparamref name="TPrediction"/> types.
        /// </param>
        /// <param name="uri">The location of the model.</param>
        /// <param name="period">
        /// How often to query if the model has been updated at the specified location.
        /// </param>
        /// <returns>
        /// The updated <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> FromUri<TData, TPrediction>(
            this PredictionEnginePoolBuilder<TData, TPrediction> builder, string modelName, string uri, TimeSpan period)
            where TData : class
            where TPrediction : class, new()
        {
            return builder.FromUri(modelName, new Uri(uri), period);
        }

        /// <summary>
        /// Adds the named model at the specified location to the builder.
        /// </summary>
        /// <param name="builder">The builder to which to add the model.</param>
        /// <param name="modelName">
        /// The name of the model which allows for uniquely identifying the model when
        /// multiple models have the same <typeparamref name="TData"/> and
        /// <typeparamref name="TPrediction"/> types.
        /// </param>
        /// <param name="uri">The location of the model.</param>
        /// <param name="period">
        /// How often to query if the model has been updated at the specified location.
        /// </param>
        /// <returns>
        /// The updated <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> FromUri<TData, TPrediction>(
            this PredictionEnginePoolBuilder<TData, TPrediction> builder, string modelName, Uri uri, TimeSpan period)
            where TData : class
            where TPrediction : class, new()
        {
            builder.Services.AddTransient<UriModelLoader, UriModelLoader>();
            builder.Services.AddOptions<PredictionEnginePoolOptions<TData, TPrediction>>(modelName)
                .Configure<UriModelLoader>((opt, loader) =>
                {
                    loader.Start(uri, period);
                    opt.ModelLoader = loader;
                });
            return builder;
        }

        /// <summary>
        /// Adds the model at the specified file to the builder.
        /// </summary>
        /// <param name="builder">The builder to which to add the model.</param>
        /// <param name="filePath">The location of the model.</param>
        /// <returns>
        /// The updated <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> FromFile<TData, TPrediction>(
            this PredictionEnginePoolBuilder<TData, TPrediction> builder, string filePath)
            where TData : class
            where TPrediction : class, new()
        {
            return builder.FromFile(string.Empty, filePath, true);
        }

        /// <summary>
        /// Adds the model at the specified file to the builder.
        /// </summary>
        /// <param name="builder">The builder to which to add the model.</param>
        /// <param name="filePath">The location of the model.</param>
        /// <param name="watchForChanges">
        /// Whether to watch for changes to the file path and update the model when the file is changed or not.
        /// </param>
        /// <returns>
        /// The updated <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> FromFile<TData, TPrediction>(
            this PredictionEnginePoolBuilder<TData, TPrediction> builder, string filePath, bool watchForChanges)
            where TData : class
            where TPrediction : class, new()
        {
            return builder.FromFile(string.Empty, filePath, watchForChanges);
        }

        /// <summary>
        /// Adds the model at the specified file to the builder.
        /// </summary>
        /// <param name="builder">The builder to which to add the model.</param>
        /// <param name="modelName">
        /// The name of the model which allows for uniquely identifying the model when
        /// multiple models have the same <typeparamref name="TData"/> and
        /// <typeparamref name="TPrediction"/> types.
        /// </param>
        /// <param name="filePath">The location of the model.</param>
        /// <returns>
        /// The updated <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> FromFile<TData, TPrediction>(
            this PredictionEnginePoolBuilder<TData, TPrediction> builder, string modelName, string filePath)
            where TData : class
            where TPrediction : class, new()
        {
            return builder.FromFile(modelName, filePath, true);
        }

        /// <summary>
        /// Adds the model at the specified file to the builder.
        /// </summary>
        /// <param name="builder">The builder to which to add the model.</param>
        /// <param name="modelName">
        /// The name of the model which allows for uniquely identifying the model when
        /// multiple models have the same <typeparamref name="TData"/> and
        /// <typeparamref name="TPrediction"/> types.
        /// </param>
        /// <param name="filePath">The location of the model.</param>
        /// <param name="watchForChanges">
        /// Whether to watch for changes to the file path and update the model when the file is changed or not.
        /// </param>
        /// <returns>
        /// The updated <see cref="PredictionEnginePoolBuilder{TData, TPrediction}"/>.
        /// </returns>
        public static PredictionEnginePoolBuilder<TData, TPrediction> FromFile<TData, TPrediction>(
            this PredictionEnginePoolBuilder<TData, TPrediction> builder, string modelName, string filePath, bool watchForChanges)
            where TData : class
            where TPrediction : class, new()
        {
            builder.Services.AddTransient<FileModelLoader, FileModelLoader>();
            builder.Services.AddOptions<PredictionEnginePoolOptions<TData, TPrediction>>(modelName)
                .Configure<FileModelLoader>((options, loader) =>
                {
                    loader.Start(filePath, watchForChanges);
                    options.ModelLoader = loader;
                });
            return builder;
        }
    }
}
