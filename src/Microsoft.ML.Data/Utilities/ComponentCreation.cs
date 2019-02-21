// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Model;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This class defines extension methods for an <see cref="IHostEnvironment"/> to facilitate creating
    /// components (loaders, transforms, trainers, scorers, evaluators, savers).
    /// </summary>
    public static class ComponentCreation
    {
        /// <summary>
        /// Create a new data view which is obtained by appending all columns of all the source data views.
        /// If the data views are of different length, the resulting data view will have the length equal to the
        /// length of the shortest source.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="sources">A non-empty collection of data views to zip together.</param>
        /// <returns>The resulting data view.</returns>
        internal static IDataView Zip(this IHostEnvironment env, IEnumerable<IDataView> sources)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(sources, nameof(sources));
            return ZipDataView.Create(env, sources);
        }

        /// <summary>
        /// Generate training examples for training a predictor or instantiating a scorer.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="data">The data to use for training or scoring.</param>
        /// <param name="features">The name of the features column. Can be null.</param>
        /// <param name="label">The name of the label column. Can be null.</param>
        /// <param name="group">The name of the group ID column (for ranking). Can be null.</param>
        /// <param name="weight">The name of the weight column. Can be null.</param>
        /// <param name="custom">Additional column mapping to be passed to the trainer or scorer (specific to the prediction type). Can be null or empty.</param>
        /// <returns>The constructed examples.</returns>
        [BestFriend]
        internal static RoleMappedData CreateExamples(this IHostEnvironment env, IDataView data, string features, string label = null,
            string group = null, string weight = null, IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> custom = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValueOrNull(label);
            env.CheckValueOrNull(features);
            env.CheckValueOrNull(group);
            env.CheckValueOrNull(weight);
            env.CheckValueOrNull(custom);

            return new RoleMappedData(data, label, features, group, weight, name: null, custom: custom);
        }

        /// <summary>
        /// Create a new <see cref="IDataView"/> over an enumerable of the items of user-defined type.
        /// The user maintains ownership of the <paramref name="data"/> and the resulting data view will
        /// never alter the contents of the <paramref name="data"/>.
        /// Since <see cref="IDataView"/> is assumed to be immutable, the user is expected to support
        /// multiple enumeration of the <paramref name="data"/> that would return the same results, unless
        /// the user knows that the data will only be cursored once.
        ///
        /// One typical usage for streaming data view could be: create the data view that lazily loads data
        /// as needed, then apply pre-trained transformations to it and cursor through it for transformation
        /// results.
        /// </summary>
        /// <typeparam name="TRow">The user-defined item type.</typeparam>
        /// <param name="catalog">The context to use for data view creation.</param>
        /// <param name="data">The data to wrap around.</param>
        /// <param name="schemaDefinition">The optional schema definition of the data view to create. If <c>null</c>,
        /// the schema definition is inferred from <typeparamref name="TRow"/>.</param>
        /// <returns>The constructed <see cref="IDataView"/>.</returns>
        public static IDataView ReadFromEnumerable<TRow>(this DataOperationsCatalog catalog, IEnumerable<TRow> data, SchemaDefinition schemaDefinition = null)
            where TRow : class
        {
            catalog.Environment.CheckValue(data, nameof(data));
            catalog.Environment.CheckValueOrNull(schemaDefinition);
            return DataViewConstructionUtils.CreateFromEnumerable(catalog.Environment, data, schemaDefinition);
        }

        /// <summary>
        /// Create an on-demand prediction engine.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="transformer">The transformer.</param>
        /// <param name="ignoreMissingColumns">Whether to ignore missing columns in the data view.</param>
        /// <param name="inputSchemaDefinition">The optional input schema. If <c>null</c>, the schema is inferred from the <typeparamref name="TSrc"/> type.</param>
        /// <param name="outputSchemaDefinition">The optional output schema. If <c>null</c>, the schema is inferred from the <typeparamref name="TDst"/> type.</param>
        internal static PredictionEngine<TSrc, TDst> CreatePredictionEngine<TSrc, TDst>(this IHostEnvironment env, ITransformer transformer,
            bool ignoreMissingColumns = false, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformer, nameof(transformer));
            env.CheckValueOrNull(inputSchemaDefinition);
            env.CheckValueOrNull(outputSchemaDefinition);
            return new PredictionEngine<TSrc, TDst>(env, transformer, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition);
        }

        /// <summary>
        /// Load the transforms (but not loader) from the model steram and apply them to the specified data.
        /// It is acceptable to have no transforms in the model stream: in this case the original
        /// <paramref name="data"/> will be returned.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="modelStream">The model stream to load from.</param>
        /// <param name="data">The data to apply transforms to.</param>
        /// <returns>The transformed data.</returns>
        internal static IDataView LoadTransforms(this IHostEnvironment env, Stream modelStream, IDataView data)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(modelStream, nameof(modelStream));
            env.CheckValue(data, nameof(data));
            return ModelFileUtils.LoadTransforms(env, data, modelStream);
        }

        // REVIEW: Add one more overload that works off SubComponents.

        /// <summary>
        /// Creates a data loader from the arguments object.
        /// </summary>
        [BestFriend]
        internal static IDataLoader CreateLoader<TArgs>(this IHostEnvironment env, TArgs arguments, IMultiStreamSource files)
            where TArgs : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(files, nameof(files));
            return CreateCore<IDataLoader, TArgs, SignatureDataLoader>(env, arguments, files);
        }

        /// <summary>
        /// Creates a data loader from the 'LoadName{settings}' string.
        /// </summary>
        [BestFriend]
        internal static IDataLoader CreateLoader(this IHostEnvironment env, string settings, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            Contracts.CheckValue(files, nameof(files));
            Type factoryType = typeof(IComponentFactory<IMultiStreamSource, IDataLoader>);
            return CreateCore<IDataLoader>(env, factoryType, typeof(SignatureDataLoader), settings, files);
        }

        /// <summary>
        /// Creates a data saver from the arguments object.
        /// </summary>
        internal static IDataSaver CreateSaver<TArgs>(this IHostEnvironment env, TArgs arguments)
            where TArgs : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            return CreateCore<IDataSaver, TArgs, SignatureDataSaver>(env, arguments);
        }

        /// <summary>
        /// Creates a data saver from the 'LoadName{settings}' string.
        /// </summary>
        internal static IDataSaver CreateSaver(this IHostEnvironment env, string settings)
        {
            Contracts.CheckValue(env, nameof(env));
            return CreateCore<IDataSaver>(env, typeof(SignatureDataSaver), settings);
        }

        /// <summary>
        /// Creates a data transform from the arguments object.
        /// </summary>
        internal static IDataTransform CreateTransform<TArgs>(this IHostEnvironment env, TArgs arguments, IDataView source)
            where TArgs : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(source, nameof(source));
            return CreateCore<IDataTransform, TArgs, SignatureDataTransform>(env, arguments, source);
        }

        /// <summary>
        /// Creates a data transform from the 'LoadName{settings}' string.
        /// </summary>
        internal static IDataTransform CreateTransform(this IHostEnvironment env, string settings, IDataView source)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(source, nameof(source));
            Type factoryType = typeof(IComponentFactory<IDataView, IDataTransform>);
            return CreateCore<IDataTransform>(env, factoryType, typeof(SignatureDataTransform), settings, source);
        }

        /// <summary>
        /// Creates a data scorer from the 'LoadName{settings}' string.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="settings">The settings string.</param>
        /// <param name="data">The data to score.</param>
        /// <param name="predictor">The predictor to score.</param>
        /// <param name="trainSchema">The training data schema from which the scorer can optionally extract
        /// additional information, for example, label names. If this is <c>null</c>, no information will be
        /// extracted.</param>
        /// <returns>The scored data.</returns>
        internal static IDataScorerTransform CreateScorer(this IHostEnvironment env, string settings,
            RoleMappedData data, IPredictor predictor, RoleMappedSchema trainSchema = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValue(predictor, nameof(predictor));
            env.CheckValueOrNull(trainSchema);

            Type factoryType = typeof(IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform>);
            Type signatureType = typeof(SignatureDataScorer);

            ICommandLineComponentFactory scorerFactorySettings = CmdParser.CreateComponentFactory(
                factoryType,
                signatureType,
                settings);

            var bindable = ScoreUtils.GetSchemaBindableMapper(env, predictor, scorerFactorySettings: scorerFactorySettings);
            var mapper = bindable.Bind(env, data.Schema);
            return CreateCore<IDataScorerTransform>(env, factoryType, signatureType, settings, data.Data, mapper, trainSchema);
        }

        /// <summary>
        /// Creates a default data scorer appropriate to the predictor's prediction kind.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="data">The data to score.</param>
        /// <param name="predictor">The predictor to score.</param>
        /// <param name="trainSchema">The training data schema from which the scorer can optionally extract
        /// additional information, for example, label names. If this is <c>null</c>, no information will be
        /// extracted.</param>
        /// <returns>The scored data.</returns>
        [BestFriend]
        internal static IDataScorerTransform CreateDefaultScorer(this IHostEnvironment env, RoleMappedData data,
            IPredictor predictor, RoleMappedSchema trainSchema = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValue(predictor, nameof(predictor));
            env.CheckValueOrNull(trainSchema);

            return ScoreUtils.GetScorer(predictor, data, env, trainSchema);
        }

        [BestFriend]
        internal static IEvaluator CreateEvaluator(this IHostEnvironment env, string settings)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(settings, nameof(settings));
            return CreateCore<IEvaluator>(env, typeof(SignatureEvaluator), settings);
        }

        /// <summary>
        /// Loads a predictor from the model stream. Returns null iff there's no predictor.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="modelStream">The model stream.</param>
        internal static IPredictor LoadPredictorOrNull(this IHostEnvironment env, Stream modelStream)
        {
            Contracts.CheckValue(modelStream, nameof(modelStream));
            return ModelFileUtils.LoadPredictorOrNull(env, modelStream);
        }

        internal static ITrainer CreateTrainer<TArgs>(this IHostEnvironment env, TArgs arguments, out string loadName)
            where TArgs : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            return CreateCore<ITrainer, TArgs, SignatureTrainer>(env, arguments, out loadName);
        }

        internal static ITrainer CreateTrainer(this IHostEnvironment env, string settings, out string loadName)
        {
            Contracts.CheckValue(env, nameof(env));
            return CreateCore<ITrainer>(env, typeof(SignatureTrainer), settings, out loadName);
        }

        private static TRes CreateCore<TRes>(
            IHostEnvironment env,
            Type signatureType,
            string settings,
            params object[] extraArgs)
            where TRes : class
        {
            return CreateCore<TRes>(env, signatureType, settings, out string loadName, extraArgs);
        }

        private static TRes CreateCore<TRes>(
            IHostEnvironment env,
            Type signatureType,
            string settings,
            out string loadName,
            params object[] extraArgs)
            where TRes : class
        {
            return CreateCore<TRes>(env, typeof(IComponentFactory<TRes>), signatureType, settings, out loadName, extraArgs);
        }

        private static TRes CreateCore<TRes>(
            IHostEnvironment env,
            Type factoryType,
            Type signatureType,
            string settings,
            params object[] extraArgs)
            where TRes : class
        {
            string loadName;
            return CreateCore<TRes>(env, factoryType, signatureType, settings, out loadName, extraArgs);
        }

        private static TRes CreateCore<TRes, TArgs, TSig>(IHostEnvironment env, TArgs args, params object[] extraArgs)
            where TRes : class
            where TArgs : class, new()
        {
            string loadName;
            return CreateCore<TRes, TArgs, TSig>(env, args, out loadName, extraArgs);
        }

        private static TRes CreateCore<TRes>(
            IHostEnvironment env,
            Type factoryType,
            Type signatureType,
            string settings,
            out string loadName,
            params object[] extraArgs)
            where TRes : class
        {
            Contracts.AssertValue(env);
            env.AssertValue(factoryType);
            env.AssertValue(signatureType);
            env.AssertValue(settings, "settings");

            var factory = CmdParser.CreateComponentFactory(factoryType, signatureType, settings);
            loadName = factory.Name;
            return ComponentCatalog.CreateInstance<TRes>(env, factory.SignatureType, factory.Name, factory.GetSettingsString(), extraArgs);
        }

        private static TRes CreateCore<TRes, TArgs, TSig>(IHostEnvironment env, TArgs args, out string loadName, params object[] extraArgs)
            where TRes : class
            where TArgs : class, new()
        {
            env.CheckValue(args, nameof(args));

            var classes = env.ComponentCatalog.FindLoadableClasses<TArgs, TSig>();
            if (classes.Length == 0)
                throw env.Except("Couldn't find a {0} class that accepts {1} as arguments.", typeof(TRes).Name, typeof(TArgs).FullName);
            if (classes.Length > 1)
                throw env.Except("Found too many {0} classes that accept {1} as arguments.", typeof(TRes).Name, typeof(TArgs).FullName);

            var lc = classes[0];
            loadName = lc.LoadNames[0];
            return lc.CreateInstance<TRes>(env, args, extraArgs);
        }
    }
}
