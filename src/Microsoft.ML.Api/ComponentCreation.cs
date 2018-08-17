// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Api
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
        public static IDataView Zip(this IHostEnvironment env, IEnumerable<IDataView> sources)
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
        public static RoleMappedData CreateExamples(this IHostEnvironment env, IDataView data, string features, string label = null,
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
        /// Create a new <see cref="IDataView"/> over an in-memory collection of the items of user-defined type.
        /// The user maintains ownership of the <paramref name="data"/> and the resulting data view will
        /// never alter the contents of the <paramref name="data"/>.
        /// Since <see cref="IDataView"/> is assumed to be immutable, the user is expected to not
        /// modify the contents of <paramref name="data"/> while the data view is being actively cursored.
        ///
        /// One typical usage for in-memory data view could be: create the data view, train a predictor.
        /// Once the predictor is fully trained, modify the contents of the underlying collection and
        /// train another predictor.
        /// </summary>
        /// <typeparam name="TRow">The user-defined item type.</typeparam>
        /// <param name="env">The host environment to use for data view creation.</param>
        /// <param name="data">The data to wrap around.</param>
        /// <param name="schemaDefinition">The optional schema definition of the data view to create. If <c>null</c>,
        /// the schema definition is inferred from <typeparamref name="TRow"/>.</param>
        /// <returns>The constructed <see cref="IDataView"/>.</returns>
        public static IDataView CreateDataView<TRow>(this IHostEnvironment env, IList<TRow> data, SchemaDefinition schemaDefinition = null)
            where TRow : class
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValueOrNull(schemaDefinition);
            return DataViewConstructionUtils.CreateFromList(env, data, schemaDefinition);
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
        /// results. This is how <see cref="BatchPredictionEngine{TSrc,TDst}"/> is implemented.
        /// </summary>
        /// <typeparam name="TRow">The user-defined item type.</typeparam>
        /// <param name="env">The host environment to use for data view creation.</param>
        /// <param name="data">The data to wrap around.</param>
        /// <param name="schemaDefinition">The optional schema definition of the data view to create. If <c>null</c>,
        /// the schema definition is inferred from <typeparamref name="TRow"/>.</param>
        /// <returns>The constructed <see cref="IDataView"/>.</returns>
        public static IDataView CreateStreamingDataView<TRow>(this IHostEnvironment env, IEnumerable<TRow> data, SchemaDefinition schemaDefinition = null)
            where TRow : class
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValueOrNull(schemaDefinition);
            return DataViewConstructionUtils.CreateFromEnumerable(env, data, schemaDefinition);
        }

        /// <summary>
        /// Create a batch prediction engine.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="modelStream">The stream to deserialize the pipeline (transforms and predictor) from.</param>
        /// <param name="ignoreMissingColumns">Whether to ignore missing columns in the data view.</param>
        /// <param name="inputSchemaDefinition">The optional input schema. If <c>null</c>, the schema is inferred from the <typeparamref name="TSrc"/> type.</param>
        /// <param name="outputSchemaDefinition">The optional output schema. If <c>null</c>, the schema is inferred from the <typeparamref name="TDst"/> type.</param>
        public static BatchPredictionEngine<TSrc, TDst> CreateBatchPredictionEngine<TSrc, TDst>(this IHostEnvironment env, Stream modelStream,
            bool ignoreMissingColumns = false, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(modelStream, nameof(modelStream));
            env.CheckValueOrNull(inputSchemaDefinition);
            env.CheckValueOrNull(outputSchemaDefinition);
            return new BatchPredictionEngine<TSrc, TDst>(env, modelStream, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition);
        }

        /// <summary>
        /// Create a batch prediction engine.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="dataPipe">The transformation pipe that may or may not include a scorer.</param>
        /// <param name="ignoreMissingColumns">Whether to ignore missing columns in the data view.</param>
        /// <param name="inputSchemaDefinition">The optional input schema. If <c>null</c>, the schema is inferred from the <typeparamref name="TSrc"/> type.</param>
        /// <param name="outputSchemaDefinition">The optional output schema. If <c>null</c>, the schema is inferred from the <typeparamref name="TDst"/> type.</param>
        public static BatchPredictionEngine<TSrc, TDst> CreateBatchPredictionEngine<TSrc, TDst>(this IHostEnvironment env, IDataView dataPipe,
            bool ignoreMissingColumns = false, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(dataPipe, nameof(dataPipe));
            env.CheckValueOrNull(inputSchemaDefinition);
            env.CheckValueOrNull(outputSchemaDefinition);
            return new BatchPredictionEngine<TSrc, TDst>(env, dataPipe, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition);
        }

        /// <summary>
        /// Create an on-demand prediction engine.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="modelStream">The stream to deserialize the pipeline (transforms and predictor) from.</param>
        /// <param name="ignoreMissingColumns">Whether to ignore missing columns in the data view.</param>
        /// <param name="inputSchemaDefinition">The optional input schema. If <c>null</c>, the schema is inferred from the <typeparamref name="TSrc"/> type.</param>
        /// <param name="outputSchemaDefinition">The optional output schema. If <c>null</c>, the schema is inferred from the <typeparamref name="TDst"/> type.</param>
        public static PredictionEngine<TSrc, TDst> CreatePredictionEngine<TSrc, TDst>(this IHostEnvironment env, Stream modelStream,
            bool ignoreMissingColumns = false, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(modelStream, nameof(modelStream));
            env.CheckValueOrNull(inputSchemaDefinition);
            env.CheckValueOrNull(outputSchemaDefinition);
            return new PredictionEngine<TSrc, TDst>(env, modelStream, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition);
        }

        /// <summary>
        /// Create an on-demand prediction engine.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="dataPipe">The transformation pipe that may or may not include a scorer.</param>
        /// <param name="ignoreMissingColumns">Whether to ignore missing columns in the data view.</param>
        /// <param name="inputSchemaDefinition">The optional input schema. If <c>null</c>, the schema is inferred from the <typeparamref name="TSrc"/> type.</param>
        /// <param name="outputSchemaDefinition">The optional output schema. If <c>null</c>, the schema is inferred from the <typeparamref name="TDst"/> type.</param>
        public static PredictionEngine<TSrc, TDst> CreatePredictionEngine<TSrc, TDst>(this IHostEnvironment env, IDataView dataPipe,
            bool ignoreMissingColumns = false, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(dataPipe, nameof(dataPipe));
            env.CheckValueOrNull(inputSchemaDefinition);
            env.CheckValueOrNull(outputSchemaDefinition);
            return new PredictionEngine<TSrc, TDst>(env, dataPipe, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition);
        }

        /// <summary>
        /// Create a prediction engine.
        /// This encapsulates the 'classic' prediction problem, where the input is denoted by the float array of features,
        /// and the output is a float score. For binary classification predictors that can output probability, there are output
        /// fields that report the predicted label and probability.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="modelStream">The model stream to load pipeline from.</param>
        /// <param name="nFeatures">Number of features.</param>
        public static SimplePredictionEngine CreateSimplePredictionEngine(this IHostEnvironment env, Stream modelStream, int nFeatures)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(modelStream, nameof(modelStream));
            env.CheckParam(nFeatures > 0, nameof(nFeatures), "Number of features must be positive.");
            return new SimplePredictionEngine(env, modelStream, nFeatures);
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
        public static IDataView LoadTransforms(this IHostEnvironment env, Stream modelStream, IDataView data)
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
        public static IDataLoader CreateLoader<TArgs>(this IHostEnvironment env, TArgs arguments, IMultiStreamSource files)
            where TArgs : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(files, nameof(files));
            return CreateCore<IDataLoader, TArgs, SignatureDataLoader>(env, arguments, files);
        }

        /// <summary>
        /// Creates a data loader from the 'LoadName{settings}' string.
        /// </summary>
        public static IDataLoader CreateLoader(this IHostEnvironment env, string settings, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            Contracts.CheckValue(files, nameof(files));
            return CreateCore<IDataLoader, SignatureDataLoader>(env, settings, files);
        }

        /// <summary>
        /// Creates a data saver from the arguments object.
        /// </summary>
        public static IDataSaver CreateSaver<TArgs>(this IHostEnvironment env, TArgs arguments)
            where TArgs : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            return CreateCore<IDataSaver, TArgs, SignatureDataSaver>(env, arguments);
        }

        /// <summary>
        /// Creates a data saver from the 'LoadName{settings}' string.
        /// </summary>
        public static IDataSaver CreateSaver(this IHostEnvironment env, string settings)
        {
            Contracts.CheckValue(env, nameof(env));
            return CreateCore<IDataSaver, SignatureDataSaver>(env, settings);
        }

        /// <summary>
        /// Creates a data transform from the arguments object.
        /// </summary>
        public static IDataTransform CreateTransform<TArgs>(this IHostEnvironment env, TArgs arguments, IDataView source)
            where TArgs : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(source, nameof(source));
            return CreateCore<IDataTransform, TArgs, SignatureDataTransform>(env, arguments, source);
        }

        /// <summary>
        /// Creates a data transform from the 'LoadName{settings}' string.
        /// </summary>
        public static IDataTransform CreateTransform(this IHostEnvironment env, string settings, IDataView source)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(source, nameof(source));
            return CreateCore<IDataTransform, SignatureDataTransform>(env, settings, source);
        }

        /// <summary>
        /// Creates a data scorer from the 'LoadName{settings}' string.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="settings">The settings string.</param>
        /// <param name="data">The data to score.</param>
        /// <param name="predictor">The predictor to score.</param>
        /// <param name="trainSchema">The training data schema from which the scorer can optionally extract
        /// additional information, e.g., label names. If this is <c>null</c>, no information will be
        /// extracted.</param>
        /// <returns>The scored data.</returns>
        public static IDataScorerTransform CreateScorer(this IHostEnvironment env, string settings,
            RoleMappedData data, Predictor predictor, RoleMappedSchema trainSchema = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValue(predictor, nameof(predictor));
            env.CheckValueOrNull(trainSchema);

            ICommandLineComponentFactory scorerFactorySettings = ParseScorerSettings(settings);
            var bindable = ScoreUtils.GetSchemaBindableMapper(env, predictor.Pred, scorerFactorySettings: scorerFactorySettings);
            var mapper = bindable.Bind(env, data.Schema);
            return CreateCore<IDataScorerTransform, SignatureDataScorer>(env, settings, data.Data, mapper, trainSchema);
        }

        private static ICommandLineComponentFactory ParseScorerSettings(string settings)
        {
            return CmdParser.CreateComponentFactory(
                typeof(IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform>),
                typeof(SignatureDataScorer),
                settings);
        }

        /// <summary>
        /// Creates a default data scorer appropriate to the predictor's prediction kind.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="data">The data to score.</param>
        /// <param name="predictor">The predictor to score.</param>
        /// <param name="trainSchema">The training data schema from which the scorer can optionally extract
        /// additional information, e.g., label names. If this is <c>null</c>, no information will be
        /// extracted.</param>
        /// <returns>The scored data.</returns>
        public static IDataScorerTransform CreateDefaultScorer(this IHostEnvironment env, RoleMappedData data,
            Predictor predictor, RoleMappedSchema trainSchema = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValue(predictor, nameof(predictor));
            env.CheckValueOrNull(trainSchema);

            return ScoreUtils.GetScorer(predictor.Pred, data, env, trainSchema);
        }

        public static IEvaluator CreateEvaluator(this IHostEnvironment env, string settings)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(settings, nameof(settings));
            return CreateCore<IEvaluator, SignatureEvaluator>(env, settings);
        }

        /// <summary>
        /// Loads a predictor from the model stream. Returns null iff there's no predictor.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="modelStream">The model stream.</param>
        public static Predictor LoadPredictorOrNull(this IHostEnvironment env, Stream modelStream)
        {
            Contracts.CheckValue(modelStream, nameof(modelStream));
            var p = ModelFileUtils.LoadPredictorOrNull(env, modelStream);
            return p == null ? null : new Predictor(p);
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
            return CreateCore<ITrainer, SignatureTrainer>(env, settings, out loadName);
        }

        private static TRes CreateCore<TRes, TSig>(IHostEnvironment env, string settings, params object[] extraArgs)
            where TRes : class
        {
            string loadName;
            return CreateCore<TRes, TSig>(env, settings, out loadName, extraArgs);
        }

        private static TRes CreateCore<TRes, TArgs, TSig>(IHostEnvironment env, TArgs args, params object[] extraArgs)
            where TRes : class
            where TArgs : class, new()
        {
            string loadName;
            return CreateCore<TRes, TArgs, TSig>(env, args, out loadName, extraArgs);
        }

        private static TRes CreateCore<TRes, TSig>(IHostEnvironment env, string settings, out string loadName, params object[] extraArgs)
            where TRes : class
        {
            Contracts.AssertValue(env);
            env.AssertValue(settings, "settings");

            var sc = SubComponent.Parse<TRes, TSig>(settings);
            loadName = sc.Kind;
            return sc.CreateInstance(env, extraArgs);
        }

        private static TRes CreateCore<TRes, TArgs, TSig>(IHostEnvironment env, TArgs args, out string loadName, params object[] extraArgs)
            where TRes : class
            where TArgs : class, new()
        {
            env.CheckValue(args, nameof(args));

            var classes = ComponentCatalog.FindLoadableClasses<TArgs, TSig>();
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
