// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(ScoreCommand.Summary, typeof(ScoreCommand), typeof(ScoreCommand.Arguments), typeof(SignatureCommand),
    "Score Predictor", "Score")]

namespace Microsoft.ML.Runtime.Data
{
    using TScorerFactory = IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform>;

    public interface IDataScorerTransform : IDataTransform, ITransformTemplate
    {
    }

    /// <summary>
    /// Signature for creating an <see cref="IDataScorerTransform"/>.
    /// </summary>
    /// <param name="data">The data containing the columns to score</param>
    /// <param name="mapper">The mapper, already bound to the schema column in <paramref name="data"/></param>
    /// <param name="trainSchema">This parameter holds a snapshot of the role mapped training schema as
    /// it existed at the point when <paramref name="mapper"/> was trained, or <c>null</c> if it not
    /// available for some reason</param>
    public delegate void SignatureDataScorer(IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema);

    public delegate void SignatureBindableMapper(IPredictor predictor);

    public sealed class ScoreCommand : DataCommand.ImplBase<ScoreCommand.Arguments>
    {
        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for features when scorer is not defined", ShortName = "feat")]
            public string FeatureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Group column name", ShortName = "group")]
            public string GroupColumn = DefaultColumnNames.GroupId;

            [Argument(ArgumentType.Multiple,
                HelpText = "Input columns: Columns with custom kinds declared through key assignments, e.g., col[Kind]=Name to assign column named 'Name' kind 'Kind'",
                ShortName = "col", SortOrder = 10)]
            public KeyValuePair<string, string>[] CustomColumn;

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", SignatureType = typeof(SignatureDataScorer))]
            public TScorerFactory Scorer;

            [Argument(ArgumentType.Multiple, HelpText = "The data saver to use", SignatureType = typeof(SignatureDataSaver))]
            public IComponentFactory<IDataSaver> Saver;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "File to save the data", ShortName = "dout")]
            public string OutputDataFile;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to include hidden columns", ShortName = "keep")]
            public bool KeepHidden;

            [Argument(ArgumentType.Multiple, HelpText = "Post processing transform", ShortName = "pxf", SignatureType = typeof(SignatureDataTransform))]
            public KeyValuePair<string, IComponentFactory<IDataView, IDataTransform>>[] PostTransform;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to output all columns or just scores", ShortName = "all")]
            public bool? OutputAllColumns;

            [Argument(ArgumentType.Multiple, HelpText = "What columns to output beyond score columns, if outputAllColumns=-.", ShortName = "outCol")]
            public string[] OutputColumn;
        }

        internal const string Summary = "Scores a data file.";

        public ScoreCommand(IHostEnvironment env, Arguments args)
            : base(env, args, nameof(ScoreCommand))
        {
            Host.CheckUserArg(!string.IsNullOrWhiteSpace(Args.InputModelFile), nameof(Args.InputModelFile), "The input model file is required.");
            Host.CheckUserArg(!string.IsNullOrWhiteSpace(Args.OutputDataFile), nameof(Args.OutputDataFile), "The output data file is required.");
            Utils.CheckOptionalUserDirectory(Args.OutputDataFile, nameof(Args.OutputDataFile));
        }

        public override void Run()
        {
            using (var ch = Host.Start("Score"))
            {
                RunCore(ch);
            }
        }

        private void RunCore(IChannel ch)
        {
            Host.AssertValue(ch);

            ch.Trace("Creating loader");

            LoadModelObjects(ch, true, out var predictor, true, out var trainSchema, out var loader);
            ch.AssertValue(predictor);
            ch.AssertValueOrNull(trainSchema);
            ch.AssertValue(loader);

            ch.Trace("Creating pipeline");
            var scorer = Args.Scorer;
            ch.Assert(scorer == null || scorer is ICommandLineComponentFactory, "ScoreCommand should only be used from the command line.");
            var bindable = ScoreUtils.GetSchemaBindableMapper(Host, predictor, scorerFactorySettings: scorer as ICommandLineComponentFactory);
            ch.AssertValue(bindable);

            // REVIEW: We probably ought to prefer role mappings from the training schema.
            string feat = TrainUtils.MatchNameOrDefaultOrNull(ch, loader.Schema,
                nameof(Args.FeatureColumn), Args.FeatureColumn, DefaultColumnNames.Features);
            string group = TrainUtils.MatchNameOrDefaultOrNull(ch, loader.Schema,
                nameof(Args.GroupColumn), Args.GroupColumn, DefaultColumnNames.GroupId);
            var customCols = TrainUtils.CheckAndGenerateCustomColumns(ch, Args.CustomColumn);
            var schema = new RoleMappedSchema(loader.Schema, label: null, feature: feat, group: group, custom: customCols, opt: true);
            var mapper = bindable.Bind(Host, schema);

            if (scorer == null)
                scorer = ScoreUtils.GetScorerComponent(Host, mapper);

            loader = CompositeDataLoader.ApplyTransform(Host, loader, "Scorer", scorer.ToString(),
                (env, view) => scorer.CreateComponent(env, view, mapper, trainSchema));

            loader = CompositeDataLoader.Create(Host, loader, Args.PostTransform);

            if (!string.IsNullOrWhiteSpace(Args.OutputModelFile))
            {
                ch.Trace("Saving the data pipe");
                SaveLoader(loader, Args.OutputModelFile);
            }

            ch.Trace("Creating saver");
            IDataSaver writer;
            if (Args.Saver == null)
            {
                var ext = Path.GetExtension(Args.OutputDataFile);
                var isText = ext == ".txt" || ext == ".tlc";
                if (isText)
                {
                    writer = new TextSaver(Host, new TextSaver.Arguments());
                }
                else
                {
                    writer = new BinarySaver(Host, new BinarySaver.Arguments());
                }
            }
            else
            {
                writer = Args.Saver.CreateComponent(Host);
            }
            ch.Assert(writer != null);
            var outputIsBinary = writer is BinaryWriter;

            bool outputAllColumns =
                Args.OutputAllColumns == true
                || (Args.OutputAllColumns == null && Utils.Size(Args.OutputColumn) == 0 && outputIsBinary);

            bool outputNamesAndLabels =
                Args.OutputAllColumns == true || Utils.Size(Args.OutputColumn) == 0;

            if (Args.OutputAllColumns == true && Utils.Size(Args.OutputColumn) != 0)
                ch.Warning(nameof(Args.OutputAllColumns) + "=+ always writes all columns irrespective of " + nameof(Args.OutputColumn) + " specified.");

            if (!outputAllColumns && Utils.Size(Args.OutputColumn) != 0)
            {
                foreach (var outCol in Args.OutputColumn)
                {
                    if (!loader.Schema.TryGetColumnIndex(outCol, out int dummyColIndex))
                        throw ch.ExceptUserArg(nameof(Arguments.OutputColumn), "Column '{0}' not found.", outCol);
                }
            }

            uint maxScoreId = 0;
            if (!outputAllColumns)
                maxScoreId = loader.Schema.GetMaxMetadataKind(out int colMax, MetadataUtils.Kinds.ScoreColumnSetId);
            ch.Assert(outputAllColumns || maxScoreId > 0); // score set IDs are one-based
            var cols = new List<int>();
            for (int i = 0; i < loader.Schema.ColumnCount; i++)
            {
                if (!Args.KeepHidden && loader.Schema.IsHidden(i))
                    continue;
                if (!(outputAllColumns || ShouldAddColumn(loader.Schema, i, maxScoreId, outputNamesAndLabels)))
                    continue;
                var type = loader.Schema.GetColumnType(i);
                if (writer.IsColumnSavable(type))
                    cols.Add(i);
                else
                {
                    ch.Warning("The column '{0}' will not be written as it has unsavable column type.",
                        loader.Schema.GetColumnName(i));
                }
            }

            ch.Check(cols.Count > 0, "No valid columns to save");

            ch.Trace("Scoring and saving data");
            using (var file = Host.CreateOutputFile(Args.OutputDataFile))
            using (var stream = file.CreateWriteStream())
                writer.SaveData(stream, loader, cols.ToArray());
        }

        /// <summary>
        /// Whether a column should be added, assuming it's not hidden
        /// (i.e.: this doesn't check for hidden
        /// </summary>
        private bool ShouldAddColumn(ISchema schema, int i, uint scoreSet, bool outputNamesAndLabels)
        {
            uint scoreSetId = 0;
            if (schema.TryGetMetadata(MetadataUtils.ScoreColumnSetIdType.AsPrimitive, MetadataUtils.Kinds.ScoreColumnSetId, i, ref scoreSetId)
                && scoreSetId == scoreSet)
            {
                return true;
            }
            if (outputNamesAndLabels)
            {
                switch (schema.GetColumnName(i))
                {
                    case "Label":
                    case "Name":
                    case "Names":
                        return true;
                    default:
                        break;
                }
            }
            if (Args.OutputColumn != null && Array.FindIndex(Args.OutputColumn, schema.GetColumnName(i).Equals) >= 0)
                return true;
            return false;
        }
    }

    public static class ScoreUtils
    {
        public static IDataScorerTransform GetScorer(IPredictor predictor, RoleMappedData data, IHostEnvironment env, RoleMappedSchema trainSchema)
        {
            var sc = GetScorerComponentAndMapper(predictor, null, data.Schema, env, null, out var mapper);
            return sc.CreateComponent(env, data.Data, mapper, trainSchema);
        }

        public static IDataScorerTransform GetScorer(
            TScorerFactory scorer,
            IPredictor predictor,
            IDataView input,
            string featureColName,
            string groupColName,
            IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> customColumns,
            IHostEnvironment env,
            RoleMappedSchema trainSchema,
            IComponentFactory<IPredictor, ISchemaBindableMapper> mapperFactory = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValueOrNull(scorer);
            env.CheckValue(predictor, nameof(predictor));
            env.CheckValue(input, nameof(input));
            env.CheckValueOrNull(featureColName);
            env.CheckValueOrNull(groupColName);
            env.CheckValueOrNull(customColumns);
            env.CheckValueOrNull(trainSchema);

            var schema = new RoleMappedSchema(input.Schema, label: null, feature: featureColName, group: groupColName, custom: customColumns, opt: true);
            var sc = GetScorerComponentAndMapper(predictor, scorer, schema, env, mapperFactory, out var mapper);
            return sc.CreateComponent(env, input, mapper, trainSchema);
        }

        /// <summary>
        /// Determines the scorer component factory (if the given one is null or empty), and creates the schema bound mapper.
        /// </summary>
        private static TScorerFactory GetScorerComponentAndMapper(
            IPredictor predictor,
            TScorerFactory scorerFactory,
            RoleMappedSchema schema,
            IHostEnvironment env,
            IComponentFactory<IPredictor, ISchemaBindableMapper> mapperFactory,
            out ISchemaBoundMapper mapper)
        {
            Contracts.AssertValue(env);

            var bindable = GetSchemaBindableMapper(env, predictor, mapperFactory, scorerFactory as ICommandLineComponentFactory);
            env.AssertValue(bindable);
            mapper = bindable.Bind(env, schema);
            if (scorerFactory != null)
                return scorerFactory;
            return GetScorerComponent(env, mapper);
        }

        /// <summary>
        /// Determine the default scorer for a schema bound mapper. This looks for text-valued ScoreColumnKind
        /// metadata on the first column of the mapper. If that text is found and maps to a scorer loadable class,
        /// that component is used. Otherwise, the GenericScorer is used.
        /// </summary>
        /// <param name="environment">The host environment.</param>.
        /// <param name="mapper">The schema bound mapper to get the default scorer.</param>.
        /// <param name="suffix">An optional suffix to append to the default column names.</param>
        public static TScorerFactory GetScorerComponent(
            IHostEnvironment environment,
            ISchemaBoundMapper mapper,
            string suffix = null)
        {
            Contracts.CheckValue(environment, nameof(environment));
            Contracts.AssertValue(mapper);

            ComponentCatalog.LoadableClassInfo info = null;
            ReadOnlyMemory<char> scoreKind = default;
            if (mapper.Schema.ColumnCount > 0 &&
                mapper.Schema.TryGetMetadata(TextType.Instance, MetadataUtils.Kinds.ScoreColumnKind, 0, ref scoreKind) &&
                !scoreKind.IsEmpty)
            {
                var loadName = scoreKind.ToString();
                info = environment.ComponentCatalog.GetLoadableClassInfo<SignatureDataScorer>(loadName);
                if (info == null || !typeof(IDataScorerTransform).IsAssignableFrom(info.Type))
                    info = null;
            }

            Func<IHostEnvironment, IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform> factoryFunc;
            if (info == null)
            {
                factoryFunc = (env, data, innerMapper, trainSchema) =>
                    new GenericScorer(
                        env,
                        new GenericScorer.Arguments() { Suffix = suffix },
                        data,
                        innerMapper,
                        trainSchema);
            }
            else
            {
                factoryFunc = (env, data, innerMapper, trainSchema) =>
                {
                    object args = info.CreateArguments();
                    if (args is ScorerArgumentsBase scorerArgs)
                    {
                        scorerArgs.Suffix = suffix;
                    }
                    return (IDataScorerTransform)info.CreateInstance(
                        env,
                        args,
                        new object[] { data, innerMapper, trainSchema });
                };
            }

            return ComponentFactoryUtils.CreateFromFunction(factoryFunc);
        }

        /// <summary>
        /// Given a predictor, an optional mapper factory, and an optional scorer factory settings,
        /// produces a compatible ISchemaBindableMapper.
        /// First, it tries to instantiate the bindable mapper using the mapper factory.
        /// Next, it tries to instantiate the bindable mapper using the <paramref name="scorerFactorySettings"/>
        /// (this will only succeed if there's a registered BindableMapper creation method with load name equal to the one
        /// of the scorer).
        /// If the above fails, it checks whether the predictor implements <see cref="ISchemaBindableMapper"/>
        /// directly.
        /// If this also isn't true, it will create a 'matching' standard mapper.
        /// </summary>
        public static ISchemaBindableMapper GetSchemaBindableMapper(
            IHostEnvironment env,
            IPredictor predictor,
            IComponentFactory<IPredictor, ISchemaBindableMapper> mapperFactory = null,
            ICommandLineComponentFactory scorerFactorySettings = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(predictor, nameof(predictor));
            env.CheckValueOrNull(mapperFactory);
            env.CheckValueOrNull(scorerFactorySettings);

            // if the mapperFactory was supplied, use it
            if (mapperFactory != null)
                return mapperFactory.CreateComponent(env, predictor);

            // See if we can instantiate a mapper using scorer arguments.
            if (scorerFactorySettings != null && TryCreateBindableFromScorer(env, predictor, scorerFactorySettings, out var bindable))
                return bindable;

            // The easy case is that the predictor implements the interface.
            bindable = predictor as ISchemaBindableMapper;
            if (bindable != null)
                return bindable;

            // Use one of the standard wrappers.
            if (predictor is IValueMapperDist)
                return new SchemaBindableBinaryPredictorWrapper(predictor);

            return new SchemaBindablePredictorWrapper(predictor);
        }

        private static bool TryCreateBindableFromScorer(IHostEnvironment env, IPredictor predictor,
            ICommandLineComponentFactory scorerSettings, out ISchemaBindableMapper bindable)
        {
            Contracts.AssertValue(env);
            env.AssertValue(predictor);
            env.AssertValue(scorerSettings);

            // Try to find a mapper factory method with the same loadname as the scorer settings.
            return ComponentCatalog.TryCreateInstance<ISchemaBindableMapper, SignatureBindableMapper>(
                env, out bindable, scorerSettings.Name, scorerSettings.GetSettingsString(), predictor);
        }
    }
}