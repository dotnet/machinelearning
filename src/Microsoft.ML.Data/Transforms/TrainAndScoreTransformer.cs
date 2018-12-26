// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Model;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(ScoringTransformer.Summary, typeof(IDataTransform), typeof(ScoringTransformer), typeof(ScoringTransformer.Arguments), typeof(SignatureDataTransform),
    "Score Predictor", "Score")]

[assembly: LoadableClass(TrainAndScoreTransformer.Summary, typeof(IDataTransform), typeof(TrainAndScoreTransformer),
    typeof(TrainAndScoreTransformer.Arguments), typeof(SignatureDataTransform), "Train and Score Predictor", "TrainScore")]

namespace Microsoft.ML.Transforms
{
    [BestFriend]
    internal static class ScoringTransformer
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for features when scorer is not defined",
                ShortName = "feat", SortOrder = 1,
                Purpose = SpecialPurpose.ColumnName)]
            public string FeatureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Group column name", ShortName = "group", SortOrder = 100,
                Purpose = SpecialPurpose.ColumnName)]
            public string GroupColumn = DefaultColumnNames.GroupId;

            [Argument(ArgumentType.Multiple,
                HelpText = "Input columns: Columns with custom kinds declared through key assignments, for example, col[Kind]=Name to assign column named 'Name' kind 'Kind'",
                ShortName = "col", SortOrder = 101, Purpose = SpecialPurpose.ColumnSelector)]
            public KeyValuePair<string, string>[] CustomColumn;

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", NullName = "<Auto>", SignatureType = typeof(SignatureDataScorer))]
            public IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform> Scorer;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "Predictor model file used in scoring",
                ShortName = "in", SortOrder = 2)]
            public string InputModelFile;
        }

        internal const string Summary = "Runs a previously trained predictor on the data.";

        /// <summary>
        /// Convenience method for creating <see cref="ScoringTransformer"/>.
        /// The <see cref="ScoringTransformer"/> allows for model stacking (i.e. to combine information from multiple predictive models to generate a new model)
        /// in the pipeline by using the scores from an already trained model.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>.</param>
        /// <param name="inputModelFile">The model file.</param>
        /// <param name="featureColumn">Role name for the features.</param>
        /// <param name="groupColumn">Role name for the group column.</param>
        public static IDataTransform Create(IHostEnvironment env,
            IDataView input,
            string inputModelFile,
            string featureColumn = DefaultColumnNames.Features,
            string groupColumn = DefaultColumnNames.GroupId)
        {
            var args = new Arguments()
            {
                FeatureColumn = featureColumn,
                GroupColumn = groupColumn,
                InputModelFile = inputModelFile
            };

            return Create(env, args, input);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckUserArg(!string.IsNullOrWhiteSpace(args.InputModelFile), nameof(args.InputModelFile), "The input model file is required.");

            IPredictor predictor;
            RoleMappedSchema trainSchema = null;
            using (var file = env.OpenInputFile(args.InputModelFile))
            using (var strm = file.OpenReadStream())
            using (var rep = RepositoryReader.Open(strm, env))
            {
                ModelLoadContext.LoadModel<IPredictor, SignatureLoadModel>(env, out predictor, rep, ModelFileUtils.DirPredictor);
                trainSchema = ModelFileUtils.LoadRoleMappedSchemaOrNull(env, rep);
            }

            string feat = TrainUtils.MatchNameOrDefaultOrNull(env, input.Schema,
                    nameof(args.FeatureColumn), args.FeatureColumn, DefaultColumnNames.Features);
            string group = TrainUtils.MatchNameOrDefaultOrNull(env, input.Schema,
                nameof(args.GroupColumn), args.GroupColumn, DefaultColumnNames.GroupId);
            var customCols = TrainUtils.CheckAndGenerateCustomColumns(env, args.CustomColumn);

            return ScoreUtils.GetScorer(args.Scorer, predictor, input, feat, group, customCols, env, trainSchema);
        }
    }

    // Essentially, all trainer estimators when fitted return a transformer that produces scores -- which is to say, all machine
    // learning algorithms actually behave more or less as this transform used to, so its presence is no longer necessary or helpful,
    // from an API perspective, but this is still how things are invoked from the command line for now.
    [BestFriend]
    internal static class TrainAndScoreTransformer
    {
        public abstract class ArgumentsBase : TransformInputBase
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for features when scorer is not defined",
                ShortName = "feat", SortOrder = 102, Purpose = SpecialPurpose.ColumnName)]
            public string FeatureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for labels", ShortName = "lab", SortOrder = 103,
                Purpose = SpecialPurpose.ColumnName)]
            public string LabelColumn = DefaultColumnNames.Label;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for grouping", ShortName = "group",
                SortOrder = 105, Purpose = SpecialPurpose.ColumnName)]
            public string GroupColumn = DefaultColumnNames.GroupId;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for example weight", ShortName = "weight",
                SortOrder = 104, Purpose = SpecialPurpose.ColumnName)]
            public string WeightColumn = DefaultColumnNames.Weight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name column name", ShortName = "name", SortOrder = 106,
                Purpose = SpecialPurpose.ColumnName)]
            public string NameColumn = DefaultColumnNames.Name;

            [Argument(ArgumentType.Multiple,
                HelpText = "Input columns: Columns with custom kinds declared through key assignments, for example, col[Kind]=Name to assign column named 'Name' kind 'Kind'",
                ShortName = "col", SortOrder = 110, Purpose = SpecialPurpose.ColumnSelector)]
            public KeyValuePair<string, string>[] CustomColumn;

            public void CopyTo(ArgumentsBase other)
            {
                other.FeatureColumn = FeatureColumn;
                other.LabelColumn = LabelColumn;
                other.GroupColumn = GroupColumn;
                other.WeightColumn = WeightColumn;
                other.NameColumn = NameColumn;
                other.CustomColumn = CustomColumn;
            }
        }

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Trainer to use", ShortName = "tr", NullName = "<None>", SortOrder = 1, SignatureType = typeof(SignatureTrainer))]
            public IComponentFactory<ITrainer> Trainer;

            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", NullName = "<None>", SignatureType = typeof(SignatureCalibrator))]
            public IComponentFactory<ICalibratorTrainer> Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of instances to train the calibrator", ShortName = "numcali")]
            public int MaxCalibrationExamples = 1000000000;

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", NullName = "<Auto>", SignatureType = typeof(SignatureDataScorer))]
            public IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform> Scorer;
        }

        internal const string Summary = "Trains a predictor, or loads it from a file, and runs it on the data.";

        /// <summary>
        /// Convenience method for creating <see cref="TrainAndScoreTransformer"/>.
        /// The <see cref="TrainAndScoreTransformer"/> allows for model stacking (i.e. to combine information from multiple predictive models to generate a new model)
        /// in the pipeline by training a model first and then using the scores from the trained model.
        ///
        /// Unlike <see cref="ScoringTransformer"/>, the <see cref="TrainAndScoreTransformer"/> trains the model on the fly as name indicates.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>.</param>
        /// <param name="trainer">The <see cref="ITrainer"/> object i.e. the learning algorithm that will be used for training the model.</param>
        /// <param name="featureColumn">Role name for features.</param>
        /// <param name="labelColumn">Role name for label.</param>
        /// <param name="groupColumn">Role name for the group column.</param>
        public static IDataTransform Create(IHostEnvironment env,
            IDataView input,
            ITrainer trainer,
            string featureColumn = DefaultColumnNames.Features,
            string labelColumn = DefaultColumnNames.Label,
            string groupColumn = DefaultColumnNames.GroupId)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));
            env.CheckValue(trainer, nameof(trainer));
            env.CheckValue(featureColumn, nameof(featureColumn));
            env.CheckValue(labelColumn, nameof(labelColumn));
            env.CheckValue(groupColumn, nameof(groupColumn));

            var args = new Arguments()
            {
                FeatureColumn = featureColumn,
                LabelColumn = labelColumn,
                GroupColumn = groupColumn
            };

            return Create(env, args, trainer, input, null);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(args.Trainer, nameof(args.Trainer),
                "Trainer cannot be null. If your model is already trained, please use ScoreTransform instead.");
            env.CheckValue(input, nameof(input));

            return Create(env, args, args.Trainer.CreateComponent(env), input, null);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input, IComponentFactory<IPredictor, ISchemaBindableMapper> mapperFactory)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(args.Trainer, nameof(args.Trainer),
                "Trainer cannot be null. If your model is already trained, please use ScoreTransform instead.");
            env.CheckValue(input, nameof(input));
            env.CheckValueOrNull(mapperFactory);

            return Create(env, args, args.Trainer.CreateComponent(env), input, mapperFactory);
        }

        private static IDataTransform Create(IHostEnvironment env, Arguments args, ITrainer trainer, IDataView input, IComponentFactory<IPredictor, ISchemaBindableMapper> mapperFactory)
        {
            Contracts.AssertValue(env, nameof(env));
            env.AssertValue(args, nameof(args));
            env.AssertValue(trainer, nameof(trainer));
            env.AssertValue(input, nameof(input));

            var host = env.Register("TrainAndScoreTransform");

            using (var ch = host.Start("Train"))
            {
                ch.Trace("Constructing trainer");
                var customCols = TrainUtils.CheckAndGenerateCustomColumns(env, args.CustomColumn);
                string feat;
                string group;
                var data = CreateDataFromArgs(ch, input, args, out feat, out group);
                var predictor = TrainUtils.Train(host, ch, data, trainer, null,
                    args.Calibrator, args.MaxCalibrationExamples, null);

                return ScoreUtils.GetScorer(args.Scorer, predictor, input, feat, group, customCols, env, data.Schema, mapperFactory);
            }
        }

        public static RoleMappedData CreateDataFromArgs(IExceptionContext ectx, IDataView input,
            ArgumentsBase args)
        {
            string feat;
            string group;
            return CreateDataFromArgs(ectx, input, args, out feat, out group);
        }

        private static RoleMappedData CreateDataFromArgs(IExceptionContext ectx, IDataView input,
            ArgumentsBase args, out string feat, out string group)
        {
            var schema = input.Schema;
            feat = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, nameof(args.FeatureColumn), args.FeatureColumn,
                DefaultColumnNames.Features);
            var label = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, nameof(args.LabelColumn), args.LabelColumn,
                DefaultColumnNames.Label);
            group = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, nameof(args.GroupColumn), args.GroupColumn,
                DefaultColumnNames.GroupId);
            var weight = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, nameof(args.WeightColumn), args.WeightColumn,
                DefaultColumnNames.Weight);
            var name = TrainUtils.MatchNameOrDefaultOrNull(ectx, schema, nameof(args.NameColumn), args.NameColumn,
                DefaultColumnNames.Name);
            var customCols = TrainUtils.CheckAndGenerateCustomColumns(ectx, args.CustomColumn);
            return new RoleMappedData(input, label, feat, group, weight, name, customCols);
        }
    }
}
