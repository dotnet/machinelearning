// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(ScoreTransform.Summary, typeof(IDataTransform), typeof(ScoreTransform), typeof(ScoreTransform.Arguments), typeof(SignatureDataTransform),
    "Score Predictor", "Score")]

[assembly: LoadableClass(TrainAndScoreTransform.Summary, typeof(IDataTransform), typeof(TrainAndScoreTransform),
    typeof(TrainAndScoreTransform.Arguments), typeof(SignatureDataTransform), "Train and Score Predictor", "TrainScore")]

namespace Microsoft.ML.Runtime.Data
{
    public static class ScoreTransform
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
                HelpText = "Input columns: Columns with custom kinds declared through key assignments, e.g., col[Kind]=Name to assign column named 'Name' kind 'Kind'",
                ShortName = "col", SortOrder = 101, Purpose = SpecialPurpose.ColumnSelector)]
            public KeyValuePair<string, string>[] CustomColumn;

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", NullName = "<Auto>")]
            public SubComponent<IDataScorerTransform, SignatureDataScorer> Scorer;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "Predictor model file used in scoring",
                ShortName = "in", SortOrder = 2)]
            public string InputModelFile;
        }

        internal const string Summary = "Runs a previously trained predictor on the data.";

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
                    "featureColumn", args.FeatureColumn, DefaultColumnNames.Features);
            string group = TrainUtils.MatchNameOrDefaultOrNull(env, input.Schema,
                "groupColumn", args.GroupColumn, DefaultColumnNames.GroupId);
            var customCols = TrainUtils.CheckAndGenerateCustomColumns(env, args.CustomColumn);

            return ScoreUtils.GetScorer(args.Scorer, predictor, input, feat, group, customCols, env, trainSchema);
        }
    }

    public static class TrainAndScoreTransform
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
                HelpText = "Input columns: Columns with custom kinds declared through key assignments, e.g., col[Kind]=Name to assign column named 'Name' kind 'Kind'",
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

        public abstract class ArgumentsBase<TSigTrainer> : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Trainer to use", ShortName = "tr", NullName = "<None>", SortOrder = 1)]
            public SubComponent<ITrainer, TSigTrainer> Trainer;
        }

        public sealed class Arguments : ArgumentsBase<SignatureTrainer>
        {
            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", NullName = "<None>")]
            public SubComponent<ICalibratorTrainer, SignatureCalibrator> Calibrator = new SubComponent<ICalibratorTrainer, SignatureCalibrator>("PlattCalibration");

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of instances to train the calibrator", ShortName = "numcali")]
            public int MaxCalibrationExamples = 1000000000;

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", NullName = "<Auto>")]
            public SubComponent<IDataScorerTransform, SignatureDataScorer> Scorer;
        }

        internal const string Summary = "Trains a predictor, or loads it from a file, and runs it on the data.";

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));
            env.CheckUserArg(args.Trainer.IsGood(), nameof(args.Trainer),
                "Trainer cannot be null. If your model is already trained, please use ScoreTransform instead.");

            var host = env.Register("TrainAndScoreTransform");

            using (var ch = host.Start("Train"))
            {
                ch.Trace("Constructing trainer");
                ITrainer trainer = args.Trainer.CreateInstance(host);
                var customCols = TrainUtils.CheckAndGenerateCustomColumns(env, args.CustomColumn);
                string feat;
                string group;
                var data = CreateDataFromArgs(ch, input, args, out feat, out group);
                var predictor = TrainUtils.Train(host, ch, data, trainer, args.Trainer.Kind, null,
                    args.Calibrator, args.MaxCalibrationExamples, null);

                ch.Done();

                return ScoreUtils.GetScorer(args.Scorer, predictor, input, feat, group, customCols, env, data.Schema);
            }
        }

        public static RoleMappedData CreateDataFromArgs<TSigTrainer>(IExceptionContext ectx, IDataView input,
            ArgumentsBase<TSigTrainer> args)
        {
            string feat;
            string group;
            return CreateDataFromArgs(ectx, input, args, out feat, out group);
        }

        private static RoleMappedData CreateDataFromArgs<TSigTrainer>(IExceptionContext ectx, IDataView input,
            ArgumentsBase<TSigTrainer> args, out string feat, out string group)
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
